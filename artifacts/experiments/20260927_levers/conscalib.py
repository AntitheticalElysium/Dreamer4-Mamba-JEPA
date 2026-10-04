"""E13e. Present-but-below-threshold, or absent? (E13d: the backbone state at the faced tile linearly encodes the changed tile's next
class at 0.79-0.86 with class-weighted probes, yet every head copies.)

(1) Categorical world (categorical_raw_teacher_s7_K4096): on DO / place transitions of training and held-out pool windows, the
    world's own predicted probability of the code nearest the TRUE next token vs the code nearest the CURRENT token, split by
    changed / unchanged (consfit's strict definition), and the rank of the true next code.
(2) Every listed world: an UNWEIGHTED logistic probe "will the faced tile change" on the backbone state h at the faced tile
    (fitted on training windows, scored on held-out): AUC, and the mean predicted P(change) on truly changed and unchanged
    transitions. Same probe on the input (faced token + action) as the reference.
Reading, declared before running:
  below_threshold   (categorical) mean P(true next code) on changed transitions in [0.1, 0.5) and ranked first < 50%, OR
                    (h probe) AUC >= 0.9 with mean P(change | changed) < 0.5: the change is represented but loses to "no change"
                    at decoding
  absent            (categorical) mean P(true next code) on changed < 0.05
Usage: conscalib.py <world.pt> ... -> evals/conscalib_<name>.json
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import teval as T  # noqa: E402
from stageprobe import ACT, FACED  # noqa: E402


def auc(score, label):
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return None
    allv = torch.cat([pos, neg]); ranks = allv.argsort().argsort().float() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logistic(Xtr, ytr, Xte, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    net = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(60):
        perm = torch.randperm(len(ytr))
        for j in range(0, len(perm), 256):
            b = perm[j:j + 256]
            loss = F.binary_cross_entropy_with_logits(net(Xtr[b])[:, 0], ytr[b].float())
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(net(Xte)[:, 0])


@torch.no_grad()
def collect(world, pool, rows, probes, device, config):
    from d4mj.train import autocast_context
    from tworld import quantize
    H, INP, CH, PN, PC, RK = [], [], [], [], [], []
    for i in range(0, len(rows), 64):
        r = rows[i:i + 64]
        s = pool["tokens"][r].float(); a = pool["actions"][r]; alive = pool["alive"][r]
        s_in = s.to(device)
        cat = world.head == "categorical"
        if cat:
            s_in = world.codes[quantize(s_in, world.codes)]
        with autocast_context(config):
            out = world(s_in, F.pad(a, (0, 1)).to(device))
        h = out[1].float().cpu()
        logits = out[2].float().cpu() if cat else None
        face = probes.facing(s[:, :5, 31].flatten(0, 1)).argmax(-1).view(len(r), 5)
        for t in range(5):
            m = torch.isin(a[:, t], torch.tensor(ACT)) & alive[:, t + 1]
            idx = torch.nonzero(m)[:, 0]
            if not len(idx):
                continue
            cell = torch.tensor([FACED[int(f)] for f in face[idx, t]])
            x0, x1 = s[idx, t, cell], s[idx, t + 1, cell]
            c0, c1 = probes.tile(x0).argmax(-1), probes.tile(x1).argmax(-1)
            ch = (c0 != c1) & (((x1 - x0) ** 2).sum(-1) > 120)
            act = F.one_hot(torch.tensor([ACT.index(int(v)) for v in a[idx, t]]), 5).float()
            H.append(h[idx, t, cell]); INP.append(torch.cat([x0, act], 1)); CH.append(ch)
            if cat:
                lg = logits[idx, t, cell]                                            # [n,K]
                p = lg.softmax(-1)
                k1 = quantize(x1.to(device), world.codes).cpu(); k0 = quantize(x0.to(device), world.codes).cpu()
                PN.append(p.gather(1, k1[:, None])[:, 0]); PC.append(p.gather(1, k0[:, None])[:, 0])
                RK.append((lg > lg.gather(1, k1[:, None])).sum(1))
    o = {"H": torch.cat(H), "INP": torch.cat(INP), "CH": torch.cat(CH)}
    if PN:
        o.update({"PN": torch.cat(PN), "PC": torch.cat(PC), "RK": torch.cat(RK)})
    return o


def main():
    from d4mj.config import config_from_dict
    import spatial as Sp
    from tworld import POOLS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, train_roots, train_seeds = T.split()
    probes = T.Probes(T.build_cache("raw", torch.device("cpu")), meta, train_roots, train_seeds)
    pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
    main_rows = torch.where(~pool["terminal"])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    train = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
    train = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(2))[:12000]]
    out_dir = HERE / "evals"
    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        tr, te = collect(world, pool, train, probes, device, config), collect(world, pool, held, probes, device, config)
        res = {"world": name}
        for lab, key in (("input", "INP"), ("backbone_h", "H")):
            p = logistic(tr[key], tr["CH"], te[key])
            res[f"{lab}_change_probe"] = {"auc": auc(p, te["CH"]), "mean_p_changed": float(p[te["CH"]].mean()),
                                          "mean_p_unchanged": float(p[~te["CH"]].mean()),
                                          "share_p_over_half_changed": float((p[te["CH"]] > 0.5).float().mean())}
        if "PN" in te:
            for split, d in (("train", tr), ("held", te)):
                ch = d["CH"]
                res[f"categorical_{split}"] = {
                    "changed": {"mean_p_true_next": float(d["PN"][ch].mean()), "mean_p_current": float(d["PC"][ch].mean()),
                                "true_next_ranked_first": float((d["RK"][ch] == 0).float().mean()),
                                "median_rank_true_next": float(d["RK"][ch].float().median())},
                    "unchanged": {"mean_p_true_next": float(d["PN"][~ch].mean())}}
        rd = {}
        hb = res["backbone_h_change_probe"]
        rd["h_below_threshold"] = hb["auc"] is not None and hb["auc"] >= 0.9 and hb["mean_p_changed"] < 0.5
        if "categorical_held" in res:
            c = res["categorical_held"]["changed"]
            rd["categorical_below_threshold"] = 0.1 <= c["mean_p_true_next"] < 0.5 and c["true_next_ranked_first"] < 0.5
            rd["categorical_absent"] = c["mean_p_true_next"] < 0.05
        res["readings"] = rd
        (out_dir / f"conscalib_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
