"""E13d. Where is the consequence lost inside the world? (E13b/c: no world predicts DO / place consequences, even on training windows;
an MLP on the INPUT faced token + action predicts the changed tile's next class at 0.977 on held-out windows.)

Stage probe on the world's own backbone output h (the normalized per-tile hidden state the output head reads), at the faced cell
of DO / place transitions, teacher-forced pool windows as consfit.py: a linear classifier (17 classes, changed transitions
weighted x5, AdamW, 60 epochs) fitted on h of 12,000 training windows, scored on tworld's 2,048 held-out windows.
Same protocol on the INPUT (faced token + action one-hot) as the reference. Reported: changed-transition next-class accuracy and
unchanged accuracy, per stage.
Reading, declared before running (per world):
  backbone_encodes  h-probe changed accuracy >= 0.8: the backbone computes the consequence and the output head fails to
                    express it (readout bottleneck)
  backbone_misses   h-probe changed accuracy <= 0.3: the backbone never computes it (learning-signal problem)
Usage: stageprobe.py <world.pt> ... -> evals/stageprobe_<name>.json
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

FACED = {0: 30, 1: 32, 2: 22, 3: 40}
ACT = (5, 7, 8, 9, 10)


def fit_eval(Xtr, Ytr, CHtr, Xte, Yte, CHte, seed=0):
    torch.manual_seed(seed)
    net = nn.Linear(Xtr.shape[1], 17)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    w = torch.where(CHtr, 5.0, 1.0)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    for _ in range(60):
        perm = torch.randperm(len(Ytr))
        for j in range(0, len(perm), 256):
            b = perm[j:j + 256]
            loss = (F.cross_entropy(net(Xtr[b]), Ytr[b], reduction="none") * w[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = net(Xte).argmax(-1)
    return {"changed_acc": float((p == Yte)[CHte].float().mean()), "unchanged_acc": float((p == Yte)[~CHte].float().mean())}


@torch.no_grad()
def collect(world, pool, rows, probes, device, config):
    from d4mj.train import autocast_context
    H, INP, Y, CH = [], [], [], []
    for i in range(0, len(rows), 64):
        r = rows[i:i + 64]
        s = pool["tokens"][r].float(); a = pool["actions"][r]; alive = pool["alive"][r]
        s_in = s.to(device)
        if world.head == "categorical":
            from tworld import quantize
            s_in = world.codes[quantize(s_in, world.codes)]
        with autocast_context(config):
            _, h, _ = world(s_in, F.pad(a, (0, 1)).to(device))
        h = h.float().cpu()                                                          # [b,6,81,D]
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
            H.append(h[idx, t, cell]); INP.append(torch.cat([x0, act], 1)); Y.append(c1); CH.append(ch)
    return torch.cat(H), torch.cat(INP), torch.cat(Y), torch.cat(CH)


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
        Htr, Itr, Ytr, CHtr = collect(world, pool, train, probes, device, config)
        Hte, Ite, Yte, CHte = collect(world, pool, held, probes, device, config)
        res = {"world": name, "n": {"train": len(Ytr), "train_changed": int(CHtr.sum()), "held": len(Yte), "held_changed": int(CHte.sum())},
               "input_probe_linear": fit_eval(Itr, Ytr, CHtr, Ite, Yte, CHte),
               "backbone_h_probe_linear": fit_eval(Htr, Ytr, CHtr, Hte, Yte, CHte)}
        ca = res["backbone_h_probe_linear"]["changed_acc"]
        res["readings"] = {"backbone_encodes": ca >= 0.8, "backbone_misses": ca <= 0.3}
        (out_dir / f"stageprobe_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
