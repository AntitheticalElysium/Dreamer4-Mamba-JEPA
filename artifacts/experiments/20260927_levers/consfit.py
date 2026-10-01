"""E13b. Are interaction consequences learned at all? Underfitting (wrong even on TRAINING transitions) or generalization (right on
training, wrong on held-out)? And how much learning signal do they carry? (E11o: the decision-flipping tile is a DO / place
consequence mispredicted on the faced tile; E13 part C asks whether that error exists from true inputs.)

Data: spatial_pool_v1, the exact windows the worlds were trained on (6 layer-normed Raw token frames, 5 actions): 4,096 training
windows (seeded sample of tworld's training rows) and tworld's 2,048 held-out main windows. Teacher-forced one-step predictions
at positions 0-4 (the world's own training input).
Per transition whose action is DO (5) or place (7-10), next frame alive: the faced cell (teval facing probe on the player token
of frame t), its tile class before / after (teval tile probe on the TRUE tokens of frames t / t+1) and predicted (probe on the
prediction). changed = class before != class after.
  changed:   caught (pred == after) / copied (pred == before) / other
  unchanged: right (pred == after) / hallucinated (pred != after)
Also: the share of all (window, position) transitions that are DO / place with a changed faced tile (consequence density), and
the share of the teacher-forced L1 objective (summed over all tokens and positions) carried by those changed faced tiles.
Probe-noise guard: "probe" changed = class before != after; "strict" also requires the true token to move more than the 99th
percentile of the probe-unchanged transitions' token change (spurious probe flips would otherwise look like copied changes).
Readings use the strict definition. Declared before running (per world):
  underfit          caught rate on TRAINING changed transitions < 0.5
  generalization    caught rate on training >= 0.5 and held-out caught rate <= training - 0.2
Usage: consfit.py <world.pt> ... -> evals/consfit_<name>.json
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import teval as T  # noqa: E402

FACED = {0: 30, 1: 32, 2: 22, 3: 40}
ACT = (5, 7, 8, 9, 10)


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as Sp
    from tworld import POOLS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, train_roots, train_seeds = T.split()
    probes = T.Probes(T.build_cache("raw", device), meta, train_roots, train_seeds)
    pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
    main_rows = torch.where(~pool["terminal"])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    train_rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
    train_rows = train_rows[torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(20261001))[:4096]]
    out_dir = HERE / "evals"
    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        res = {"world": name}
        for split, rows in (("train", train_rows), ("held", held)):
            recs, n_trans, l1_total = [], 0, 0.0
            err_cells = []
            for i in range(0, len(rows), 64):
                r = rows[i:i + 64]
                s = pool["tokens"][r].float()                                       # [b,6,81,192]
                a = pool["actions"][r]; alive = pool["alive"][r]
                s_in = s.to(device)
                if world.head == "categorical":                                       # trained on quantized inputs (teval.step)
                    from tworld import quantize
                    s_in = world.codes[quantize(s_in, world.codes)]
                with autocast_context(config):
                    pred = world(s_in, F.pad(a, (0, 1)).to(device))[0].float().cpu()   # pred[:,t] -> frame t+1
                err = (pred[:, :5] - s[:, 1:]).abs().sum(-1)                         # [b,5,81] L1 per token
                ok = alive[:, 1:]
                l1_total += float((err.sum(-1) * ok).sum()); n_trans += int(ok.sum())
                facing = probes.facing(s[:, :5, 31].flatten(0, 1)).argmax(-1).view(len(r), 5)
                for t in range(5):
                    m = torch.isin(a[:, t], torch.tensor(ACT)) & ok[:, t]
                    for j in torch.nonzero(m)[:, 0].tolist():
                        cell = FACED[int(facing[j, t])]
                        x0, x1, xp = s[j, t, cell], s[j, t + 1, cell], pred[j, t, cell]
                        cls = probes.tile(torch.stack([x0, x1, xp])).argmax(-1).tolist()
                        recs.append((cls[0], cls[1], cls[2], float(((x1 - x0) ** 2).sum()), float(((xp - x1) ** 2).sum()),
                                     float(((xp - x0) ** 2).sum()), float(err[j, t, cell]), int(a[j, t])))
            R_ = torch.tensor([[x[0], x[1], x[2]] for x in recs]); D = torch.tensor([[x[3], x[4], x[5], x[6]] for x in recs])
            before, after, pc = R_[:, 0], R_[:, 1], R_[:, 2]
            d_true, d_pa, d_pb, l1c = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
            probe_changed = before != after
            cut = float(d_true[~probe_changed].quantile(0.99)) if (~probe_changed).any() else 0.0
            out = {"act_transitions": len(recs), "transitions": n_trans, "strict_cut_token_change": cut,
                   "d_true_median": {"probe_changed": float(d_true[probe_changed].median()) if probe_changed.any() else None,
                                     "probe_unchanged": float(d_true[~probe_changed].median())}}
            for label, ch in (("probe", probe_changed), ("strict", probe_changed & (d_true > cut))):
                un = ~probe_changed
                n_ch, n_un = int(ch.sum()), int(un.sum())
                out[label] = {"changed": n_ch, "unchanged": n_un,
                              "caught_rate": float(((pc == after) & ch).sum() / max(n_ch, 1)),
                              "copied_rate": float(((pc == before) & ch).sum() / max(n_ch, 1)),
                              "closer_to_after_rate": float(((d_pa < d_pb) & ch).sum() / max(n_ch, 1)),
                              "hallucinated_rate": float(((pc != after) & un).sum() / max(n_un, 1)),
                              "consequence_density": n_ch / max(n_trans, 1),
                              "objective_share_of_changed_faced_tiles": float(l1c[ch].sum()) / max(l1_total, 1e-9)}
            by_action = {}
            for act in ACT:
                m = D.new_tensor([x[7] == act for x in recs]).bool()
                ch = probe_changed & (d_true > cut) & m
                by_action[str(act)] = {"n": int(m.sum()), "changed": int(ch.sum()),
                                       "caught_rate": float(((pc == after) & ch).sum() / max(int(ch.sum()), 1))}
            out["by_action_strict"] = by_action
            res[split] = out
        tr, he = res["train"]["strict"], res["held"]["strict"]
        res["readings"] = {"underfit": tr["caught_rate"] < 0.5,
                           "generalization": tr["caught_rate"] >= 0.5 and he["caught_rate"] <= tr["caught_rate"] - 0.2}
        (out_dir / f"consfit_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({"world": name, "train": res["train"]["strict"], "held": res["held"]["strict"],
                          "train_probe": res["train"]["probe"], **res["readings"]}), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
