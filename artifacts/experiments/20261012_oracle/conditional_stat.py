"""CPU diagnostic: average true successors only when the oracle's full mob layout agrees.

Primary conditioning is at depth 1: same simulator root and action, independent
environment keys, identical complete 63-cell next-layout in the four channels
supplied by OracleCond. No model training or inference. Compare the empirical
mean/median with individual real frames using the frozen saved mob reader.
Groups need >=2 members; their selection and finite-five-future coverage are
reported. Bootstrap is by root, preserving all groups and branch weights.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/eda/oracle_v1"
MAPC = [c for c in range(63) if c != 31]
NAMES = ("zombie", "skeleton", "cow", "arrow")


def main():
    torch.set_num_threads(4)
    meta = torch.load(ROOT / "artifacts/eda/diagnosis_rollouts_v1/meta.pt", weights_only=False)
    saved = torch.load(OUT / "snap_bank_readers.pt", weights_only=False)
    readers = saved["heads"]["mob"]
    threshold = torch.tensor(saved["thresholds"])
    R = len(meta["seed"])
    futures = np.memmap(ROOT / "artifacts/eda/stochdiag_v1/fut5.f16", dtype=np.float16,
                        mode="r", shape=(R, 5, 16, 81, 192))
    test = ~torch.isin(meta["seed"], saved["train_seeds"])
    alive = ~meta["future_dead"][:, :, 0].bool()
    vis = meta["future_visible"][:, :, 0].float()
    flags = vis[..., 1071:1512].reshape(R, 5, 63, 7) > 0
    # Native visible order: zombie,cow,skeleton,arrow orientations (four).
    layouts = torch.stack((flags[..., 0], flags[..., 1], flags[..., 2], flags[..., 3:].any(-1)), -1)
    truth = layouts[..., [0, 2, 1, 3]][..., MAPC, :]
    rows = torch.zeros(R, 3, 4, 4, dtype=torch.float64)  # mode, mob, truth/pred/TP/FP
    groups = []

    def decode(x):
        c = x[:, MAPC].double()
        z = (c - readers["mu"].double()) / readers["sd"].double()
        z = torch.cat((z, torch.ones(*z.shape[:-1], 1, dtype=z.dtype)), -1)
        return z @ readers["w"].double() > threshold

    for r in torch.where(test)[0].tolist():
        buckets = {}
        for s in range(5):
            if alive[r, s]:
                key = layouts[r, s].numpy().tobytes()
                buckets.setdefault(key, []).append(s)
        for ss in buckets.values():
            if len(ss) < 2:
                continue
            y = torch.from_numpy(np.array(futures[r, ss, 0])).float()
            t = truth[r, ss]
            assert (t == t[:1]).all()
            tokens = (y, y.mean(0, keepdim=True), y.median(0, keepdim=True).values)
            for m, tok in enumerate(tokens):
                dec = decode(tok)
                lab = t if m == 0 else t[:1]
                weight = 1 if m == 0 else len(ss)
                rows[r, m, :, 0] += lab.sum((0, 1)) * weight
                rows[r, m, :, 1] += dec.sum((0, 1)) * weight
                rows[r, m, :, 2] += (lab & dec).sum((0, 1)) * weight
                rows[r, m, :, 3] += (~lab & dec).sum((0, 1)) * weight
            groups.append({"root": r, "seed": int(meta["seed"][r]), "members": ss,
                           "same_all_seven_flags": bool((flags[r, ss] == flags[r, ss[:1]]).all()),
                           "mob_count": t[0].sum(0).tolist()})
    totals = rows.sum(0)
    modes = ("real", "conditional_mean", "conditional_median")

    def metrics(v):
        den = v[..., 0].clamp_min(1)
        return torch.stack((v[..., 1] / den, v[..., 2] / den, v[..., 3] / den), -1)

    summary = {mode: {name: {"true": int(totals[m, j, 0]), "predicted": int(totals[m, j, 1]),
                            "true_positive": int(totals[m, j, 2]), "false_positive": int(totals[m, j, 3]),
                            "census_ratio": float(metrics(totals)[m, j, 0]),
                            "recall": float(metrics(totals)[m, j, 1]),
                            "fp_per_true": float(metrics(totals)[m, j, 2])}
                      for j, name in enumerate(NAMES)} for m, mode in enumerate(modes)}
    included = torch.tensor(sorted({g["root"] for g in groups}))
    rng = torch.Generator().manual_seed(20261012)
    idx = torch.randint(len(included), (4000, len(included)), generator=rng)
    boot = metrics(rows[included][idx].sum(1))
    contrasts = {}
    for m in (1, 2):
        delta = metrics(totals)[m] - metrics(totals)[0]
        ci = torch.quantile(boot[:, m] - boot[:, 0], torch.tensor([.025, .975], dtype=torch.float64), dim=0)
        contrasts[modes[m] + "_minus_real"] = {name: {measure: {"delta": float(delta[j, k]),
                                                               "ci95": [float(ci[0, j, k]), float(ci[1, j, k])]}
                                                     for k, measure in enumerate(("census_ratio", "recall", "fp_per_true"))}
                                                for j, name in enumerate(NAMES)}
    report = {"schema": "oracle_conditional_stat_v1", "depth": 1, "test_roots": int(test.sum()),
              "alive_test_branches": int(alive[test].sum()), "groups_ge2": len(groups),
              "included_roots": len(included), "included_branches": sum(len(g["members"]) for g in groups),
              "groups_matching_all_seven_flags": sum(g["same_all_seven_flags"] for g in groups),
              "group_sizes": {str(n): sum(len(g["members"]) == n for g in groups) for n in range(2, 6)},
              "group_mob_presence": {name: sum(g["mob_count"][j] > 0 for g in groups) for j, name in enumerate(NAMES)},
              "metrics": summary, "contrasts": contrasts, "groups": groups,
              "limits": ["Empirical groups of 2-5 samples, not population conditional optima.",
                         "Only exact-layout groups with repeats are covered; singleton outcomes excluded.",
                         "Means and medians are evaluated, not the complete D-loss optimum."],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "conditional_stat.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
