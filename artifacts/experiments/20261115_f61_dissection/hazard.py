"""F6.1 dissection, check 3 (PLAN.md): is the late waking a rise drawn too rarely, or drawn at the wrong step? Saved
quiet-bank rollouts only (CPU): per scene the simulator's rise step d (energy 8 -> 9; deterministic in the phase), per
generated draw its first energy-9 step; the generated rise hazard per asleep step at energy 8, by step relative to d;
the rank correlation of the scenes' generated and true rise steps.

    python -m artifacts.experiments.20261115_f61_dissection.hazard
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
BANKS = {"quiet_bank": {"F6.1": EXP / "20261114_p1f61/readings/followups/sleep/run.pt",
                        "F6": EXP / "20261113_p1f6/readings/followups/sleep/run.pt",
                        "F5": EXP / "20261111_p1f5/readings/followups/sleep/run.pt"},
         "repeat_sleep_bank": {"F6.1": HERE / "run/sleep_paid/F6.1/run.pt", "F6": HERE / "run/sleep_paid/F6/run.pt"}}


def first(mask, h):
    return np.where(mask.any(-1), mask.argmax(-1), h + 1)


def read(path):
    z = torch.load(path, map_location="cpu", weights_only=False)
    a, b, h = z["generated"], z["truth"], z["H"]
    d = first(b[..., 0] >= 9, h)                                       # [scenes, draws]
    assert (d == d[:, :1]).all(), "the simulator's rise step should not vary across draws"
    d = d[:, 0]
    live = a[..., 3].astype(bool) & ~a[..., 2].astype(bool)
    g = first(live & (a[..., 0] >= 9), h)                              # first generated energy 9
    at8 = np.ones(a.shape[:3], bool)                                   # still at 8 before step k (the source is at 8)
    at8[:, :, 1:] = np.cumprod(live[:, :, :-1] & (a[:, :, :-1, 0] < 9) & (a[:, :, :-1, 1] > .5), 2).astype(bool)
    rise = live & (a[..., 0] >= 9)
    rel = np.arange(h)[None, None] - d[:, None, None]
    out = {"scenes": len(d), "draws": int(a.shape[1]), "true_rise_step_mean": float(d.mean()),
           "generated_rise_step_median_of_scene_medians": float(np.median(np.median(np.minimum(g, h + 1), 1))),
           "hazard_per_step_at_8": float(rise[at8].mean()),
           "needed_flat_hazard": float(1 / (d.mean() + 1)),
           "hazard_by_step_relative_to_due": {}, "rise_exactly_at_due": float((g == d[:, None]).mean()),
           "rise_within_1_of_due": float((np.abs(g - d[:, None]) <= 1).mean()),
           "rise_before_due": float((g < d[:, None]).mean()),
           "no_rise_within_H": float((g > h).mean()),
           "spearman_true_vs_generated_median": float(spearmanr(d, np.median(np.minimum(g, h + 1), 1)).statistic)}
    for name, m in (("before_due", rel < 0), ("at_due", rel == 0), ("1-3_after", (rel >= 1) & (rel <= 3)), ("4+_after", rel >= 4)):
        sel = at8 & m
        out["hazard_by_step_relative_to_due"][name] = {"hazard": float(rise[sel].mean()) if sel.any() else None, "at_risk": int(sel.sum())}
    return out


def main():
    res = {bank: {run: read(p) for run, p in paths.items() if p.exists()} for bank, paths in BANKS.items()}
    for bank, rows in res.items():
        for run, r in rows.items():
            print(bank, run, json.dumps(r), flush=True)
    (HERE / "hazard.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
