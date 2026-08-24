"""Pre-bottleneck encoder features against Z*, paired on identical roots.

The probe is the one used for A-F: Linear(d, 64) -> GELU -> Linear(64, 17), AdamW at
1e-3 / 1e-2, 150 epochs, batch 256, checkpoint chosen on the tune split by within-state
AUC. The only concession to dimensionality is a third arm in which the 8192-dim
pre-bottleneck features are carried through a fixed, untrained random projection to
512, so one comparison holds probe input width equal to Z*'s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_damage_classifier import auc, interval
from probe_observability import Probe, seed_split

DEVICE = "cuda"


def fit_probe(x, labels, splits, seed):
    """Identical to `probe_observability.run`, but returning the per-root AUCs."""
    fit, tune, test = (splits == s for s in ("fit", "tune", "test"))
    mean, std = x[fit].mean(0, keepdim=True), x[fit].std(0, keepdim=True).clamp(min=1e-6)
    x = ((x - mean) / std).to(DEVICE)
    y = labels.to(DEVICE)
    torch.manual_seed(seed)
    probe = Probe(x.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    weight = torch.tensor(float((y[fit] <= 0).sum() / y[fit].sum().clamp(min=1)), device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    index = np.where(fit)[0]
    best = {"auc": -1.0, "state": None}
    for epoch in range(150):
        probe.train()
        order = np.random.default_rng(seed + epoch).permutation(index)
        for lo in range(0, len(order), 256):
            batch = torch.from_numpy(order[lo : lo + 256]).to(DEVICE)
            opt.zero_grad()
            criterion(probe(x[batch]), y[batch]).backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            scores = probe(x[torch.from_numpy(np.where(tune)[0]).to(DEVICE)]).cpu().numpy()
        truth = labels[tune].numpy()
        values = np.array([auc(scores[i], truth[i]) for i in range(len(scores))])
        values = values[~np.isnan(values)]
        if values.mean() > best["auc"]:
            best = {"auc": float(values.mean()),
                    "state": {k: v.clone() for k, v in probe.state_dict().items()}}
    probe.load_state_dict(best["state"]); probe.eval()
    with torch.no_grad():
        scores = probe(x[torch.from_numpy(np.where(test)[0]).to(DEVICE)]).cpu().numpy()
    truth = labels[test].numpy()
    values = np.array([auc(scores[i], truth[i]) for i in range(len(scores))])
    keep = ~np.isnan(values)
    return values[keep], keep, best["auc"]


def main() -> None:
    rows = torch.load(HERE / "state_features/prebottleneck.pt", weights_only=False)
    splits = np.array([seed_split(r["seed"]) for r in rows])
    labels = torch.stack([r["label"] for r in rows])
    z = torch.stack([r["z"] for r in rows])
    pre = torch.stack([r["pre"] for r in rows])
    print(f"{len(rows)} roots  fit {int((splits=='fit').sum())} "
          f"tune {int((splits=='tune').sum())} test {int((splits=='test').sum())}")
    print(f"Z* dim {z.shape[1]}, pre-bottleneck dim {pre.shape[1]}")

    generator = torch.Generator().manual_seed(20260819)
    projection = torch.randn(pre.shape[1], z.shape[1], generator=generator) / pre.shape[1] ** 0.5
    arms = {
        "Z* (post-bottleneck, tanh)": (z, 2),
        "pre-bottleneck, full 8192": (pre, 6),
        "pre-bottleneck, random-projected to 512": (pre @ projection, 7),
    }
    values, results = {}, {}
    print(f"\n{'representation':<44}{'within AUC':>11}  {'95% CI':<20}")
    for name, (x, seed) in arms.items():
        v, keep, tune = fit_probe(x, labels, splits, seed)
        values[name] = (v, keep)
        a, (lo, hi) = interval(v, 17)
        results[name] = {"within_auc": a, "ci": [lo, hi], "roots": int(len(v)),
                         "tune": tune, "dim": int(x.shape[1])}
        print(f"  {name:<42}{a:>11.4f}  [{lo:.4f}, {hi:.4f}]   tune {tune:.4f}")

    base_v, base_keep = values["Z* (post-bottleneck, tanh)"]
    print()
    for name in list(arms)[1:]:
        v, keep = values[name]
        both = base_keep & keep
        d, (lo, hi) = interval(v[both[keep]] - base_v[both[base_keep]], 23)
        print(f"  paired {name} minus Z*: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"n={int(both.sum())}")
        results[name]["paired_minus_zstar"] = {"delta": d, "ci": [lo, hi]}
    (HERE / "state_features/prebottleneck_report.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
