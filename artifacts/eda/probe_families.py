"""Does the bottleneck lose control-relevant signal generally, or only damage?

Three consequence families with simulator truth, three compressions each, all from
one frozen feature set and one whole-seed split. The comparison inside a family is
what carries the argument; counts differ between families because action-varying
states do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluate_damage_classifier import auc, interval
from probe_bottleneck_split import fit_learned
from probe_observability import seed_split
from probe_prebottleneck import fit_probe

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.representation import Encoder

config = Config()
N_LATENT, D_MODEL = config.n_latents, config.d_model_encoder

rows = torch.load(HERE / "state_features/union_features.pt", weights_only=False)
splits = np.array([seed_split(r["seed"]) for r in rows])
pre = torch.stack([r["pre"] for r in rows]).float()
z = torch.stack([r["z"] for r in rows])
print(f"{len(rows)} action-varying states; pre {tuple(pre.shape)}, Z* {tuple(z.shape)}")

results = {}
for family in ("damage", "death", "reward"):
    labels = torch.stack([r[f"label_{family}"] for r in rows])
    varying = np.array([bool(l.any() and not l.all()) for l in labels])
    keep = np.where(varying)[0]
    sub_splits, sub_labels = splits[keep], labels[keep]
    counts = {s: int((sub_splits == s).sum()) for s in ("fit", "tune", "test")}
    if counts["test"] < 60 or counts["fit"] < 200:
        print(f"\n{family}: only {counts} -- too few to report")
        continue
    print(f"\n=== {family} ===  roots {len(keep)}  {counts}")
    sub_pre, sub_z = pre[keep], z[keep]
    arms = {}
    v, k, tune = fit_probe(sub_pre, sub_labels, sub_splits, 6)
    arms["pre-bottleneck (8192)"] = (v, k, tune)
    v, k, tune = fit_probe(sub_z, sub_labels, sub_splits, 2)
    arms["Z* production (512)"] = (v, k, tune)
    v, k, tune = fit_learned(sub_pre, sub_labels, sub_splits, 10)
    arms["task-learned 256->16 (512)"] = (v, k, tune)

    base_v, base_k = arms["Z* production (512)"][:2]
    row = {}
    for name, (v, k, tune) in arms.items():
        a, (lo, hi) = interval(v, 17)
        print(f"  {name:<30}{a:>9.4f}  [{lo:.4f}, {hi:.4f}]   tune {tune:.4f}")
        row[name] = {"within_auc": a, "ci": [lo, hi], "tune": tune}
    for name, (v, k, tune) in arms.items():
        if name.startswith("Z*"):
            continue
        both = base_k & k
        d, (lo, hi) = interval(v[both[k]] - base_v[both[base_k]], 23)
        print(f"    paired {name} minus Z*: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        row[name]["paired_minus_zstar"] = {"delta": d, "ci": [lo, hi]}
    row["roots"] = int(len(keep))
    results[family] = row

(HERE / "state_features/family_report.json").write_text(json.dumps(results, indent=2))
print("\nwrote family_report.json")
