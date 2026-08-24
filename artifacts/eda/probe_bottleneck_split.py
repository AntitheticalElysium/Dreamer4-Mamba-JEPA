"""Split the bottleneck: is it the rank reduction, the squash, or the orientation?

Everything runs from the frozen 8,192-dim pre-bottleneck features already on disk,
on the same roots, split and probe as A-F. Five compressions of the same features,
all sharing production's per-token structure where the comparison calls for it:

  full              the 8,192-dim features, no compression
  pre-tanh          production's Linear(256->16), squash removed
  Z*                production's Linear(256->16) then tanh
  matched random    an untrained Linear(256->16) then tanh, production's exact shape
  task-learned      the same 256->16 shape, trained on fit roots to preserve damage

The last is a positive control: it asks whether this exact 512-dim interface *can*
carry the signal, not whether the reconstruction objective chose to make it.
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
from probe_observability import Probe, seed_split
from probe_prebottleneck import fit_probe

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.representation import Encoder

DEVICE = "cuda"
config = Config()
N_LATENT, D_MODEL, D_BOTTLE = config.n_latents, config.d_model_encoder, config.d_bottleneck


class LearnedBottleneck(nn.Module):
    """Production's shape -- per-token Linear(256->16) then tanh -- then the probe."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.compress = nn.Linear(D_MODEL, D_BOTTLE)
        self.probe = Probe(N_LATENT * D_BOTTLE)

    def forward(self, pre):
        z = torch.tanh(self.compress(pre.reshape(-1, N_LATENT, D_MODEL)))
        return self.probe(z.reshape(len(pre), -1))


def fit_learned(pre, labels, splits, seed, epochs=150):
    fit, tune, test = (splits == s for s in ("fit", "tune", "test"))
    x, y = pre.to(DEVICE), labels.to(DEVICE)
    model = LearnedBottleneck(seed).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    weight = torch.tensor(float((y[fit] <= 0).sum() / y[fit].sum().clamp(min=1)), device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    index = np.where(fit)[0]
    best = {"auc": -1.0, "state": None}
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(index)
        for lo in range(0, len(order), 256):
            batch = torch.from_numpy(order[lo : lo + 256]).to(DEVICE)
            opt.zero_grad()
            criterion(model(x[batch]), y[batch]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            scores = model(x[torch.from_numpy(np.where(tune)[0]).to(DEVICE)]).cpu().numpy()
        truth = labels[tune].numpy()
        values = np.array([auc(scores[i], truth[i]) for i in range(len(scores))])
        values = values[~np.isnan(values)]
        if values.mean() > best["auc"]:
            best = {"auc": float(values.mean()),
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"]); model.eval()
    with torch.no_grad():
        scores = model(x[torch.from_numpy(np.where(test)[0]).to(DEVICE)]).cpu().numpy()
    truth = labels[test].numpy()
    values = np.array([auc(scores[i], truth[i]) for i in range(len(scores))])
    keep = ~np.isnan(values)
    return values[keep], keep, best["auc"]


def main() -> None:
    rows = torch.load(HERE / "state_features/prebottleneck.pt", weights_only=False)
    splits = np.array([seed_split(r["seed"]) for r in rows])
    labels = torch.stack([r["label"] for r in rows])
    pre = torch.stack([r["pre"] for r in rows])
    tokens = pre.reshape(len(pre), N_LATENT, D_MODEL)

    encoder = Encoder(config)
    load(ROOT / "artifacts/stage_a_terminalfix/phase1a.pt", config, part0=encoder)
    encoder.eval()
    with torch.no_grad():
        u = encoder.bottleneck(tokens)                 # production projection, pre-squash
        z = torch.tanh(u)
        torch.manual_seed(20260819)
        random_projection = nn.Linear(D_MODEL, D_BOTTLE)
        z_random = torch.tanh(random_projection(tokens))
    flat = lambda t: t.reshape(len(t), -1)

    print(f"{len(rows)} roots  fit {int((splits=='fit').sum())} "
          f"tune {int((splits=='tune').sum())} test {int((splits=='test').sum())}")
    magnitude = u.abs()
    print(f"\npre-squash |u|: mean {magnitude.mean():.3f}  median {magnitude.median():.3f}  "
          f"p99 {magnitude.flatten().quantile(0.99):.3f}")
    for limit in (1, 2, 3):
        print(f"  coordinates with |u| > {limit}: {(magnitude > limit).float().mean():.2%}")
    derivative = 1 - torch.tanh(u) ** 2
    print(f"  mean tanh derivative: {derivative.mean():.4f}  "
          f"share below 0.1 (saturated): {(derivative < 0.1).float().mean():.2%}")

    arms = {
        "full pre-bottleneck (8192)": (flat(tokens), 6),
        "pre-tanh, production projection (512)": (flat(u), 8),
        "Z* = tanh(production projection) (512)": (flat(z), 2),
        "matched random 256->16 + tanh (512)": (flat(z_random), 9),
    }
    values, results = {}, {}
    print(f"\n{'compression':<44}{'within AUC':>11}  {'95% CI':<20}")
    for name, (x, seed) in arms.items():
        v, keep, tune = fit_probe(x, labels, splits, seed)
        values[name] = (v, keep)
        a, (lo, hi) = interval(v, 17)
        results[name] = {"within_auc": a, "ci": [lo, hi], "tune": tune, "dim": int(x.shape[1])}
        print(f"  {name:<42}{a:>11.4f}  [{lo:.4f}, {hi:.4f}]   tune {tune:.4f}")

    v, keep, tune = fit_learned(flat(tokens), labels, splits, 10)
    values["task-learned 256->16 + tanh (512)"] = (v, keep)
    a, (lo, hi) = interval(v, 17)
    results["task-learned 256->16 + tanh (512)"] = {
        "within_auc": a, "ci": [lo, hi], "tune": tune, "dim": 512}
    print(f"  {'task-learned 256->16 + tanh (512)':<42}{a:>11.4f}  [{lo:.4f}, {hi:.4f}]"
          f"   tune {tune:.4f}")

    base_v, base_keep = values["Z* = tanh(production projection) (512)"]
    print()
    for name, (v, keep) in values.items():
        if name.startswith("Z*"):
            continue
        both = base_keep & keep
        d, (lo, hi) = interval(v[both[keep]] - base_v[both[base_keep]], 23)
        print(f"  paired {name:<44} minus Z*: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        results[name]["paired_minus_zstar"] = {"delta": d, "ci": [lo, hi]}
    (HERE / "state_features/bottleneck_split_report.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
