"""Is the precursor observable? A visible / hidden ablation on identical roots.

Four representations of the same state, one probe topology (the repo's
512->64->17 GELU all-action probe), one whole-seed split, one label:

  visible              what the renderer draws
  visible + timing     plus the four player counters and per-mob attack_cooldown
  full                 plus everything else hidden (mob health, absolute position,
                       timestep)
  frozen latent        the encoder's z at the same root, the bridge to our models

`action_only` is the control. The reading is fixed before the numbers: full high and
visible low with timing restoring it means partial observability; visible high with
frozen latent low means the information is on screen and the representation or
predictor loses it; all low means the features are inadequate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_damage_classifier import auc, interval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMING = list(range(4)) + list(range(7, 31, 2))     # counters, then attack_cooldowns


def seed_split(seed: int) -> str:
    draw = int.from_bytes(hashlib.sha256(f"paired-seed:{seed}".encode()).digest()[:8],
                          "little") % 10
    return "fit" if draw < 7 else ("tune" if draw < 8 else "test")


class Probe(nn.Module):
    def __init__(self, width: int, actions: int = 17, bias_only: bool = False):
        super().__init__()
        self.bias_only = bias_only
        if bias_only:
            self.bias = nn.Parameter(torch.zeros(actions))
        else:
            self.net = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, actions))

    def forward(self, x):
        return self.bias[None].expand(len(x), -1) if self.bias_only else self.net(x)


def run(name, x, labels, splits, seed=0):
    fit, tune, test = (splits == s for s in ("fit", "tune", "test"))
    if x is not None:
        mean, std = x[fit].mean(0, keepdim=True), x[fit].std(0, keepdim=True).clamp(min=1e-6)
        x = ((x - mean) / std).to(DEVICE)
        width = x.shape[1]
    else:
        width = 1
    y = labels.to(DEVICE)
    torch.manual_seed(seed)
    probe = Probe(width, bias_only=x is None).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    weight = torch.tensor(float((y[fit] <= 0).sum() / y[fit].sum().clamp(min=1)), device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    index = np.where(fit)[0]
    best = {"auc": -1.0, "state": None}
    dummy = torch.zeros(len(labels), 1, device=DEVICE)
    source = x if x is not None else dummy
    for epoch in range(150):
        probe.train()
        order = np.random.default_rng(seed + epoch).permutation(index)
        for lo in range(0, len(order), 256):
            batch = torch.from_numpy(order[lo : lo + 256]).to(DEVICE)
            opt.zero_grad()
            criterion(probe(source[batch]), y[batch]).backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            scores = probe(source[torch.from_numpy(np.where(tune)[0]).to(DEVICE)]).cpu().numpy()
        truth = labels[tune].numpy()
        values = np.array([auc(scores[i], truth[i]) for i in range(len(scores))])
        values = values[~np.isnan(values)]
        if values.mean() > best["auc"]:
            best = {"auc": float(values.mean()),
                    "state": {k: v.clone() for k, v in probe.state_dict().items()}}
    probe.load_state_dict(best["state"])
    probe.eval()
    with torch.no_grad():
        scores = probe(source[torch.from_numpy(np.where(test)[0]).to(DEVICE)]).cpu().numpy()
    truth = labels[test].numpy()
    values = np.array([auc(scores[i], truth[i]) for i in range(len(scores))])
    values = values[~np.isnan(values)]
    a, (lo, hi) = interval(values, 17)
    print(f"  {name:<26}{a:>9.4f}  [{lo:.4f}, {hi:.4f}]   roots {len(values)}  "
          f"tune {best['auc']:.4f}")
    return {"within_auc": a, "ci": [lo, hi], "roots": len(values), "tune": best["auc"]}


def main() -> None:
    rows = torch.load(HERE / "state_features/features.pt", weights_only=False)
    splits = np.array([seed_split(r["seed"]) for r in rows])
    labels = torch.stack([r["label"] for r in rows])
    visible = torch.stack([r["visible"] for r in rows])
    hidden = torch.stack([r["hidden"] for r in rows])
    latent = torch.stack([r["latent"] for r in rows])
    print(f"{len(rows)} hazard roots  fit {int((splits=='fit').sum())} "
          f"tune {int((splits=='tune').sum())} test {int((splits=='test').sum())}")
    print(f"visible dim {visible.shape[1]}, hidden dim {hidden.shape[1]}, "
          f"latent dim {latent.shape[1]}")
    print(f"\n{'representation':<26}{'within AUC':>9}  {'95% CI':<20}")
    results = {
        "action_only": run("action only (control)", None, labels, splits, 1),
        "visible": run("visible", visible, labels, splits, 2),
        "visible_plus_timing": run("visible + timing/counters",
                                   torch.cat([visible, hidden[:, TIMING]], 1),
                                   labels, splits, 3),
        "full": run("full simulator state", torch.cat([visible, hidden], 1),
                    labels, splits, 4),
        "frozen_latent": run("frozen Z* at the root", latent, labels, splits, 5),
    }
    (HERE / "state_features/probe_report.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
