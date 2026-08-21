"""Does frozen Z* carry a within-state damage/no-damage distinction that `d` misses?

Feature is the action-induced change x = z_next - z_root. Roots are split whole, by
a hash of their identity, so no action of a test root is ever fitted or selected on.
Three scorers on identical rows: the fixed TRAIN fatality direction (no fitting),
the linear probe, and the existing small 512-64-1 GELU probe.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
DATA = HERE / "latent_forks"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOOT = 2000

prepared = torch.load(
    ROOT / "artifacts/terminal_diversity_v2/preparation/prepared.pt", weights_only=False
)
direction = prepared["direction"].float().flatten()

records = []
for path in sorted(DATA.glob("shard-*.pt")):
    records += torch.load(path, weights_only=False)
print(f"{len(records)} fork roots loaded", flush=True)


def split_of(record) -> str:
    key = f"consequence-probe:{record['shard']}:{record['slot']}:{record['t']}"
    draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % 10
    return "fit" if draw < 6 else ("tune" if draw < 8 else "test")


x, label, root_id, split_id, kind, terminal = [], [], [], [], [], []
for index, record in enumerate(records):
    health = record["health"].numpy()
    dead = record["dead"].numpy()
    lava = record["lava"].numpy()
    positives = (health <= -1) | dead
    negatives = (health >= 0) & ~dead
    if not positives.any() or not negatives.any():
        continue
    near = min(record["mob_distance"].values())
    for a in range(17):
        if not (positives[a] or negatives[a]):
            continue
        x.append(record["z_next"][a] - record["z_root"])
        label.append(bool(positives[a]))
        root_id.append(index)
        split_id.append(split_of(record))
        terminal.append(bool(dead[a]))
        kind.append("lava" if lava[a] else ("mob" if near <= 1 else "other"))

x = torch.stack(x)
label = torch.tensor(label, dtype=torch.float32)
root_id = np.array(root_id)
split_id = np.array(split_id)
kind = np.array(kind)
terminal = np.array(terminal)
usable = np.unique(root_id)
print(f"usable roots (both classes present): {len(usable)} of {len(records)} "
      f"({len(usable)/len(records):.1%})")
print(f"rows {len(x)}, positives {int(label.sum())} ({float(label.mean()):.1%})")
for name in ("fit", "tune", "test"):
    mask = split_id == name
    print(f"  {name:<5} roots {len(np.unique(root_id[mask])):>5}  rows {int(mask.sum()):>6}"
          f"  positives {int(label.numpy()[mask].sum()):>5}")
print("  positive kinds: " + ", ".join(
    f"{k} {int((kind[label.numpy() > 0] == k).sum())}" for k in ("mob", "lava", "other")))
print(f"  positives that are deaths: {int(terminal[label.numpy() > 0].sum())}", flush=True)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    unique, inverse = np.unique(scores, return_inverse=True)
    for value in range(len(unique)):
        tie = inverse == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    pos, neg = labels > 0, labels <= 0
    if not pos.any() or not neg.any():
        return float("nan")
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def within_state(scores: np.ndarray, mask: np.ndarray, subset=None):
    """Per-root AUC and standardized logit contrast, averaged over roots."""
    aucs, contrasts = [], []
    roots = []
    for r in np.unique(root_id[mask]):
        cell = mask & (root_id == r)
        if subset is not None:
            keep = cell & (subset | (label.numpy() <= 0))
            if not (label.numpy()[keep] > 0).any():
                continue
            cell = keep
        y = label.numpy()[cell]
        s = scores[cell]
        if not (y > 0).any() or not (y <= 0).any():
            continue
        aucs.append(auc(s, y))
        scale = s.std() if s.std() > 0 else 1.0
        contrasts.append((s[y > 0].mean() - s[y <= 0].mean()) / scale)
        roots.append(r)
    return np.array(aucs), np.array(contrasts), np.array(roots)


def interval(values: np.ndarray, seed: int = 0):
    if not len(values):
        return float("nan"), (float("nan"), float("nan"))
    generator = np.random.default_rng(seed)
    draws = np.array([values[generator.integers(0, len(values), len(values))].mean()
                      for _ in range(BOOT)])
    return float(values.mean()), (float(np.quantile(draws, 0.025)),
                                  float(np.quantile(draws, 0.975)))


class Probe(nn.Module):
    """The topologies `probe_branched_policy_states.Probe` already uses."""

    def __init__(self, architecture: str, width: int = 512):
        super().__init__()
        if architecture == "linear":
            self.net = nn.Linear(width, 1)
        elif architecture == "small":
            self.net = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        else:
            raise ValueError(architecture)

    def forward(self, value):
        return self.net(value)[:, 0]


def train(architecture: str, seed: int) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    fit, tune = split_id == "fit", split_id == "tune"
    xf, yf = x[fit].to(DEVICE), label[fit].to(DEVICE)
    xt = x[tune].to(DEVICE)
    weight = float((yf <= 0).sum() / max(float(yf.sum()), 1.0))
    model = Probe(architecture).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=DEVICE))
    best = {"auc": -1.0, "state": None, "epoch": -1}
    generator = torch.Generator().manual_seed(seed + 1)
    for epoch in range(120):
        model.train()
        order = torch.randperm(len(xf), generator=generator).to(DEVICE)
        for lo in range(0, len(order), 256):
            batch = order[lo : lo + 256]
            optimiser.zero_grad()
            criterion(model(xf[batch]), yf[batch]).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            scores = np.full(len(x), np.nan)
            scores[tune] = model(xt).cpu().numpy()
        aucs, _, _ = within_state(scores, tune)
        value = float(np.mean(aucs))
        if value > best["auc"]:
            best = {"auc": value, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model.eval(), {"tune_within_auc": best["auc"], "selected_epoch": best["epoch"]}


scorers = {}
scorers["fixed_direction_d"] = (x @ direction).numpy()
selection = {}
for architecture in ("linear", "small"):
    model, info = train(architecture, seed=20260817)
    with torch.no_grad():
        scorers[f"probe_{architecture}"] = model(x.to(DEVICE)).cpu().numpy()
    selection[architecture] = info
    print(f"  {architecture}: selected epoch {info['selected_epoch']} at tune "
          f"within-state AUC {info['tune_within_auc']:.4f}", flush=True)

test = split_id == "test"
print()
print("=" * 122)
print(f"HELD-OUT ROOTS ONLY -- {len(np.unique(root_id[test]))} roots, "
      f"{int(test.sum())} rows, never fitted or selected on")
print("=" * 122)
print(f"{'scorer':<24}{'within AUC':>13}{'95% CI':>22}{'within contrast':>18}"
      f"{'95% CI':>22}{'pooled AUC':>13}")
report = {}
for name, scores in scorers.items():
    aucs, contrasts, _ = within_state(scores, test)
    a, (alo, ahi) = interval(aucs, 1)
    c, (clo, chi) = interval(contrasts, 2)
    pooled = auc(scores[test], label.numpy()[test])
    report[name] = {"within_auc": a, "within_auc_ci": [alo, ahi],
                    "within_contrast": c, "within_contrast_ci": [clo, chi],
                    "pooled_auc": pooled, "roots": len(aucs)}
    print(f"{name:<24}{a:>13.4f}{f'[{alo:.4f}, {ahi:.4f}]':>22}{c:>18.4f}"
          f"{f'[{clo:.4f}, {chi:.4f}]':>22}{pooled:>13.4f}")

print()
print("Paired differences on the same held-out roots")
base_aucs, base_contrasts, base_roots = within_state(scorers["fixed_direction_d"], test)
for name in ("probe_linear", "probe_small"):
    aucs, contrasts, roots = within_state(scorers[name], test)
    assert np.array_equal(roots, base_roots)
    d_auc, (lo, hi) = interval(aucs - base_aucs, 3)
    print(f"  {name} minus fixed d: within-state AUC {d_auc:+.4f} [{lo:+.4f}, {hi:+.4f}]")
aucs_small, _, _ = within_state(scorers["probe_small"], test)
aucs_linear, _, _ = within_state(scorers["probe_linear"], test)
d_auc, (lo, hi) = interval(aucs_small - aucs_linear, 4)
print(f"  probe_small minus probe_linear:   within-state AUC {d_auc:+.4f} [{lo:+.4f}, {hi:+.4f}]")

print()
print("Restricted to nonterminal damage (deaths removed from the positive class)")
nonterminal_positive = ~terminal
print(f"{'scorer':<24}{'within AUC':>13}{'95% CI':>22}{'roots':>8}")
for name, scores in scorers.items():
    aucs, _, _ = within_state(scores, test, subset=nonterminal_positive)
    a, (lo, hi) = interval(aucs, 5)
    print(f"{name:<24}{a:>13.4f}{f'[{lo:.4f}, {hi:.4f}]':>22}{len(aucs):>8}")

print()
print("By damage type, held-out roots")
for label_name, subset in (("mob damage", kind == "mob"), ("lava / environment", kind == "lava")):
    counts = int((subset & (label.numpy() > 0) & test).sum())
    print(f"  {label_name} ({counts} held-out positive actions)")
    if counts < 15:
        print("    too few held-out positives to report")
        continue
    for name, scores in scorers.items():
        aucs, _, _ = within_state(scores, test, subset=subset)
        a, (lo, hi) = interval(aucs, 6)
        print(f"    {name:<22}{a:>10.4f}  [{lo:.4f}, {hi:.4f}]  roots {len(aucs)}")

(HERE / "probe_report.json").write_text(json.dumps(
    {"selection": selection, "held_out": report,
     "usable_roots": int(len(usable)), "records": len(records)}, indent=2))
print("\nwrote probe_report.json")
