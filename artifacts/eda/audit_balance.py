"""Was the all-action arm's consequence exposure matched to the factual arm's?

Both losses are replayed exactly as trained -- same seeds, same draw order -- and the
per-step positive counts recomputed, so the comparison is of what the optimizer
actually saw rather than of the code's intent.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import corpus
from train_damage_classifier import Sampler

from d4mj.config import Config

config = Config(transition="direct", time_mixer="attention")
rows = corpus.train_rows()
blob = np.load(HERE / "damage_labels.npz")
off, damage = blob["offsets"], blob["damage"]
STEPS = 20_000

# ------------------------------------------------------------------ factual arm
sampler = Sampler(rows, off, damage, config, seed=config.seed + 91)
per_step_f, windows = [], set()
for step in range(STEPS):
    finetune = step >= STEPS * (1 - config.long_only_fraction)
    long = finetune or (step + 1) % config.long_batch_every == 0
    length = config.sequence_long if long else config.sequence
    total = 0
    for episode, start in sampler.draw(step, STEPS, config.batch, length):
        n = rows[episode]["steps"]
        stop = min(start + length - 1, n)
        total += int(damage[off[episode] + start : off[episode] + stop].sum())
        windows.add((episode, start, length))
    per_step_f.append(total)
per_step_f = np.array(per_step_f)

# --------------------------------------------------------------- all-action arm
def split_of(record) -> str:
    key = f"consequence-probe:{record['shard']}:{record['slot']}:{record['t']}"
    draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % 10
    return "fit" if draw < 6 else ("tune" if draw < 8 else "test")

records = []
for path in sorted((HERE / "latent_forks").glob("shard-*.pt")):
    records += torch.load(path, weights_only=False)
fit_labels = []
for record in records:
    if split_of(record) != "fit":
        continue
    health, dead = record["health"].numpy(), record["dead"].numpy()
    fit_labels.append(((health <= -1) | dead).astype(np.float32))
fit_labels = np.stack(fit_labels)
draw = np.random.default_rng(config.seed + 91)
per_step_a, roots_seen = [], set()
for step in range(STEPS):
    chosen = draw.integers(0, len(fit_labels), config.batch)
    per_step_a.append(int(fit_labels[chosen].sum()))
    roots_seen.update(chosen.tolist())
per_step_a = np.array(per_step_a)

def weight_share(counts, per_step_targets):
    """With pos_weight = N_neg/N_pos, positive and negative mass are equal, so the
    positive share is exactly 50% whenever a batch has any positive, and 0% when it
    has none. The run-level share is therefore the fraction of steps with a positive."""
    has = counts > 0
    return float(has.mean()), float(np.mean(np.where(has, 0.5, 0.0)))

targets_f = np.where(np.arange(STEPS) % 4 == 3, 63, 15) * config.batch
targets_a = np.full(STEPS, 17 * config.batch)

print("=" * 104)
print("CONSEQUENCE EXPOSURE, AS ACTUALLY TRAINED")
print("=" * 104)
print(f"{'':<44}{'factual arm':>18}{'all-action arm':>20}")
rowfmt = "{:<44}{:>18}{:>20}"
print(rowfmt.format("optimizer steps", f"{STEPS:,}", f"{STEPS:,}"))
print(rowfmt.format("scored targets per step (mean)",
                    f"{targets_f.mean():.1f}", f"{targets_a.mean():.1f}"))
print(rowfmt.format("total positive target presentations",
                    f"{int(per_step_f.sum()):,}", f"{int(per_step_a.sum()):,}"))
print(rowfmt.format("positive targets per step (mean)",
                    f"{per_step_f.mean():.2f}", f"{per_step_a.mean():.2f}"))
print(rowfmt.format("positive fraction of scored targets",
                    f"{per_step_f.sum()/targets_f.sum():.3%}",
                    f"{per_step_a.sum()/targets_a.sum():.3%}"))
hf, sf = weight_share(per_step_f, targets_f)
ha, sa = weight_share(per_step_a, targets_a)
print(rowfmt.format("steps containing at least one positive", f"{hf:.1%}", f"{ha:.1%}"))
print(rowfmt.format("mean loss weight on positives", f"{sf:.1%}", f"{sa:.1%}"))
print(rowfmt.format("distinct positive examples available",
                    f"{int(damage.sum()):,}", f"{int(fit_labels.sum()):,}"))
print(rowfmt.format("distinct training units",
                    f"{len(windows):,} windows", f"{len(roots_seen):,} roots"))
print(rowfmt.format("hazard-choice roots in the training pool", "n/a",
                    f"{int(((fit_labels.sum(1) > 0) & (fit_labels.sum(1) < 17)).sum()):,}"))
print()
print("BCE form: both arms use pos_weight = N_neg/N_pos computed per batch, then a")
print("mean over scored entries. The per-batch balancing is mathematically identical.")
print("What differs is how often a batch contains a positive at all.")
print()
print("distribution of positives per step")
for name, counts in (("factual", per_step_f), ("all-action", per_step_a)):
    hist = np.bincount(counts, minlength=8)[:8]
    print(f"  {name:<12}" + "  ".join(f"{k}:{hist[k]/STEPS:.1%}" for k in range(8))
          + f"   >=8:{(counts >= 8).mean():.1%}")
np.savez(HERE / "audit_balance.npz", factual=per_step_f, allaction=per_step_a)
