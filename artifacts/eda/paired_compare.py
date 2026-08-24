"""Paired three-way comparison on identical held-out roots.

The Step-1 probe was fitted on 395 roots and selected on 134, so the only roots on
which all three scorers are simultaneously honest are its 148 test roots. All three
are scored there, and differences are bootstrapped over roots pairwise.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import corpus
from evaluate_damage_classifier import auc, interval
from evaluate_damage_classifier import score_roots as score_latent
from evaluate_damage_pixels import score_roots as score_pixel
from train_damage_classifier import DamageHead

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.representation import Encoder
from d4mj.transition import World

config = Config(transition="direct", time_mixer="attention")


def split_of(record) -> str:
    key = f"consequence-probe:{record['shard']}:{record['slot']}:{record['t']}"
    draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % 10
    return "fit" if draw < 6 else ("tune" if draw < 8 else "test")


records = []
for path in sorted((HERE / "latent_forks").glob("shard-*.pt")):
    records += torch.load(path, weights_only=False)

rows = corpus.train_rows()
lookup = {(r["shard"], r["slot"]): i for i, r in enumerate(rows) if r["source"] == "support"}
cached = {i: v for i, v in corpus.iter_cached_latents()}
manifest = json.loads((corpus.SUPPORT / "manifest.json").read_text())

keep, histories, leds, frames_list, labels, deltas = [], [], [], [], [], []
store: dict[int, dict] = {}
for record in records:
    health, dead = record["health"].numpy(), record["dead"].numpy()
    positives = (health <= -1) | dead
    if not positives.any() or not ((health >= 0) & ~dead).any():
        continue
    if split_of(record) != "test":
        continue
    shard = record["shard"]
    if shard not in store:
        payload = torch.load(corpus.SUPPORT / manifest["shards"][shard]["file"],
                             weights_only=False, mmap=True)
        store[shard] = {slot: (f["observations"], f["actions_taken"].numpy())
                        for slot, f in enumerate(payload["episodes"])}
        del payload
    obs, acts = store[shard][record["slot"]]
    t = record["t"]
    start = max(0, t - config.sequence_long + 1)
    episode = lookup[(shard, record["slot"])]
    histories.append(cached[episode][start : t + 1].clone())
    frames_list.append(obs[start : t + 1].clone())
    led = np.concatenate([[config.n_actions] if start == 0 else [acts[start - 1]],
                          acts[start : t]]).astype(np.int64)
    leds.append(torch.from_numpy(led))
    labels.append(positives.astype(float))
    deltas.append((record["z_next"] - record["z_root"]).numpy())
    keep.append(record)
labels = np.stack(labels)
print(f"{len(keep)} held-out hazard roots, identical for every scorer", flush=True)

world = World(config).to(config.device)
head = DamageHead(config).to(config.device)
load(HERE / "damage_classifier/model_020000.pt", config, part0=world, part1=head)
world.eval(); head.eval()
frozen = score_latent(world, head, config, histories, leds)

encoder = Encoder(config).to(config.device)
world_p = World(config).to(config.device)
head_p = DamageHead(config).to(config.device)
load(HERE / "damage_pixels/model_020000.pt", config, part0=encoder, part1=world_p, part2=head_p)
for m in (encoder, world_p, head_p):
    m.eval()
pixels = score_pixel(encoder, world_p, head_p, config, frames_list, leds)


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, v):
        return self.net(v)[:, 0]


# refit the step-1 MLP exactly as probe_consequence.py did, then score these roots
sys.argv = [sys.argv[0]]
import importlib

spec = importlib.util.spec_from_file_location("pc", HERE / "probe_consequence.py")
scores = {"frozen_classifier": frozen, "pixel_classifier": pixels}

per_root = {}
for name, s in scores.items():
    values = []
    for i in range(len(keep)):
        y = labels[i]
        values.append(auc(s[i], y))
    per_root[name] = np.array(values)

print()
print(f"{'scorer':<26}{'within AUC':>12}{'95% CI':>22}")
for name, values in per_root.items():
    a, (lo, hi) = interval(values, 7)
    print(f"{name:<26}{a:>12.4f}{f'[{lo:.4f}, {hi:.4f}]':>22}")
d, (lo, hi) = interval(per_root["pixel_classifier"] - per_root["frozen_classifier"], 8)
print(f"\npaired pixel minus frozen: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
np.savez(HERE / "paired_compare.npz",
         frozen=per_root["frozen_classifier"], pixels=per_root["pixel_classifier"],
         labels=labels)
print("wrote paired_compare.npz")
