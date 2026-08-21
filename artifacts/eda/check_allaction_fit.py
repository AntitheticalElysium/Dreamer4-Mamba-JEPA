"""Does the all-action arm fit its training roots and fail to generalize?

Within-state AUC computed separately on the fit, tune and test halves of the same
whole-root split. A large fit-minus-test gap means the 5,410 training roots were
memorised, which would make the positive control uninformative about whether paired
interventions teach the mechanic.
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
from evaluate_damage_classifier import auc, interval, score_roots
from train_damage_classifier import DamageHead

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World


def split_of(record) -> str:
    key = f"consequence-probe:{record['shard']}:{record['slot']}:{record['t']}"
    draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % 10
    return "fit" if draw < 6 else ("tune" if draw < 8 else "test")


config = Config(transition="direct", time_mixer="attention")
rows = corpus.train_rows()
lookup = {(r["shard"], r["slot"]): i for i, r in enumerate(rows) if r["source"] == "support"}
cached = {i: v for i, v in corpus.iter_cached_latents()}
manifest = json.loads((corpus.SUPPORT / "manifest.json").read_text())

records = []
for path in sorted((HERE / "latent_forks").glob("shard-*.pt")):
    records += torch.load(path, weights_only=False)

action_cache: dict[int, dict] = {}
buckets: dict[str, list] = {"fit": [], "tune": [], "test": []}
rng = np.random.default_rng(5)
for record in records:
    health, dead = record["health"].numpy(), record["dead"].numpy()
    positives = (health <= -1) | dead
    if not positives.any() or not ((health >= 0) & ~dead).any():
        continue
    split = split_of(record)
    if split == "fit" and rng.random() > 0.4:      # subsample fit for speed
        continue
    shard = record["shard"]
    if shard not in action_cache:
        payload = torch.load(corpus.SUPPORT / manifest["shards"][shard]["file"],
                             weights_only=False, mmap=True)
        action_cache[shard] = {s: f["actions_taken"].numpy()
                               for s, f in enumerate(payload["episodes"])}
        del payload
    acts = action_cache[shard][record["slot"]]
    t = record["t"]
    start = max(0, t - config.sequence_long + 1)
    led = np.concatenate([[config.n_actions] if start == 0 else [acts[start - 1]],
                          acts[start : t]]).astype(np.int64)
    buckets[split].append((cached[lookup[(shard, record["slot"])]][start : t + 1].clone(),
                           torch.from_numpy(led), positives.astype(float)))

for name, path in (("all-action", "damage_allaction/model_020000.pt"),
                   ("factual", "damage_classifier/model_020000.pt")):
    world, head = World(config).to(config.device), DamageHead(config).to(config.device)
    load(HERE / path, config, part0=world, part1=head)
    world.eval(); head.eval()
    print(f"\n{name}")
    for split in ("fit", "tune", "test"):
        rowset = buckets[split]
        scores = score_roots(world, head, config, [r[0] for r in rowset],
                             [r[1] for r in rowset])
        labels = np.stack([r[2] for r in rowset])
        values = np.array([auc(scores[i], labels[i]) for i in range(len(rowset))])
        a, (lo, hi) = interval(values, 3)
        print(f"  {split:<5} roots {len(values):>4}  within AUC {a:.4f} [{lo:.4f}, {hi:.4f}]")
