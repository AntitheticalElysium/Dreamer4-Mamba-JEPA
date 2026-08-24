"""Tasks 4 and 5: action-dependent consequences that are not death.

All 17 actions are executed from ordinary TRAIN states stratified by achievement
band and behaviour source. Simulator outcomes need no model; the fatality-direction
delta `y` encodes each successor under the state's own causal history, which is the
protocol `evaluate_consequence_learnability` uses for its fork successors.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import replay

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import episode_splits, patchify
from d4mj.representation import Encoder, pack
from d4mj.train import _cache_digest

PER_CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 80
BANDS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 22)]
EPS = [0.1, 0.25, 0.5, 1.0]
CONTEXT = 32          # >= receptive_field (31)

base = Config()
encoder = Encoder(base).to(base.device)
load(ROOT / "artifacts/stage_a_terminalfix/phase1a.pt", base, part0=encoder)
encoder.eval()
prepared = torch.load(
    ROOT / "artifacts/terminal_diversity_v2/preparation/prepared.pt", weights_only=False
)
digest = _cache_digest(encoder, base)
assert digest == prepared["cache_digest"], (digest, prepared["cache_digest"])
direction = prepared["direction"].float().flatten().to(base.device)
print(f"encoder matches production cache digest {digest}", flush=True)

# ------------------------------------------------ candidate states from support TRAIN
manifest = replay.manifest()
rng = np.random.default_rng(41)
candidates: dict[tuple, list] = {}
for shard_index, record in enumerate(manifest["shards"]):
    payload = torch.load(replay.STORE / record["file"], weights_only=False, mmap=True)
    for slot, fields in enumerate(payload["episodes"]):
        if fields["split"] != "train":
            continue
        steps = len(fields["actions_taken"])
        if steps < 40:
            continue
        events = fields["events"].numpy()
        progress = np.cumsum(events) - events
        for band in BANDS:
            inside = np.where((progress >= band[0]) & (progress <= band[1]))[0]
            inside = inside[(inside >= CONTEXT) & (inside < steps - 1)]
            if len(inside):
                candidates.setdefault((band, float(fields["epsilon"])), []).append(
                    (shard_index, slot, int(inside[rng.integers(len(inside))]))
                )
    del payload
    if (shard_index + 1) % 100 == 0:
        print(f"  scanned {shard_index + 1}/{len(manifest['shards'])} shards", flush=True)

selected = []
for band in BANDS:
    for eps in EPS:
        pool = candidates.get((band, eps), [])
        if not pool:
            continue
        take = rng.choice(len(pool), size=min(PER_CELL, len(pool)), replace=False)
        selected += [(band, eps, *pool[j]) for j in take]
print(f"{len(selected)} states across {len(BANDS)}x{len(EPS)} cells", flush=True)

import jax

KEY = jax.random.PRNGKey(7)
_, _, _, step_fn, frame_fn = replay.env_and_render()
print(f"replay check, max pixel diff: {replay.verify(0, 5)}", flush=True)


@torch.no_grad()
def latents_for(context_frames: np.ndarray, successors: np.ndarray):
    """z of the root and of all 17 successors under one shared causal history.

    Seventeen copies of the context each followed by one successor: the encoder
    applies its own bounded window, so this equals carrying memory forward and
    costs one call instead of memory surgery.
    """
    context = torch.from_numpy(context_frames)
    stacked = torch.stack([
        torch.cat([context, torch.from_numpy(successors[a])[None]]) for a in range(17)
    ])
    z, _, _ = encoder(patchify(stacked, base.patch).to(base.device))
    packed = pack(z, base).flatten(2)
    return packed[0, -2], packed[:, -1]


INV = ("wood", "stone", "coal", "iron", "diamond", "sapling")
records = []
started = time.time()
for n, (band, eps, shard_index, slot, t) in enumerate(selected):
    state = replay.advance_to(shard_index, slot, t)
    if replay.is_dead(state):
        continue
    root = replay.scalars(state)
    frames = replay.episode_fields(shard_index, slot)["observations"].numpy()
    context = frames[max(0, t - CONTEXT + 1) : t + 1]
    successors, health, reward, ach, dead, inventory = [], [], [], [], [], []
    for action in range(17):
        _, nxt, r, _, _ = step_fn(KEY, state, action)
        successors.append(np.asarray(frame_fn(nxt)))
        s = replay.scalars(nxt)
        health.append(s["health"] - root["health"])
        reward.append(float(r))
        ach.append(s["achievements"] - root["achievements"])
        dead.append(replay.is_dead(nxt))
        inventory.append(sum(abs(s[k] - root[k]) for k in INV))
    z_root, z_next = latents_for(context, np.stack(successors))
    y = ((z_next - z_root[None]) @ direction).cpu().numpy()
    records.append(dict(
        band=band, epsilon=eps, shard=shard_index, slot=slot, t=t,
        health=health, reward=reward, achievements=ach, dead=dead,
        inventory=inventory, y=y.tolist(),
        base_health=root["health"], base_ach=root["achievements"],
        zombies=root["n_zombies"], skeletons=root["n_skeletons"],
        light=root["light"], sleeping=root["sleeping"],
    ))
    if (n + 1) % 100 == 0:
        rate = (n + 1) / (time.time() - started)
        print(f"  {n + 1}/{len(selected)} [{time.time()-started:.0f}s, "
              f"{(len(selected)-n-1)/rate:.0f}s left]", flush=True)

torch.save(records, HERE / "nonterminal_forks.pt")
print(f"wrote {len(records)} fork records")
