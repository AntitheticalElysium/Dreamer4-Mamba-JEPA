"""The 1,222-state all-action fork collection, re-run at scale and keeping latents.

Same selection rule, same encoder, same fork protocol; the only change is that the
17 successor latents and the root latent are retained instead of only their
projection onto `d`. Sampling is enlarged because the within-state question is only
defined on roots that offer both a damaging and a non-damaging action, which is
8.5% of ordinary states -- 1,222 states would leave ~20 held-out roots.
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
from d4mj.data import patchify
from d4mj.representation import Encoder, pack
from d4mj.train import _cache_digest

PER_CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 700
OUT = HERE / "latent_forks"
OUT.mkdir(exist_ok=True)
BANDS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 22)]
EPS = [0.1, 0.25, 0.5, 1.0]
CONTEXT = 32
SHARD = 512

base = Config()
encoder = Encoder(base).to(base.device)
load(ROOT / "artifacts/stage_a_terminalfix/phase1a.pt", base, part0=encoder)
encoder.eval()
prepared = torch.load(
    ROOT / "artifacts/terminal_diversity_v2/preparation/prepared.pt", weights_only=False
)
digest = _cache_digest(encoder, base)
assert digest == prepared["cache_digest"], (digest, prepared["cache_digest"])
direction = prepared["direction"].float().flatten()
print(f"encoder matches production cache digest {digest}", flush=True)

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
    if (shard_index + 1) % 140 == 0:
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
    """Root and 17 successor latents under one shared causal history."""
    context = torch.from_numpy(context_frames)
    stacked = torch.stack([
        torch.cat([context, torch.from_numpy(successors[a])[None]]) for a in range(17)
    ])
    z, _, _ = encoder(patchify(stacked, base.patch).to(base.device))
    packed = pack(z, base).flatten(2)
    return packed[0, -2].cpu(), packed[:, -1].cpu()


def mob_distance(state):
    position = np.asarray(state.player_position)
    out = {}
    for name, mobs in (("zombie", state.zombies), ("skeleton", state.skeletons),
                       ("arrow", state.arrows)):
        mask = np.asarray(mobs.mask)
        pos = np.asarray(mobs.position)[mask]
        out[name] = int(np.abs(pos - position).sum(-1).min()) if len(pos) else 99
    return out


def lava_tile(state) -> bool:
    from craftax.craftax_classic.constants import BlockType

    return bool(
        state.map[state.player_position[0], state.player_position[1]]
        == BlockType.LAVA.value
    )


INV = ("wood", "stone", "coal", "iron", "diamond", "sapling")
buffer, written, kept = [], 0, 0
started = time.time()
for n, (band, eps, shard_index, slot, t) in enumerate(selected):
    state = replay.advance_to(shard_index, slot, t)
    if replay.is_dead(state):
        continue
    root = replay.scalars(state)
    frames = replay.episode_fields(shard_index, slot)["observations"].numpy()
    context = frames[max(0, t - CONTEXT + 1) : t + 1]
    successors, health, reward, ach, dead, inventory, lava = [], [], [], [], [], [], []
    for action in range(17):
        _, nxt, r, _, _ = step_fn(KEY, state, action)
        successors.append(np.asarray(frame_fn(nxt)))
        s = replay.scalars(nxt)
        health.append(s["health"] - root["health"])
        reward.append(float(r))
        ach.append(s["achievements"] - root["achievements"])
        dead.append(replay.is_dead(nxt))
        inventory.append(sum(abs(s[k] - root[k]) for k in INV))
        lava.append(lava_tile(nxt))
    z_root, z_next = latents_for(context, np.stack(successors))
    buffer.append(dict(
        band=band, epsilon=eps, shard=shard_index, slot=slot, t=t,
        z_root=z_root, z_next=z_next,
        health=torch.tensor(health, dtype=torch.float32),
        reward=torch.tensor(reward, dtype=torch.float32),
        achievements=torch.tensor(ach, dtype=torch.int16),
        dead=torch.tensor(dead), lava=torch.tensor(lava),
        inventory=torch.tensor(inventory, dtype=torch.int16),
        base_health=root["health"], base_ach=root["achievements"],
        zombies=root["n_zombies"], skeletons=root["n_skeletons"],
        light=root["light"], sleeping=root["sleeping"],
        mob_distance=mob_distance(state),
    ))
    kept += 1
    if len(buffer) >= SHARD:
        torch.save(buffer, OUT / f"shard-{written:04d}.pt")
        written += 1
        buffer = []
    if (n + 1) % 500 == 0:
        rate = (n + 1) / (time.time() - started)
        print(f"  {n + 1}/{len(selected)} [{time.time()-started:.0f}s, "
              f"{(len(selected)-n-1)/rate:.0f}s left]", flush=True)

if buffer:
    torch.save(buffer, OUT / f"shard-{written:04d}.pt")
    written += 1
(OUT / "manifest.json").write_text(json.dumps({
    "states": kept, "shards": written, "cache_digest": digest,
    "per_cell": PER_CELL, "context": CONTEXT,
    "direction_sha": str(float(direction.sum())),
}, indent=2))
print(f"wrote {kept} states in {written} shards")
