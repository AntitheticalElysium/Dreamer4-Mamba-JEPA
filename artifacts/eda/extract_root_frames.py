"""Frame histories at the damage roots, with the rollout's own Z* for verification.

Alternate bottleneck geometries must encode exactly the same physical states, so the
trajectory is fixed by rolling the *production* encoder and policy once and saving
raw frames. Each arm then re-encodes those frames offline.

The rollout's Z* is saved alongside so the offline path can be verified against it
before any new geometry is trained: `Z*_offline(saved history)` must reproduce
`Z*_rollout`. The encoder's receptive field is 1 + (depth/time_every)(window-1) = 31,
so 32 saved frames is exactly enough for the final frame's latent to be complete.
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

from artifacts.localize_counterfactual import load_models
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.transition import observe

OUT = HERE / "root_frames"
OUT.mkdir(exist_ok=True)
base = Config()
config = Config(transition="direct", time_mixer="attention")
HISTORY = 32
assert HISTORY >= base.receptive_field, (HISTORY, base.receptive_field)
print(f"receptive field {base.receptive_field}, saving {HISTORY} frames per root", flush=True)

encoder, world, heads = load_models(
    ROOT / "artifacts/stage_a_terminalfix/phase1a.pt",
    ROOT / "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt", base, config)

wanted: dict[int, dict[int, np.ndarray]] = {}
for path in sorted((HERE / "branched_damage").glob("seed-*.pt")):
    payload = torch.load(path, weights_only=False)
    for row in payload["rows"]:
        health, dead = row["health"].numpy(), row["dead"].numpy()
        label = (health <= -1) | dead
        if label.any() and not label.all():
            wanted.setdefault(int(payload["seed"]), {})[int(row["step"])] = \
                label.astype(np.float32)
print(f"{sum(len(v) for v in wanted.values())} damage roots across {len(wanted)} seeds",
      flush=True)

rows, started, shard = [], time.time(), 0
with torch.no_grad():
    for order, seed in enumerate(sorted(wanted)):
        observation, env_state = reset(seed)
        state = None
        incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=config.device)
        world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
        frames: list[torch.Tensor] = []
        for index in range(min(max(wanted[seed]) + 1, 400)):
            frames.append(observation.clone())
            patches = patchify(observation[None, None], config.patch).to(config.device)
            state, agent = observe(world, encoder, state, incoming, patches, world_rng, config)
            logits = heads(agent)["policy"][:, -1, 0]
            chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
            if index in wanted[seed]:
                rows.append({
                    "seed": seed, "step": index,
                    "frames": torch.stack(frames[-HISTORY:]),
                    "z_rollout": state.world.latent[0, -1].reshape(-1).cpu().clone(),
                    "label": torch.from_numpy(wanted[seed][index]),
                })
            observation, env_state, _, terminated, truncated = env_step(
                env_state, chosen, seed + index + 1)
            incoming.fill_(chosen)
            if terminated or truncated:
                break
        if len(rows) >= 800:
            torch.save(rows, OUT / f"shard-{shard:03d}.pt")
            shard, rows = shard + 1, []
        if (order + 1) % 50 == 0:
            rate = (order + 1) / (time.time() - started)
            print(f"  seed {order+1}/{len(wanted)} [{time.time()-started:.0f}s, "
                  f"{(len(wanted)-order-1)/rate:.0f}s left]", flush=True)
if rows:
    torch.save(rows, OUT / f"shard-{shard:03d}.pt")
    shard += 1
(OUT / "manifest.json").write_text(json.dumps({"shards": shard, "history": HISTORY}, indent=2))
print(f"wrote {shard} shards", flush=True)
