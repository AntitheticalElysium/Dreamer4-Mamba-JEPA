"""The 17 successor frames at every damage root, on the fixed production trajectory.

Pairs with `root_frames`, which already holds each root's 32-frame causal history.
Together they let any tokenizer encode all 17 successors under the root's own history
-- the causal contract Z* is defined by -- without re-rolling per arm, and without
letting a new encoder alter the policy and move the root set.

Keyed by (seed, step) so it joins to `root_frames` exactly.
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

OUT = HERE / "fork_successors"
OUT.mkdir(exist_ok=True)
base = Config()
config = Config(transition="direct", time_mixer="attention")
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
            wanted.setdefault(int(payload["seed"]), {})[int(row["step"])] = label.astype(np.float32)
print(f"{sum(len(v) for v in wanted.values())} damage roots", flush=True)

rows, started, shard = [], time.time(), 0
with torch.no_grad():
    for order, seed in enumerate(sorted(wanted)):
        observation, env_state = reset(seed)
        state = None
        incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=config.device)
        world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
        for index in range(min(max(wanted[seed]) + 1, 400)):
            patches = patchify(observation[None, None], config.patch).to(config.device)
            state, agent = observe(world, encoder, state, incoming, patches, world_rng, config)
            logits = heads(agent)["policy"][:, -1, 0]
            chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
            successors = [env_step(env_state, action, seed + index + 1) for action in range(17)]
            if index in wanted[seed]:
                rows.append({
                    "seed": seed, "step": index,
                    "successors": torch.stack([s[0] for s in successors]),
                    "terminated": torch.tensor([s[3] for s in successors]),
                    "label": torch.from_numpy(wanted[seed][index]),
                })
            observation, env_state, _, terminated, truncated = successors[chosen]
            incoming.fill_(chosen)
            if terminated or truncated:
                break
        if len(rows) >= 400:
            torch.save(rows, OUT / f"shard-{shard:03d}.pt")
            shard, rows = shard + 1, []
        if (order + 1) % 50 == 0:
            rate = (order + 1) / (time.time() - started)
            print(f"  seed {order+1}/{len(wanted)} [{time.time()-started:.0f}s, "
                  f"{(len(wanted)-order-1)/rate:.0f}s left]", flush=True)
if rows:
    torch.save(rows, OUT / f"shard-{shard:03d}.pt")
    shard += 1
(OUT / "manifest.json").write_text(json.dumps({"shards": shard}, indent=2))
print(f"wrote {shard} shards", flush=True)
