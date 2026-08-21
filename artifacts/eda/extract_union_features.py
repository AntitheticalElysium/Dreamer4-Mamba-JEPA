"""Pre-bottleneck and Z* features at every action-varying state, all families at once.

One re-roll covers damage, death and reward: the union of states where any of the
three outcomes differs across the 17 actions. Labels for all three are stored per
state so each family can be filtered afterwards without another pass.
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

OUT = HERE / "state_features"
base = Config()
config = Config(transition="direct", time_mixer="attention")
encoder, world, heads = load_models(
    ROOT / "artifacts/stage_a_terminalfix/phase1a.pt",
    ROOT / "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt", base, config)
captured: dict[str, torch.Tensor] = {}
encoder.bottleneck.register_forward_pre_hook(
    lambda module, inputs: captured.__setitem__("pre", inputs[0].detach()))

collection = ROOT / "artifacts/branched_coverage_gate/collection"
manifest = json.loads((collection / "manifest.json").read_text())
reward_truth = {}
for record in manifest["shards"]:
    payload = torch.load(collection / record["file"], weights_only=False)
    seed = int(payload["seed"])
    for i, step in enumerate(payload["step"].tolist()):
        reward_truth[(seed, step)] = payload["true_reward"][i].numpy()

wanted: dict[int, dict[int, dict]] = {}
for path in sorted((HERE / "branched_damage").glob("seed-*.pt")):
    payload = torch.load(path, weights_only=False)
    seed = int(payload["seed"])
    for row in payload["rows"]:
        health, dead = row["health"].numpy(), row["dead"].numpy()
        labels = {
            "damage": ((health <= -1) | dead).astype(np.float32),
            "death": dead.astype(np.float32),
            "reward": (reward_truth.get((seed, int(row["step"])),
                                        np.zeros(17)) > 0).astype(np.float32),
        }
        if any(v.any() and not v.all() for v in labels.values()):
            wanted.setdefault(seed, {})[int(row["step"])] = labels
print(f"{sum(len(v) for v in wanted.values())} action-varying states across "
      f"{len(wanted)} seeds", flush=True)

rows, started = [], time.time()
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
            if index in wanted[seed]:
                rows.append({
                    "seed": seed, "step": index,
                    "pre": captured["pre"][0, -1].reshape(-1).cpu().clone().half(),
                    "z": state.world.latent[0, -1].reshape(-1).cpu().clone(),
                    **{f"label_{k}": torch.from_numpy(v)
                       for k, v in wanted[seed][index].items()},
                })
            observation, env_state, _, terminated, truncated = env_step(
                env_state, chosen, seed + index + 1)
            incoming.fill_(chosen)
            if terminated or truncated:
                break
        if (order + 1) % 50 == 0:
            rate = (order + 1) / (time.time() - started)
            print(f"  seed {order+1}/{len(wanted)} states {len(rows)} "
                  f"[{time.time()-started:.0f}s, {(len(wanted)-order-1)/rate:.0f}s left]",
                  flush=True)

torch.save(rows, OUT / "union_features.pt")
print(f"wrote {len(rows)} states")
