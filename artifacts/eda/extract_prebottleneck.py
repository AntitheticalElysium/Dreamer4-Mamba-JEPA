"""Encoder features immediately before the bottleneck, at the same hazard roots.

`Encoder.forward` ends `z = tanh(bottleneck(encoded[:, :, :n_latents]))`, so a
forward hook on `bottleneck` captures its input exactly -- the 32x256 backbone
output, 8192 dims against Z*'s 512. Both are taken from the same forward pass on the
same frame, so the pair differs only by the projection.
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
from d4mj.representation import pack
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

wanted: dict[int, dict[int, np.ndarray]] = {}
for path in sorted((HERE / "branched_damage").glob("seed-*.pt")):
    payload = torch.load(path, weights_only=False)
    for row in payload["rows"]:
        label = ((row["health"].numpy() <= -1) | row["dead"].numpy())
        if label.any() and not label.all():
            wanted.setdefault(int(payload["seed"]), {})[int(row["step"])] = \
                label.astype(np.float32)
print(f"{sum(len(v) for v in wanted.values())} hazard roots across {len(wanted)} seeds",
      flush=True)

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
                pre = captured["pre"]                       # (1, 1, 32, 256)
                rows.append({
                    "seed": seed, "step": index,
                    "label": torch.from_numpy(wanted[seed][index]),
                    "pre": pre[0, -1].reshape(-1).cpu().clone(),
                    "z": state.world.latent[0, -1].reshape(-1).cpu().clone(),
                })
            observation, env_state, _, terminated, truncated = env_step(
                env_state, chosen, seed + index + 1)
            incoming.fill_(chosen)
            if terminated or truncated:
                break
        if (order + 1) % 50 == 0:
            rate = (order + 1) / (time.time() - started)
            print(f"  seed {order+1}/{len(wanted)} roots {len(rows)} "
                  f"[{time.time()-started:.0f}s, {(len(wanted)-order-1)/rate:.0f}s left]",
                  flush=True)

torch.save(rows, OUT / "prebottleneck.pt")
print(f"wrote {len(rows)} roots; pre dim {rows[0]['pre'].numel()}, "
      f"z dim {rows[0]['z'].numel()}")
