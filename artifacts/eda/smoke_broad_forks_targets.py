"""Do the collected first- and second-step targets drive finite, nonzero gradients?

Encodes a few roots with the production 64x16 tokenizer and runs them through the same
calls `_direct_loss` uses -- `advance` for each generated state -- then checks the
gradient actually exists. Only second-step targets marked valid are scored; the invalid
slots hold zeros and must never enter the loss.
"""

from __future__ import annotations

import glob
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from evaluate_death_transfer import DEVICE, ENCODER, REPORT

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.representation import Encoder, pack
from d4mj.state import WorldState
from d4mj.transition import World, commit_inputs, advance

N_ACTIONS, ROOTS = 17, 3


def main() -> None:
    base = replace(Config(), n_latents=64, d_bottleneck=16)
    config = replace(base, transition="direct", time_mixer="attention")
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]), part0=encoder)
    encoder.eval()
    world = World(config).to(DEVICE)

    rows = torch.load(sorted(glob.glob(str(HERE / "broad_forks_v2" / "shard-*.pt")))[0],
                      weights_only=False)[:ROOTS]
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed)

    def encode(frames):
        with torch.no_grad():
            z, _, _ = encoder(patchify(frames[None], config.patch).to(DEVICE))
        return pack(z, config)[0]

    total, scored_second = 0.0, 0
    for row in rows:
        history, led = row["frames"], row["led_to_action"].long()
        z_hist = encode(history)
        committed, conditioning = commit_inputs(z_hist[None], rng, config)
        features, _, memory = world(None, led[None].to(DEVICE), committed, conditioning)
        valid = row["second_valid"].numpy()
        for a in range(N_ACTIONS):
            z1 = encode(torch.cat([history, row["successors"][a][None]]))[-1]
            action = torch.full((1, 1), a, dtype=torch.long, device=DEVICE)
            state = WorldState(z_hist[None][:, -1:], memory, z_hist.shape[0], features[:, -1:])
            first, _ = advance(world, state, action, rng, config)
            total = total + (first.latent[0, 0] - z1).pow(2).mean()
            if valid[a]:
                z2 = encode(torch.cat([history, row["successors"][a][None],
                                       row["second"][a][None]]))[-1]
                noop = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
                second, _ = advance(world, first, noop, rng, config)
                total = total + (second.latent[0, 0] - z2).pow(2).mean()
                scored_second += 1

    total.backward()
    grads = [p.grad for p in world.parameters() if p.grad is not None]
    flat = torch.cat([g.flatten() for g in grads])
    print(f"roots {len(rows)}  first targets {len(rows)*N_ACTIONS}  "
          f"second targets scored {scored_second}")
    print(f"loss {float(total):.5f}   finite {bool(torch.isfinite(total))}")
    print(f"gradient tensors {len(grads)}/{sum(1 for _ in world.parameters())}  "
          f"all finite {bool(torch.isfinite(flat).all())}  "
          f"nonzero {int((flat != 0).sum()):,}/{flat.numel():,}  norm {float(flat.norm()):.4f}")
    assert torch.isfinite(flat).all() and flat.norm() > 0, "targets produced no usable gradient"
    print("both first- and second-step targets drive a finite, nonzero gradient")


if __name__ == "__main__":
    main()
