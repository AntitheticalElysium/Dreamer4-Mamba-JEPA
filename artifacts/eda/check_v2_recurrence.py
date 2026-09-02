"""Does the v2 trainer's second step equal `advance(advance(state, action), NOOP)`?

The trainer batches all 17 actions of several roots through one backbone pass and repeats
the carried memory across them. That is only legitimate if it reproduces the production
recurrence applied to each root and action on its own. An earlier version passed
`memory=None` and trained a history-free transition instead, which no smoke of the core
model could catch, because the core model was never wrong.

Compares the batched path against per-root, per-action `advance` calls, for both mixers.
"""

from __future__ import annotations

import glob
import sys
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from train_terminal_arms import DEVICE, N_ACTIONS, repeat_memory, v2_roots

from d4mj.config import Config
from d4mj.state import WorldState
from d4mj.transition import World, advance, commit_inputs


def main() -> None:
    base = replace(Config(), n_latents=64, d_bottleneck=16)
    history, branch, led, _, _, second, valid = v2_roots()
    roots = torch.arange(3)
    spatial, d = base.n_spatial, base.d_spatial

    for mixer in ("attention", "mamba"):
        config = replace(base, transition="direct", time_mixer=mixer)
        torch.manual_seed(config.seed + 1)
        world = World(config).to(DEVICE).eval()
        for parameter in world.parameters():
            parameter.requires_grad_(False)

        with torch.no_grad():
            # --- the trainer's batched path
            rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
            z = history[roots].to(DEVICE)
            n, t = z.shape[0], z.shape[1]
            committed, conditioning = commit_inputs(z.view(n, t, spatial, d), rng, config)
            features, _, memory = world(None, led[roots].to(DEVICE), committed, conditioning)
            a = N_ACTIONS
            flat = torch.arange(a, device=DEVICE).repeat(n)[:, None]
            carried = repeat_memory(memory, n, a)
            state = WorldState(z[:, -1:].view(n, 1, spatial, d).repeat_interleave(a, 0),
                               carried, t, features[:, -1:].repeat_interleave(a, dim=0))
            first, _ = advance(world, state, flat, rng, config)
            batched, _ = advance(world, first, torch.zeros_like(flat), rng, config)

            # --- one root, one action at a time
            singles = []
            for i in range(n):
                for act in range(a):
                    # the history is re-run for every action: `world` may write into the
                    # memory it is handed, so a memory reused across actions is not the
                    # same starting state and would make this reference wrong
                    rng_i = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
                    zi = history[roots[i]][None].to(DEVICE)
                    ci, gi = commit_inputs(zi.view(1, t, spatial, d), rng_i, config)
                    fi, _, mi = world(None, led[roots[i]][None].to(DEVICE), ci, gi)
                    si = WorldState(zi[:, -1:].view(1, 1, spatial, d), mi, t, fi[:, -1:])
                    one = torch.tensor([[act]], device=DEVICE)
                    f1, _ = advance(world, si, one, rng_i, config)
                    f2, _ = advance(world, f1, torch.zeros_like(one), rng_i, config)
                    singles.append(f2.latent[0, 0])
            singles = torch.stack(singles)

        worst = float((batched.latent.flatten(2)[:, 0] - singles.flatten(1)).abs().max())
        print(f"  {mixer:<10} batched vs per-root advance: worst difference {worst:.3e}"
              f"   {'OK' if worst < 1e-4 else 'MISMATCH'}")
        assert worst < 1e-4, f"{mixer}: the trainer is not running the production recurrence"
    print("the v2 second step is the production two-step rollout, both mixers")


if __name__ == "__main__":
    main()
