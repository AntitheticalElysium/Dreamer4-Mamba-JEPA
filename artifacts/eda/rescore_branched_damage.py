"""Can the existing 67,365-state S82 collection be rescored for health damage?

It stored `true_death` and `true_reward` per action but not health. The states are
reproducible -- the same frozen BC policy under checkpoints that still hash to their
recorded digests, already verified to replay exactly -- so each seed is re-rolled and
all 17 actions re-executed at every recorded state, this time keeping health.

Stored per seed: the full latent trajectory and led-to actions once, so any root's
history is a slice, plus the 17 outcomes at each recorded step. `true_death` is
checked against the stored collection at every state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from artifacts.localize_counterfactual import load_models
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import _env, reset, step as env_step
from d4mj.transition import observe

COLLECTION = ROOT / "artifacts/branched_coverage_gate/collection"
OUT = HERE / "branched_damage"
OUT.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=512)
    parser.add_argument("--seed-start", type=int, default=14_000)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    encoder, world, heads = load_models(
        ROOT / "artifacts/stage_a_terminalfix/phase1a.pt",
        ROOT / "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt",
        base, config)
    env, params = _env()

    def fork(state, key):
        def one(action):
            _, nxt, _, _, _ = env.step(key, state, action, params)
            from craftax.craftax_classic.constants import BlockType

            lava = nxt.map[nxt.player_position[0], nxt.player_position[1]] == BlockType.LAVA.value
            return nxt.player_health, lava | (nxt.player_health <= 0)

        return jax.vmap(one)(jnp.arange(17))

    fork = jax.jit(fork)

    manifest = json.loads((COLLECTION / "manifest.json").read_text())
    stored = {}
    for record in manifest["shards"]:
        payload = torch.load(COLLECTION / record["file"], weights_only=False)
        stored[int(payload["seed"])] = payload

    started, states, checked = time.time(), 0, 0
    with torch.no_grad():
        for order, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds)):
            if seed not in stored:
                continue
            truth = stored[seed]
            want = {int(s): i for i, s in enumerate(truth["step"].tolist())}
            observation, env_state = reset(seed)
            state = None
            incoming = torch.full((1, 1), config.n_actions, dtype=torch.long,
                                  device=config.device)
            world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
            policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
            latents, led, rows = [], [config.n_actions], []
            for index in range(args.limit):
                patches = patchify(observation[None, None], config.patch).to(config.device)
                state, agent = observe(world, encoder, state, incoming, patches,
                                       world_rng, config)
                latents.append(state.world.latent[0, -1].cpu())
                logits = heads(agent)["policy"][:, -1, 0]
                chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
                if index in want:
                    key = jax.random.PRNGKey(seed + index + 1)
                    health, dead = fork(env_state, key)
                    health = np.asarray(health) - float(env_state.player_health)
                    dead = np.asarray(dead)
                    reference = truth["true_death"][want[index]].numpy()
                    if not np.array_equal(dead, reference):
                        raise AssertionError(f"true_death changed at {seed}:{index}")
                    checked += 1
                    rows.append({"step": index,
                                 "health": torch.from_numpy(health.astype(np.float32)),
                                 "dead": torch.from_numpy(dead)})
                    states += 1
                observation, env_state, _, terminated, truncated = env_step(
                    env_state, chosen, seed + index + 1)
                led.append(chosen)
                incoming.fill_(chosen)
                if terminated or truncated:
                    break
            torch.save({"seed": seed, "latents": torch.stack(latents),
                        "led_to_action": torch.tensor(led[:len(latents)], dtype=torch.long),
                        "rows": rows}, OUT / f"seed-{seed:06d}.pt")
            if (order + 1) % 20 == 0:
                rate = (order + 1) / (time.time() - started)
                print(f"  seed {order+1}/{args.seeds}  states {states:,} "
                      f"[{time.time()-started:.0f}s, {(args.seeds-order-1)/rate:.0f}s left]",
                      flush=True)

    damaging = 0
    hazard = 0
    for path in OUT.glob("seed-*.pt"):
        for row in torch.load(path, weights_only=False)["rows"]:
            positive = (row["health"].numpy() <= -1) | row["dead"].numpy()
            damaging += int(positive.sum())
            hazard += int(positive.any() and not positive.all())
    print(f"\nrescored {states:,} states, true_death verified at all {checked:,}")
    print(f"  damaging (state, action) pairs: {damaging:,}")
    print(f"  hazard-choice roots (some damage, some not): {hazard:,}")
    (OUT / "manifest.json").write_text(json.dumps(
        {"states": states, "verified": checked, "damaging_pairs": damaging,
         "hazard_roots": hazard, "seeds": args.seeds}, indent=2))


if __name__ == "__main__":
    main()
