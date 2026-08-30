"""Does the one-pass collector reproduce the existing corpus exactly on seed 14000?

The BC trajectory must be identical or the roots are different states and nothing
downstream is comparable. Compares, against the archives the original five passes wrote:

  led_to_action   the whole action sequence, from `branched_damage`
  health / dead   per action at every recorded step, from `branched_damage`
  frames          root history and all 17 first successors, from `fork_histories`
                  and `fork_successors`
  second step     the surviving branches' NOOP successors, from `multistep_forks`
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from collect_broad_forks import (HISTORY, N_ACTIONS, NOOP, frames_of, load_policy,
                                 make_fork)

from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.transition import observe

import argparse

SEED = 14_000


def main() -> None:
    global SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    SEED = parser.parse_args().seed
    base = replace(Config(), n_latents=32)
    config = replace(base, transition="direct", time_mixer="attention")
    encoder, world, heads = load_policy(base)
    fork = make_fork()
    truth = torch.load(HERE / "branched_damage" / f"seed-{SEED:06d}.pt", weights_only=False)
    recorded = {int(r["step"]): r for r in truth["rows"]}

    observation, env_state = reset(SEED)
    state, led, ours = None, [config.n_actions], {}
    incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=config.device)
    world_rng = torch.Generator(device=config.device).manual_seed(SEED + 2**21)
    policy_rng = torch.Generator(device=config.device).manual_seed(SEED + 2**20)
    frames, mismatches = [], []

    with torch.no_grad():
        for index in range(400):
            frames.append(observation.clone())
            patches = patchify(observation[None, None], config.patch).to(config.device)
            state, agent = observe(world, encoder, state, incoming, patches, world_rng, config)
            chosen = int(torch.multinomial(heads(agent)["policy"][:, -1, 0].softmax(-1), 1,
                                           generator=policy_rng))
            key = jax.random.PRNGKey(SEED + index + 1)
            obs1, state1, _, dead1, trunc1, health1, _ = fork(env_state, key,
                                                              jnp.arange(N_ACTIONS))
            delta = np.asarray(health1) - float(env_state.player_health)
            dead = np.asarray(dead1)
            if index in recorded:
                if not np.allclose(delta, recorded[index]["health"].numpy()):
                    mismatches.append(f"health at step {index}")
                if not np.array_equal(dead, recorded[index]["dead"].numpy()):
                    mismatches.append(f"dead at step {index}")
            ours[index] = {"frames": list(frames[-HISTORY:]), "obs1": obs1,
                           "state1": state1, "alive": ~dead & ~np.asarray(trunc1)}
            observation, env_state, _, terminated, truncated = env_step(
                env_state, chosen, SEED + index + 1)
            led.append(chosen)
            incoming.fill_(chosen)
            if terminated or truncated:
                break

    stored_led = truth["led_to_action"].tolist()
    ours_led = led[: len(stored_led)]
    print(f"seed {SEED}")
    print(f"trajectory: ours {len(led)-1} steps, archive {len(stored_led)-1}")
    print(f"  led_to_action identical over the archive's length: {ours_led == stored_led}")
    first = next((i for i, (a, b) in enumerate(zip(ours_led, stored_led)) if a != b), None)
    if first is not None:
        print(f"  first differing action at index {first}: ours {ours_led[first]} "
              f"archive {stored_led[first]}")
        print(f"  ours    {ours_led[max(0,first-3):first+4]}")
        print(f"  archive {stored_led[max(0,first-3):first+4]}")
    print(f"  recorded steps in archive {len(recorded)}, matched by us "
          f"{len(set(recorded) & set(ours))}")
    print(f"  health/dead mismatches: {len(mismatches)}"
          + (f" -> {mismatches[:3]}" if mismatches else ""))

    successors = {}
    for path in sorted(glob.glob(str(HERE / "fork_successors" / "shard-*.pt"))):
        for row in torch.load(path, weights_only=False):
            if int(row["seed"]) == SEED:
                successors[int(row["step"])] = row
    histories = {int(r["step"]): r for r in torch.load(
        HERE / "fork_histories" / "branched_965.pt", weights_only=False)
        if int(r["seed"]) == SEED}
    ok_f = ok_h = 0
    for step, row in successors.items():
        if step not in ours:
            continue
        ok_f += int(torch.equal(frames_of(ours[step]["obs1"]).cpu(), row["successors"].cpu()))
        if step in histories:
            mine = torch.stack(ours[step]["frames"])
            theirs = histories[step]["frames"]
            ok_h += int(torch.equal(mine.cpu(), theirs[-len(mine):].cpu()))
    print(f"  fork_successors roots {len(successors)}: 17-successor frames identical "
          f"{ok_f}/{len(successors)}")
    print(f"  fork_histories roots {len(histories)}: history frames identical "
          f"{ok_h}/{len(histories)}")

    multi = [r for r in torch.load(HERE / "multistep_forks" / "rollouts.pt",
                                   weights_only=False) if int(r["seed"]) == SEED]
    ok_s = checked = 0
    for row in multi:
        step = int(row["step"])
        if step not in ours:
            continue
        key2 = jax.random.PRNGKey(SEED + step + 2)
        for a in np.where(ours[step]["alive"])[0]:
            branch = jax.tree_util.tree_map(lambda x, i=int(a): x[i], ours[step]["state1"])
            o2, *_ = fork(branch, key2, jnp.array([NOOP]))
            checked += 1
            ok_s += int(torch.equal(frames_of(o2[0]).cpu(), row["successors"][a, 1].cpu()))
    print(f"  multistep roots for this seed: {len(multi)}; second NOOP successors "
          f"identical: {ok_s}/{checked}")


if __name__ == "__main__":
    main()
