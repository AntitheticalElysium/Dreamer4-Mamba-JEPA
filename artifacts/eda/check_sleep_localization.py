"""Why the actor stops sleeping, localized to one of reward, critic, or optimization.

WAKE_UP is the largest single achievement the Phase-3 actors give up. This replays the
BC on DEV seeds, stops at the states where the BC actually slept in an episode that
later woke, and asks what each policy sees there: the probability it puts on SLEEP, the
reward head's immediate prediction, the critic's value of the slept-into state, and the
sign of the PMPO advantage -- through the real `lambda_returns`, not a restatement of it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

# Must precede the craftax import: importing it initialises the jax backend, and
# Craftax must not reserve the 6 GB the world model trains on.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from craftax.craftax_classic.constants import Achievement, Action

from d4mj.actor_critic import lambda_returns
from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.imagination import Trajectory, _expect
from d4mj.representation import Decoder, Encoder
from d4mj.transition import World, advance, observe

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"
SLEEP = Action.SLEEP.value
WAKE_UP = [a.name for a in Achievement].index("WAKE_UP")


@torch.no_grad()
def _forced(world, heads, state, agent, action, rng, stream, config, samples):
    """One imagined rollout with the first action forced, the rest sampled from `heads`.

    Mirrors `imagination.imagine` step for step -- the only change is the committed
    first action -- and hands the trajectory to the production `lambda_returns`, so the
    advantage is the quantity PMPO actually takes the sign of.
    """
    out = []
    for sample in range(samples):
        # Same continuation noise for every policy at this state and action, so a
        # difference in advantage is the policy and critic, not the draw.
        policy_rng = torch.Generator(device=DEVICE).manual_seed(stream + sample)
        readout = heads(agent)
        logits = [readout["policy"][:, -1, 0]]
        values = [_expect(readout["value"][:, -1], heads.centers)]
        actions, rewards, continuations = [], [], []
        chosen = torch.full((1,), action, dtype=torch.long, device=DEVICE)
        local, carried = state, agent
        for lead in range(config.horizon):
            local, carried = advance(world, local, chosen[:, None], rng, config)
            readout = heads(carried)
            actions.append(chosen)
            rewards.append(_expect(readout["reward"][:, -1, 0], heads.centers))
            continuations.append(readout["continuation"][:, -1, 0].sigmoid())
            values.append(_expect(readout["value"][:, -1], heads.centers))
            if lead + 1 < config.horizon:
                logits.append(readout["policy"][:, -1, 0])
                chosen = torch.multinomial(
                    logits[-1].softmax(-1), 1, generator=policy_rng).squeeze(-1)
        trajectory = Trajectory(
            action=torch.stack(actions, 1), logits=torch.stack(logits, 1),
            reward=torch.stack(rewards, 1), continuation=torch.stack(continuations, 1),
            value=torch.stack(values, 1), agent=carried,
        )
        returns = lambda_returns(trajectory, config)
        out.append((float(returns[0, 0] - trajectory.value[0, 0]), float(rewards[0]),
                    float(values[1]), float(continuations[0])))
    advantage = [row[0] for row in out]
    return {
        "advantage": float(np.mean(advantage)),
        "positive_fraction": float(np.mean([value >= 0 for value in advantage])),
        "reward": float(np.mean([row[1] for row in out])),
        "value_next": float(np.mean([row[2] for row in out])),
        "continuation": float(np.mean([row[3] for row in out])),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("attention", "mamba"))
    parser.add_argument("--seed-base", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-sleeps", type=int, default=3)
    args = parser.parse_args()

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    saved = replace(base, transition="direct", time_mixer=args.arm)
    config = replace(saved, horizon=saved.direct_rollout)
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))
    world, prior = World(saved).to(DEVICE), Heads(saved).to(DEVICE)
    load(HERE / f"v2_phase2_{args.arm}" / "phase2_final.pt", saved, part0=world, part1=prior)
    policies = {"bc": prior.eval()}
    for tag, name in (("", "actor2500"), ("_10k", "actor10000")):
        actor = Heads(saved).to(DEVICE)
        load(HERE / f"v2_phase3_{args.arm}{tag}" / "phase3_final.pt", config,
             part0=World(saved).to(DEVICE), part1=actor)
        policies[name] = actor.eval()
    world.eval(), encoder.eval()

    rows = torch.load(HERE / "v2_paired_execution" / "episodes.pt",
                      weights_only=False)[f"{args.arm}_bc"]
    events, checked = [], 0
    for seed in range(args.seed_base, args.seed_base + args.episodes):
        rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**20)
        observation, env_state = reset(seed)
        state, total, sleeps = None, 0.0, []
        action = torch.full((1, 1), config.n_actions, dtype=torch.long, device=DEVICE)
        for index in range(config.horizon_eval):
            patches = patchify(observation[None, None], config.patch).to(DEVICE)
            state, agent = observe(world, encoder, state, action, patches, rng, config)
            logits = prior(agent)["policy"][:, -1, 0]
            choice = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
            if choice == SLEEP and len(sleeps) < args.max_sleeps:
                sleeps.append((index, state.world, agent))
            observation, env_state, reward, terminated, truncated = env_step(
                env_state, choice, seed + index + 1)
            total += reward
            action = torch.full((1, 1), choice, dtype=torch.long, device=DEVICE)
            if terminated or truncated:
                break
        unlocked = np.asarray(env_state.achievements)
        # The replay must be the deployed loop, not a lookalike: same generators, same
        # seed schedule, same observe call. Anything else and these are other states.
        assert rows[seed]["steps"] == index + 1, f"replay diverged at seed {seed}"
        assert abs(rows[seed]["reward"] - total) < 1e-5, f"reward diverged at seed {seed}"
        checked += 1
        if not unlocked[WAKE_UP]:
            continue
        for where, start, token in sleeps:
            record = {"seed": seed, "step": where}
            for name, heads in policies.items():
                probabilities = heads(token)["policy"][:, -1, 0].softmax(-1)[0]
                per_action = [_forced(world, heads, start, token, a, rng,
                                      seed * 4099 + where * 17 + a * 131, config,
                                      args.samples) for a in range(config.n_actions)]
                best = max(range(config.n_actions),
                           key=lambda a: per_action[a]["advantage"])
                record[name] = {
                    "probability_sleep": float(probabilities[SLEEP]),
                    "argmax_action": Action(int(probabilities.argmax())).name,
                    "sleep": per_action[SLEEP],
                    "best_action": Action(best).name,
                    "best_advantage": per_action[best]["advantage"],
                    "sleep_rank": int(sorted(range(config.n_actions),
                                             key=lambda a: -per_action[a]["advantage"]).index(SLEEP)),
                    "value_next_mean_other": float(np.mean(
                        [per_action[a]["value_next"] for a in range(config.n_actions) if a != SLEEP])),
                }
            events.append(record)

    out = HERE / f"v2_phase3_{args.arm}"
    (out / "sleep_localization.json").write_text(json.dumps(
        {"arm": args.arm, "episodes": checked, "events": len(events),
         "samples": args.samples, "rows": events}, indent=2, default=float))

    print(f"\n{args.arm}: {len(events)} BC sleep events in {checked} replayed episodes "
          f"that later woke ({args.samples} rollout samples per action)")
    if not events:
        return
    print(f"{'policy':<12}{'P(sleep)':>10}{'argmax':>16}{'adv sleep':>11}{'adv>=0':>8}"
          f"{'rank/17':>9}{'r pred':>9}{'v(next)':>9}{'v(other)':>10}{'c pred':>8}")
    for name in policies:
        take = lambda key, inner=None: float(np.mean(
            [row[name][key] if inner is None else row[name][key][inner] for row in events]))
        common = max(set(row[name]["argmax_action"] for row in events),
                     key=lambda a: sum(row[name]["argmax_action"] == a for row in events))
        print(f"{name:<12}{take('probability_sleep'):>10.4f}{common:>16}"
              f"{take('sleep', 'advantage'):>11.4f}{take('sleep', 'positive_fraction'):>8.2f}"
              f"{take('sleep_rank') + 1:>9.1f}{take('sleep', 'reward'):>9.4f}"
              f"{take('sleep', 'value_next'):>9.4f}{take('value_next_mean_other'):>10.4f}"
              f"{take('sleep', 'continuation'):>8.4f}")


if __name__ == "__main__":
    main()
