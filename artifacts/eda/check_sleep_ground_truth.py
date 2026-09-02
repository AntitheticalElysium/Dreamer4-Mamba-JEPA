"""Ground truth for the SLEEP question: does the simulator agree with the world model?

The localization showed both critics value the *generated* slept-into state below its
alternatives, but could not say whether the generated successor, the critic, the reward
head, or SLEEP's real worth is at fault -- nor why mamba's actor prefers SLEEP and still
never wakes. Four measurements on one state set:

1. SLEEP opportunities and choices along every policy's own deployed trajectory. An
   opportunity is exact, not a light-level guess: `update_player_intrinsics` starts sleep
   only when `player_energy < 9`, and WAKE_UP fires when energy reaches 9 while asleep.
2. The simulator's real SLEEP reward and observed successor against the head's predicted
   reward and Direct's generated successor.
3. The critic on both successors, observed and generated.
4. A real forked rollout -- SLEEP against alternatives, each followed by the same BC
   under the same noise -- long enough to measure SLEEP's true delayed value.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from craftax.craftax_classic.constants import Achievement, Action

from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.imagination import _expect
from d4mj.representation import Decoder, Encoder
from d4mj.transition import World, advance, observe

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"
SLEEP = Action.SLEEP.value
WAKE_UP = [a.name for a in Achievement].index("WAKE_UP")


def _tokens(action: int) -> torch.Tensor:
    return torch.full((1, 1), action, dtype=torch.long, device=DEVICE)


def _value(heads: Heads, agent: torch.Tensor) -> float:
    return float(_expect(heads(agent)["value"][:, -1], heads.centers))


@torch.no_grad()
def _replay(world, encoder, heads, seed, config, keep=None):
    """`execution.run_episode`'s loop, generator for generator, with the sleep mechanics
    logged. `keep` collects the states where this policy starts a real sleep."""
    rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**21)
    policy_rng = torch.Generator(device=DEVICE).manual_seed(seed + 2**20)
    observation, env_state = reset(seed)
    state, total, action = None, 0.0, _tokens(config.n_actions)
    tally = {"steps": 0, "opportunities": 0, "sleep_actions": 0,
             "effective_sleeps": 0, "asleep_steps": 0}
    for index in range(config.horizon_eval):
        patches = patchify(observation[None, None], config.patch).to(DEVICE)
        state, agent = observe(world, encoder, state, action, patches, rng, config)
        choice = int(torch.multinomial(
            heads(agent)["policy"][:, -1, 0].softmax(-1), 1, generator=policy_rng))
        asleep, energy = bool(env_state.is_sleeping), int(env_state.player_energy)
        chance = (not asleep) and energy < 9
        tally["opportunities"] += chance
        tally["sleep_actions"] += choice == SLEEP
        tally["asleep_steps"] += asleep
        if choice == SLEEP and chance:
            tally["effective_sleeps"] += 1
            if keep is not None and len(keep) < keep.maxlen:
                keep.append((index, seed, env_state, state, agent, observation))
        observation, env_state, reward, terminated, truncated = env_step(
            env_state, choice, seed + index + 1)
        total += reward
        action = _tokens(choice)
        if terminated or truncated:
            break
    tally["steps"] = index + 1
    tally["reward"] = total
    tally["woke"] = bool(np.asarray(env_state.achievements)[WAKE_UP])
    return tally


@torch.no_grad()
def _true_return(world, encoder, prior, seed, index, env_state, state, action,
                 rng, config, steps):
    """Execute `action` for real, then follow the BC under noise fixed by the branch
    point, so SLEEP and its alternatives differ only by the committed first action."""
    policy_rng = torch.Generator(device=DEVICE).manual_seed(seed * 7919 + index)
    observation, local, reward, terminated, truncated = env_step(
        env_state, action, seed + index + 1)
    total, discount = reward, config.gamma
    carried = observe(world, encoder, state, _tokens(action),
                      patchify(observation[None, None], config.patch).to(DEVICE),
                      rng, config)
    for offset in range(steps):
        if terminated or truncated:
            break
        choice = int(torch.multinomial(
            prior(carried[1])["policy"][:, -1, 0].softmax(-1), 1, generator=policy_rng))
        observation, local, reward, terminated, truncated = env_step(
            local, choice, seed + index + offset + 2)
        total += discount * reward
        discount *= config.gamma
        carried = observe(world, encoder, carried[0], _tokens(choice),
                          patchify(observation[None, None], config.patch).to(DEVICE),
                          rng, config)
    return total, bool(np.asarray(local.achievements)[WAKE_UP])


@torch.no_grad()
def main() -> None:
    from collections import deque

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("attention", "mamba"))
    parser.add_argument("--seed-base", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--branch-steps", type=int, default=250)
    parser.add_argument("--alternatives", type=int, default=3)
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

    # 1. Sleep mechanics along each policy's own trajectory.
    seeds = range(args.seed_base, args.seed_base + args.episodes)
    keep, mechanics = deque(maxlen=args.states), {}
    for name, heads in policies.items():
        tallies = [_replay(world, encoder, heads, seed, config,
                           keep if name == "bc" else None) for seed in seeds]
        total = lambda key: float(sum(row[key] for row in tallies))
        mechanics[name] = {
            "episodes": len(tallies), "steps": total("steps"),
            "opportunities": total("opportunities"), "sleep_actions": total("sleep_actions"),
            "effective_sleeps": total("effective_sleeps"), "asleep_steps": total("asleep_steps"),
            "woke_rate": float(np.mean([row["woke"] for row in tallies])),
            "sleep_per_opportunity": total("effective_sleeps") / max(total("opportunities"), 1),
        }
        print(f"  replayed {name}", flush=True)

    print(f"\n{args.arm}: SLEEP mechanics on {args.episodes} DEV seeds "
          f"(opportunity = awake and energy < 9)")
    print(f"{'policy':<13}{'steps':>9}{'chances':>9}{'SLEEP acts':>12}{'effective':>11}"
          f"{'per chance':>12}{'asleep':>9}{'woke':>8}")
    for name, row in mechanics.items():
        print(f"{name:<13}{row['steps']:>9.0f}{row['opportunities']:>9.0f}"
              f"{row['sleep_actions']:>12.0f}{row['effective_sleeps']:>11.0f}"
              f"{row['sleep_per_opportunity']:>12.4f}{row['asleep_steps']:>9.0f}"
              f"{row['woke_rate']:>8.3f}")

    # 2-4. Observed against generated, at the BC's own real sleep starts.
    rng = torch.Generator(device=DEVICE).manual_seed(2**19)
    rows = []
    for index, seed, env_state, state, agent, _ in keep:
        truth, predicted = {}, {}
        for action in range(config.n_actions):
            observation, _, reward, _, _ = env_step(env_state, action, seed + index + 1)
            patches = patchify(observation[None, None], config.patch).to(DEVICE)
            seen, seen_agent = observe(world, encoder, state, _tokens(action), patches,
                                       rng, config)
            made, made_agent = advance(world, state.world, _tokens(action), rng, config)
            truth[action] = (reward, seen.world.latent, seen_agent)
            predicted[action] = (made.latent, made_agent)
        record = {"seed": seed, "step": index,
                  "true_reward_sleep": truth[SLEEP][0],
                  "true_reward_other": float(np.mean(
                      [truth[a][0] for a in truth if a != SLEEP]))}
        drift = lambda a: float(F.mse_loss(predicted[a][0], truth[a][1]))
        cosine = lambda a: float(F.cosine_similarity(
            predicted[a][0].flatten(), truth[a][1].flatten(), dim=0))
        record["drift_sleep"], record["cosine_sleep"] = drift(SLEEP), cosine(SLEEP)
        record["drift_other"] = float(np.mean([drift(a) for a in truth if a != SLEEP]))
        record["cosine_other"] = float(np.mean([cosine(a) for a in truth if a != SLEEP]))
        for name, heads in policies.items():
            readout = heads(agent)
            record[name] = {
                "predicted_reward_sleep": float(_expect(
                    readout["reward"][:, -1, 0], heads.centers)),
                "probability_sleep": float(
                    readout["policy"][:, -1, 0].softmax(-1)[0, SLEEP]),
                "value_observed_sleep": _value(heads, truth[SLEEP][2]),
                "value_generated_sleep": _value(heads, predicted[SLEEP][1]),
                "value_observed_other": float(np.mean(
                    [_value(heads, truth[a][2]) for a in truth if a != SLEEP])),
                "value_generated_other": float(np.mean(
                    [_value(heads, predicted[a][1]) for a in truth if a != SLEEP])),
            }
        # 4. Is SLEEP actually worth it? Same BC, same noise, only the first action differs.
        # What each policy would do instead, plus a few draws so the comparison is
        # not only against the two actions the models already favour.
        alternatives = {int(policies["actor10000"](agent)["policy"][:, -1, 0].argmax()),
                        int(prior(agent)["policy"][:, -1, 0].argmax())}
        alternatives |= set(np.random.default_rng(seed + index).choice(
            [a for a in range(config.n_actions) if a != SLEEP],
            args.alternatives, replace=False).tolist())
        alternatives = sorted(alternatives - {SLEEP})
        record["true_return_sleep"], record["true_woke_sleep"] = _true_return(
            world, encoder, prior, seed, index, env_state, state, SLEEP, rng, config,
            args.branch_steps)
        returns = {}
        for action in alternatives:
            returns[Action(action).name] = _true_return(
                world, encoder, prior, seed, index, env_state, state, action, rng,
                config, args.branch_steps)
        record["true_return_alternatives"] = {k: v[0] for k, v in returns.items()}
        record["true_woke_alternatives"] = {k: v[1] for k, v in returns.items()}
        record["true_return_best_alternative"] = max(v[0] for v in returns.values())
        rows.append(record)
        print(f"  state {len(rows)}/{len(keep)} seed {seed} step {index}", flush=True)

    out = HERE / f"v2_phase3_{args.arm}"
    (out / "sleep_ground_truth.json").write_text(json.dumps(
        {"arm": args.arm, "mechanics": mechanics, "states": len(rows),
         "branch_steps": args.branch_steps, "rows": rows}, indent=2, default=float))
    if not rows:
        return

    mean = lambda key: float(np.mean([row[key] for row in rows]))
    inner = lambda name, key: float(np.mean([row[name][key] for row in rows]))
    print(f"\n{args.arm}: {len(rows)} real BC sleep starts, "
          f"{args.branch_steps}-step forked BC rollouts, gamma={config.gamma}")
    print(f"  true SLEEP return {mean('true_return_sleep'):+.4f}   "
          f"best alternative {mean('true_return_best_alternative'):+.4f}   "
          f"woke after SLEEP {float(np.mean([row['true_woke_sleep'] for row in rows])):.3f}")
    print(f"  true reward  sleep {mean('true_reward_sleep'):+.4f} "
          f"other {mean('true_reward_other'):+.4f}")
    print(f"  successor drift MSE  sleep {mean('drift_sleep'):.5f} "
          f"other {mean('drift_other'):.5f}   cosine sleep {mean('cosine_sleep'):.4f} "
          f"other {mean('cosine_other'):.4f}")
    print(f"\n{'policy':<13}{'r pred':>9}{'P(sleep)':>10}{'v obs sleep':>13}"
          f"{'v gen sleep':>13}{'v obs other':>13}{'v gen other':>13}")
    for name in policies:
        print(f"{name:<13}{inner(name, 'predicted_reward_sleep'):>9.4f}"
              f"{inner(name, 'probability_sleep'):>10.4f}"
              f"{inner(name, 'value_observed_sleep'):>13.4f}"
              f"{inner(name, 'value_generated_sleep'):>13.4f}"
              f"{inner(name, 'value_observed_other'):>13.4f}"
              f"{inner(name, 'value_generated_other'):>13.4f}")


if __name__ == "__main__":
    main()
