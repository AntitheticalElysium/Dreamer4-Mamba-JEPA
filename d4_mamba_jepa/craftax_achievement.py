"""Executed-achievement evaluation for Craftax policies.

The Craftax analogue of the CartPole ``evaluate_actor_parity``: run a frozen
policy (random / BC / imagination actor) directly in the live Craftax-Classic
env, score with the official geometric-mean Crafter score, and report paired
episode-level bootstrap confidence intervals for actor-minus-BC and
actor-minus-random.

Deployment samples the categorical policy at temperature 1 (D053; greedy argmax
collapses an imbalanced discrete policy onto the most frequent action). Imports
craftax via ``craftax_env`` -- run as an evaluation job, not from training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .craftax_env import CraftaxPixelEnv, N_ACTIONS, achievement_names
from .executed_control import _crafter_score


@torch.inference_mode()
def _policy_action(
    world, policy, *, observations, led_to_actions, context, device,
    mode="sample", temperature=1.0, generator=None,
) -> int:
    """Action from the clean post-context agent token (no denoising/planner)."""
    window = observations[-context:]
    frames = torch.from_numpy(np.stack(window))[None].to(device)  # [1,T,C,H,W]
    led = torch.tensor(
        led_to_actions[-context:], device=device, dtype=torch.long
    )[None]
    time_steps = frames.shape[1]
    steps = torch.full(
        (1, time_steps), world.cfg.max_step_index, device=device, dtype=torch.long
    )
    signals = torch.full(
        (1, time_steps), world.cfg.k_max, device=device, dtype=torch.long
    )
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        packed = world.encode_frames(frames, frozen=True).packed
        _, agent = world.forward_dynamics(packed, led, steps, signals)
        logits = policy(agent[:, -1:].float())[:, 0].float()
    if mode == "greedy":
        return int(logits.argmax(dim=-1).item())
    probs = (logits / max(1e-6, temperature)).softmax(dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


def run_policy_episode(
    *, world, policy, policy_name, env_seed, policy_seed, context,
    max_steps, device, mode="sample",
) -> dict:
    """Roll one episode; return the achievement set and official per-episode score."""
    env = CraftaxPixelEnv(seed=int(env_seed))
    obs = env.reset()
    observations = [obs]
    led_to_actions = [-1]
    rng = np.random.default_rng(policy_seed)
    generator = torch.Generator(device=device).manual_seed(int(policy_seed))
    names = achievement_names()
    final = np.zeros(len(names), dtype=bool)
    length = 0
    for _ in range(int(max_steps)):
        if policy_name == "random":
            action = int(rng.integers(N_ACTIONS))
        else:
            action = _policy_action(
                world, policy, observations=observations,
                led_to_actions=led_to_actions,
                context=min(context, len(observations)), device=device,
                mode=mode, generator=generator,
            )
        result = env.step(action)
        observations.append(result.obs)
        led_to_actions.append(action)
        final = result.achievements
        length += 1
        if result.done:
            break
    return {
        "policy": policy_name,
        "env_seed": int(env_seed),
        "achievements": {names[i]: int(final[i]) for i in range(len(names))},
        "achievement_count": int(final.sum()),
        "length": length,
    }


def _paired_score_ci(rows_a, rows_b, seeds, *, seed, draws=20000):
    """Paired episode-bootstrap CI for the official-score difference (a - b).

    The Crafter score is a set-level aggregate, so each draw resamples seeds with
    replacement and recomputes both scores on the SAME resampled seeds (paired).
    """
    rng = np.random.default_rng(seed)
    point = _crafter_score([rows_a[s] for s in seeds])[0] - _crafter_score(
        [rows_b[s] for s in seeds])[0]
    seed_arr = np.asarray(seeds)
    diffs = np.empty(draws, dtype=float)
    n = len(seeds)
    for d in range(draws):
        pick = seed_arr[rng.integers(0, n, size=n)]
        diffs[d] = (_crafter_score([rows_a[int(s)] for s in pick])[0]
                    - _crafter_score([rows_b[int(s)] for s in pick])[0])
    return float(point), [float(np.percentile(diffs, 2.5)),
                          float(np.percentile(diffs, 97.5))]


def evaluate_craftax_achievement(
    *, world, bc_policy, actor_policy, seeds, context, max_steps, device,
    policy_seed_base=7_000_000, mode="sample",
) -> dict:
    """Run random / BC / actor over ``seeds`` and report official-score parity."""
    policies = {"random": None, "bc": bc_policy, "imagination_actor": actor_policy}
    by_policy = {name: {} for name in policies}
    for name, policy in policies.items():
        for env_seed in seeds:
            row = run_policy_episode(
                world=world, policy=policy, policy_name=("random" if name == "random" else name),
                env_seed=env_seed,
                policy_seed=policy_seed_base + env_seed + (1_000_000 if name == "random" else 0),
                context=context, max_steps=max_steps, device=device, mode=mode,
            )
            by_policy[name][int(env_seed)] = row

    summary = {}
    for name in policies:
        rows = [by_policy[name][s] for s in seeds]
        score, rates = _crafter_score(rows)
        summary[name] = {
            "crafter_score": score,
            "mean_achievement_count": float(np.mean([r["achievement_count"] for r in rows])),
            "success_rates": rates,
        }
    actor_minus_bc, actor_minus_bc_ci = _paired_score_ci(
        by_policy["imagination_actor"], by_policy["bc"], list(seeds),
        seed=policy_seed_base + 9)
    actor_minus_random, actor_minus_random_ci = _paired_score_ci(
        by_policy["imagination_actor"], by_policy["random"], list(seeds),
        seed=policy_seed_base + 8)
    return {
        "summary": summary,
        "actor_minus_bc": actor_minus_bc,
        "actor_minus_bc_ci": actor_minus_bc_ci,
        "actor_beats_bc": actor_minus_bc_ci[0] > 0.0,
        "actor_minus_random": actor_minus_random,
        "actor_minus_random_ci": actor_minus_random_ci,
        "actor_beats_random": actor_minus_random_ci[0] > 0.0,
        "rows": {name: list(by_policy[name].values()) for name in policies},
    }


__all__ = [
    "run_policy_episode",
    "evaluate_craftax_achievement",
]
