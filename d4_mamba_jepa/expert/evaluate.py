"""Spot-check a PPO expert checkpoint: per-achievement unlock rate, official
Crafter score, and survival length.

Symbolic-only (achievements/score/survival all come from ``EnvState`` -- no
render needed), vectorized over N envs under one jitted ``lax.scan``, and forced
onto CPU so it can score a checkpoint WHILE GPU training keeps writing new ones.
Each env's achievement set is frozen at its FIRST ``done`` (post-terminal steps
of a non-auto-reset env must not inflate the count).
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # never contend with GPU training

import numpy as np

from ..craftax_env import achievement_names
from ..executed_control import _crafter_score
from .ppo_expert import ENV_NAME, load_expert


def evaluate_expert(
    params_path,
    *,
    n_episodes: int = 16,
    max_steps: int = 2000,
    layer_size: int = 512,
    seed: int = 0,
    greedy: bool = True,
) -> dict:
    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name
    from .ppo_expert import ScannedRNN

    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    env_params = env.default_params
    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)
    network, params, _ = load_expert(
        params_path, obs_dim=obs_dim, action_dim=action_dim, layer_size=layer_size)

    N = int(n_episodes)
    reset_v = jax.vmap(env.reset, in_axes=(0, None))
    step_v = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    @jax.jit
    def _rollout(key):
        key, rk = jax.random.split(key)
        obs, state = reset_v(jax.random.split(rk, N), env_params)
        hidden = ScannedRNN.initialize_carry(N, layer_size)
        ach_final = state.achievements                     # [N,22]
        done_any = jnp.zeros(N, dtype=bool)
        ep_len = jnp.zeros(N, dtype=jnp.int32)

        def _step(carry, k):
            hidden, state, obs, done_prev, done_any, ep_len, ach_final = carry
            ka, sk = jax.random.split(k)
            hidden, logits, _ = network.apply(params, hidden, (obs[None], done_prev[None]))
            logits = logits[0]
            action = (jnp.argmax(logits, axis=-1) if greedy
                      else jax.random.categorical(ka, logits))
            obs2, state2, reward, done, _ = step_v(
                jax.random.split(sk, N), state, action, env_params)
            live = ~done_any                                # still in its episode
            ep_len = ep_len + live.astype(jnp.int32)
            ach_final = jnp.where(live[:, None], state2.achievements, ach_final)
            done_any = done_any | done
            return (hidden, state2, obs2, done, done_any, ep_len, ach_final), None

        init = (hidden, state, obs, jnp.zeros(N, dtype=bool), done_any, ep_len, ach_final)
        keys = jax.random.split(key, max_steps)
        carry, _ = jax.lax.scan(_step, init, keys)
        return carry[6], carry[4], carry[5]                 # ach_final, done_any, ep_len

    ach_final, done_any, ep_len = _rollout(jax.random.PRNGKey(int(seed)))
    ach_final = np.asarray(ach_final).astype(bool)
    done_any = np.asarray(done_any).astype(bool)
    ep_len = np.asarray(ep_len).astype(np.int64)

    names = achievement_names()
    rows = [{"achievements": {names[i]: int(ach_final[n, i]) for i in range(len(names))}}
            for n in range(N)]
    score, rates = _crafter_score(rows)
    rates = {k: rates.get(k, 0.0) for k in names}          # ensure all 22 present

    return {
        "params_path": str(params_path),
        "n_episodes": N,
        "max_steps": int(max_steps),
        "greedy": bool(greedy),
        "crafter_score": float(score),
        "mean_achievement_count": float(ach_final.sum(axis=1).mean()),
        "achievement_rates_pct": {k: round(v, 1) for k, v in rates.items()},
        "mean_survival": float(ep_len.mean()),
        "median_survival": float(np.median(ep_len)),
        "max_survival": int(ep_len.max()),
        "terminated_fraction": float(done_any.mean()),  # rest hit max_steps (truncated)
    }


def _cli() -> None:
    p = argparse.ArgumentParser(description="Spot-check a PPO expert checkpoint.")
    p.add_argument("--params", required=True)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--layer-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample", action="store_true", help="temp-1 sampling (default greedy)")
    args = p.parse_args()
    r = evaluate_expert(
        args.params, n_episodes=args.episodes, max_steps=args.max_steps,
        layer_size=args.layer_size, seed=args.seed, greedy=not args.sample)
    unlocked = {k: v for k, v in r["achievement_rates_pct"].items() if v > 0}
    r_print = dict(r)
    r_print["achievement_rates_pct"] = dict(sorted(unlocked.items(), key=lambda kv: -kv[1]))
    print(json.dumps(r_print, indent=2))


if __name__ == "__main__":
    _cli()


__all__ = ["evaluate_expert"]
