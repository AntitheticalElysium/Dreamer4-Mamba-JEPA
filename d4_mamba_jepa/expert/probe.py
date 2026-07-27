"""Collect oracle probe data on EXPERT frames.

The representation oracle asks whether the world encoder's latent still carries
privileged simulator state. That question is only answerable on the distribution
the world was TRAINED on: our worlds saw expert pixels, so probing scripted
``forage`` frames would confound "the latent lost it" with "the encoder never
saw frames like this". This collector therefore rolls the SAME PPO expert that
produced the training replay and pairs each retained frame with its ground-truth
labels.

Rollout mechanics are the vectorized/jitted ``lax.scan`` of ``expert.generate``
(one episode per env slot, sliced at that slot's FIRST done) and the render is
the identical 7px -> padded 64x64 path, so probe frames are byte-comparable to
training frames.

Two deliberate departures from ``craftax_oracle.collect_probe_data``:

  * FRAME STRIDE. That collector keeps every frame. Expert episodes run to 2,500
    steps, so 40 episodes would be 100k frames, and the oracle's raw-pixel ridge
    does an economy SVD of an ``[n, 12288]`` matrix -- O(n^2 p), which is
    intractable well before that. Consecutive Craftax frames are also highly
    autocorrelated, so a stride buys near-independent samples rather than losing
    information. Episodes remain the bootstrap unit, so the CIs stay valid.
  * Labels are read from the vectorized ``EnvState`` inside the scan rather than
    from a stateful single env, matching how the frames are produced.

Imports JAX/Craftax -- run as a collection job, never from training. Output is a
``.probe_only.pt`` payload, which the replay loader refuses to train on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..craftax_env import (
    INVENTORY_FIELDS,
    N_ACHIEVEMENTS,
    N_ACTIONS,
    VITAL_FIELDS,
    achievement_names,
)
from ..craftax_oracle import ProbeData, save_probe_data
from .ppo_expert import ENV_NAME, load_expert


def collect_expert_probe_data(
    *,
    params_path: str | Path,
    n_episodes: int,
    max_steps: int = 2_500,
    stride: int = 40,
    layer_size: int = 512,
    seed: int = 0,
    greedy: bool = False,
    target_size: int = 64,
    num_envs: int = 16,
) -> ProbeData:
    """Roll the expert and keep every ``stride``-th frame with its labels."""
    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name
    from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
    from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT
    from .ppo_expert import ScannedRNN

    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    env_params = env.default_params
    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)
    assert action_dim == N_ACTIONS, action_dim

    network, params, _ = load_expert(
        params_path, obs_dim=obs_dim, action_dim=action_dim, layer_size=layer_size
    )
    render_fn = make_craftax_pixel_renderer(int(BLOCK_PIXEL_SIZE_AGENT))
    pad = int(target_size) - 63
    if pad < 0:
        raise ValueError("target_size must be >= native 63")

    def _to_chw(state):
        frame = render_fn(state)
        scaled = jnp.clip(jnp.round(frame), 0, 255).astype(jnp.uint8)
        if pad:
            scaled = jnp.pad(scaled, ((0, pad), (0, pad), (0, 0)))
        return jnp.transpose(scaled, (2, 0, 1))

    def _labels(state):
        vitals = jnp.stack([getattr(state, f) for f in VITAL_FIELDS]).astype(jnp.float32)
        inventory = jnp.stack(
            [getattr(state.inventory, f) for f in INVENTORY_FIELDS]
        ).astype(jnp.float32)
        return vitals, inventory, state.achievements

    reset_v = jax.vmap(env.reset, in_axes=(0, None))
    step_v = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    render_v = jax.vmap(_to_chw)
    labels_v = jax.vmap(_labels)

    @jax.jit
    def _rollout(key):
        key, rk = jax.random.split(key)
        obs_sym, state = reset_v(jax.random.split(rk, num_envs), env_params)
        hidden = ScannedRNN.initialize_carry(num_envs, layer_size)

        def _step(carry, k):
            hidden, state, obs_sym, done_prev = carry
            ka, sk = jax.random.split(k)
            hidden, logits, _ = network.apply(params, hidden, (obs_sym[None], done_prev[None]))
            logits = logits[0]
            action = (jnp.argmax(logits, axis=-1) if greedy
                      else jax.random.categorical(ka, logits))
            obs2, state2, _, done, _ = step_v(
                jax.random.split(sk, num_envs), state, action, env_params)
            vitals, inventory, achievements = labels_v(state2)
            out = (render_v(state2), vitals, inventory, achievements, done)
            return (hidden, state2, obs2, done), out

        keys = jax.random.split(key, max_steps)
        _, outs = jax.lax.scan(
            _step, (hidden, state, obs_sym, jnp.zeros(num_envs, dtype=bool)), keys)
        return outs

    frames, vitals, inventory, achievements, episode_id = [], [], [], [], []
    produced = 0
    n_batches = (int(n_episodes) + num_envs - 1) // num_envs
    for batch in range(n_batches):
        f, v, inv, ach, dones = _rollout(jax.random.PRNGKey(int(seed) + batch))
        f = np.asarray(f); v = np.asarray(v); inv = np.asarray(inv)
        ach = np.asarray(ach).astype(bool); dones = np.asarray(dones).astype(bool)
        for n in range(num_envs):
            if produced >= int(n_episodes):
                break
            done_col = dones[:, n]
            length = int(np.argmax(done_col)) + 1 if done_col.any() else int(max_steps)
            keep = np.arange(0, length, int(stride))
            if keep.size < 2:
                continue
            frames.append(f[keep, n])
            vitals.append(v[keep, n])
            inventory.append(inv[keep, n])
            achievements.append(ach[keep, n])
            episode_id.append(np.full(keep.size, produced, dtype=np.int64))
            produced += 1

    if not frames:
        raise RuntimeError("no probe episodes collected")
    return ProbeData(
        frames=np.concatenate(frames).astype(np.uint8),
        vitals=np.concatenate(vitals).astype(np.float32),
        inventory=np.concatenate(inventory).astype(np.float32),
        achievements=np.concatenate(achievements).astype(bool),
        episode_id=np.concatenate(episode_id),
    )


def _cli() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Collect expert-frame oracle probe data.")
    p.add_argument("--params", type=Path,
                   default=repo_root / "d4_mamba_jepa/artifacts/expert/ppo_expert_v2.msgpack")
    p.add_argument("--out", type=Path,
                   default=repo_root / "d4_mamba_jepa/artifacts/expert/expert_probe_v1.probe_only.pt")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=2_500)
    p.add_argument("--stride", type=int, default=40)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--layer-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=90_000)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()

    data = collect_expert_probe_data(
        params_path=args.params, n_episodes=args.episodes,
        max_steps=args.max_steps, stride=args.stride, num_envs=args.num_envs,
        layer_size=args.layer_size, seed=args.seed, greedy=args.greedy,
    )
    save_probe_data(args.out, data)
    summary = {
        "out": str(args.out),
        "frames": int(data.frames.shape[0]),
        "episodes": int(len(np.unique(data.episode_id))),
        "stride": args.stride,
        "frames_per_episode_mean": float(
            data.frames.shape[0] / len(np.unique(data.episode_id))
        ),
        "achievement_names": achievement_names(),
        "mean_achievements_final": float(
            np.mean([
                data.achievements[data.episode_id == e][-1].sum()
                for e in np.unique(data.episode_id)
            ])
        ),
        "inventory_nonzero_fraction": float((data.inventory > 0).mean()),
        "vitals_std": [float(x) for x in data.vitals.std(axis=0)],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()


__all__ = ["collect_expert_probe_data"]
