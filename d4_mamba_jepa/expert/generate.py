"""Generate an offline expert replay by rolling a trained PPO policy in Craftax.

Architecture (per the verified sources -- Craftax paper 2402.16801 gets speed
from JAX vectorization + compilation, and CrafterDojo separates rollout from a
vmapped/jitted batched render):

  * VECTORIZED rollout: N independent envs stepped together under one jitted
    ``lax.scan`` (no per-step Python/GPU-launch overhead, no CPU-GPU transfer
    inside the rollout). ONE episode per env slot -- after a slot's ``done`` we
    keep stepping (masked) and slice each slot at its FIRST done, so episode
    boundaries and terminal frames are exact (no auto-reset ambiguity).
  * The pixel render is VMAPPED across the N slots inside the scan (parallel,
    not the launch-bound single-env render). Rendered at 7px -> padded 64x64,
    byte-identical to ``craftax_env``.

The expert acts on the SYMBOLIC observation (what it was trained on). Each saved
episode carries per-frame cumulative achievements; full contiguous trajectories
are kept (world-model contiguity), noop balance reported not filtered.

Imports JAX/Craftax + torch (.pt replay) -- run as a generation job on the free
GPU after training. Memory is ~ max_steps * num_envs * 3*target^2 bytes for the
frame buffer; size num_envs/max_steps accordingly.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch

from ..craftax_env import N_ACHIEVEMENTS, N_ACTIONS, achievement_names
from .ppo_expert import ENV_NAME, REFERENCE, load_expert

FORMAT = "d4_mamba_jepa_craftax_expert_replay_v1"


@dataclass
class ExpertReplayManifest:
    format: str
    policy: str
    reference: str
    params_path: str
    params_sha256: str
    n_episodes: int
    n_transitions: int
    replay_sha256: str
    achievement_names: list
    mean_achievements: float
    deep_achievement_episodes: int
    noop_fraction: float
    mean_episode_length: float
    truncated_episodes: int
    greedy: bool
    seed: int
    num_envs: int
    max_steps: int
    created: str


def generate_expert_replay(
    *,
    params_path: str | Path,
    out_path: str | Path,
    n_episodes: int,
    max_steps: int,
    layer_size: int = 512,
    seed: int = 0,
    greedy: bool = False,
    target_size: int = 64,
    num_envs: int = 16,
) -> ExpertReplayManifest:
    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name
    from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
    from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT
    from .ppo_expert import ScannedRNN

    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    env_params = env.default_params
    max_timesteps = int(env_params.max_timesteps)
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
        frame = render_fn(state)                       # HWC [0,255]
        scaled = jnp.clip(jnp.round(frame), 0, 255).astype(jnp.uint8)
        if pad:
            scaled = jnp.pad(scaled, ((0, pad), (0, pad), (0, 0)))
        return jnp.transpose(scaled, (2, 0, 1))        # CHW uint8

    reset_v = jax.vmap(env.reset, in_axes=(0, None))
    step_v = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    render_v = jax.vmap(_to_chw)                        # [N,...] states -> [N,3,H,W]

    @jax.jit
    def _rollout(key):
        key, rk = jax.random.split(key)
        obs_sym, state = reset_v(jax.random.split(rk, num_envs), env_params)
        hidden = ScannedRNN.initialize_carry(num_envs, layer_size)
        frame0 = render_v(state)                        # [N,3,H,W]
        ach0 = state.achievements                       # [N,22]

        def _step(carry, k):
            hidden, state, obs_sym, done_prev = carry
            ka, sk = jax.random.split(k)
            ac_in = (obs_sym[None], done_prev[None])    # [1,N,obs], [1,N]
            hidden, logits, _ = network.apply(params, hidden, ac_in)
            logits = logits[0]                          # [N,17]
            action = (jnp.argmax(logits, axis=-1) if greedy
                      else jax.random.categorical(ka, logits))
            obs2, state2, reward, done, _ = step_v(
                jax.random.split(sk, num_envs), state, action, env_params)
            out = (render_v(state2), action, reward, done,
                   state2.achievements, state2.timestep)
            return (hidden, state2, obs2, done), out

        keys = jax.random.split(key, max_steps)
        _, (frames, actions, rewards, dones, achs, ts) = jax.lax.scan(
            _step, (hidden, state, obs_sym, jnp.zeros(num_envs, dtype=bool)), keys)
        return frame0, ach0, frames, actions, rewards, dones, achs, ts

    records: list[dict] = []
    total_transitions = 0
    total_noop = 0
    deep_episodes = 0
    truncated = 0
    lengths: list[int] = []
    achievement_totals = np.zeros(N_ACHIEVEMENTS, dtype=np.int64)
    names = achievement_names()
    deep_idx = [i for i, n in enumerate(names)
                if n.startswith(("place_", "make_")) or n in
                {"collect_coal", "collect_iron", "collect_diamond", "eat_plant",
                 "defeat_skeleton", "wake_up"}]

    n_batches = (int(n_episodes) + num_envs - 1) // num_envs
    produced = 0
    for batch in range(n_batches):
        frame0, ach0, frames, actions, rewards, dones, achs, ts = _rollout(
            jax.random.PRNGKey(int(seed) + batch))
        # one transfer of the whole batch
        frame0 = np.asarray(frame0); ach0 = np.asarray(ach0).astype(bool)
        frames = np.asarray(frames); actions = np.asarray(actions)
        rewards = np.asarray(rewards); dones = np.asarray(dones).astype(bool)
        achs = np.asarray(achs).astype(bool); ts = np.asarray(ts).astype(np.int64)
        # frames [T,N,3,H,W]; slice each env slot at its first done.
        for n in range(num_envs):
            if produced >= int(n_episodes):
                break
            done_col = dones[:, n]
            if done_col.any():
                t_end = int(np.argmax(done_col))
                L = t_end + 1
                terminal = int(ts[t_end, n]) < max_timesteps
            else:
                L = int(max_steps)
                terminal = False
                truncated += 1
            obs_arr = np.concatenate([frame0[n][None], frames[:L, n]], axis=0)
            ach_arr = np.concatenate([ach0[n][None], achs[:L, n]], axis=0)
            act_arr = actions[:L, n].astype(np.int64)
            rew_arr = rewards[:L, n].astype(np.float32)
            cont = np.ones(L, dtype=np.float32)
            if terminal:
                cont[-1] = 0.0
            total_noop += int((act_arr == 0).sum())
            total_transitions += L
            lengths.append(L)
            achievement_totals += ach_arr[-1].astype(np.int64)
            if int(ach_arr[-1][deep_idx].sum()) > 0:
                deep_episodes += 1
            records.append({
                "obs": torch.from_numpy(obs_arr.astype(np.uint8)),
                "actions": torch.from_numpy(act_arr),
                "rewards": torch.from_numpy(rew_arr),
                "continues": torch.from_numpy(cont),
                "achievements": torch.from_numpy(ach_arr),
                "env_seed": int(seed) + batch * num_envs + n,
            })
            produced += 1

    buffer = io.BytesIO()
    torch.save(records, buffer)
    data = buffer.getvalue()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)

    manifest = ExpertReplayManifest(
        format=FORMAT, policy="ppo_expert", reference=REFERENCE,
        params_path=str(params_path),
        params_sha256=hashlib.sha256(Path(params_path).read_bytes()).hexdigest(),
        n_episodes=len(records), n_transitions=total_transitions,
        replay_sha256=hashlib.sha256(data).hexdigest(),
        achievement_names=names,
        mean_achievements=float(achievement_totals.sum() / max(1, len(records))),
        deep_achievement_episodes=deep_episodes,
        noop_fraction=float(total_noop / max(1, total_transitions)),
        mean_episode_length=float(np.mean(lengths)) if lengths else 0.0,
        truncated_episodes=truncated, greedy=greedy, seed=int(seed),
        num_envs=int(num_envs), max_steps=int(max_steps),
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    (out.with_suffix(out.suffix + ".manifest.json")).write_text(
        json.dumps(asdict(manifest), indent=2) + "\n")
    return manifest


def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate an expert Craftax replay.")
    p.add_argument("--params", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--layer-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    m = generate_expert_replay(
        params_path=args.params, out_path=args.out, n_episodes=args.episodes,
        max_steps=args.max_steps, num_envs=args.num_envs,
        layer_size=args.layer_size, seed=args.seed, greedy=args.greedy)
    print(json.dumps(asdict(m), indent=2))


if __name__ == "__main__":
    _cli()


__all__ = ["FORMAT", "ExpertReplayManifest", "generate_expert_replay"]
