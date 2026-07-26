"""Generate an offline expert replay by rolling a trained PPO policy in Craftax.

The expert acts on the SYMBOLIC observation (what it was trained on) while we
render the 64x64 pixel observation from the SAME ``EnvState`` -- rendered at 7px
(``BLOCK_PIXEL_SIZE_AGENT``) and padded to 64, byte-identical to what
``craftax_env.CraftaxPixelEnv`` produces, so the dataset matches deployment. Each
saved episode carries per-frame cumulative achievements. The full contiguous
trajectory is kept (the world model needs contiguous transitions); noop balance
is reported as a stat and left to the BC sampler rather than filtered out here
(which would break dynamics contiguity).

Imports JAX/Craftax + torch (for the .pt replay) -- run as a generation job.
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
from .ppo_expert import ENV_NAME, load_expert

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
    greedy: bool
    seed: int
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
) -> ExpertReplayManifest:
    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name
    from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
    from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT

    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    env_params = env.default_params
    max_timesteps = int(env_params.max_timesteps)
    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)
    assert action_dim == N_ACTIONS, action_dim

    network, params, _ = load_expert(
        params_path, obs_dim=obs_dim, action_dim=action_dim, layer_size=layer_size
    )
    from .ppo_expert import ScannedRNN

    render_fn = make_craftax_pixel_renderer(int(BLOCK_PIXEL_SIZE_AGENT))
    pad = int(target_size) - 63
    if pad < 0:
        raise ValueError("target_size must be >= native 63")

    def _to_chw(hwc255):
        # render is HWC [0,255]; match craftax_env: round, bottom-right zero-pad
        # to target, transpose to CHW uint8. Done on-device inside the jit.
        scaled = jnp.clip(jnp.round(hwc255), 0, 255).astype(jnp.uint8)
        if pad:
            scaled = jnp.pad(scaled, ((0, pad), (0, pad), (0, 0)))
        return jnp.transpose(scaled, (2, 0, 1))

    @jax.jit
    def _reset_frame(state):
        return _to_chw(render_fn(state))

    @jax.jit
    def _advance(hidden, state, obs_sym, done_flag, key):
        # One jitted step: RNN act + env.step + render (CrafterDojo jits
        # network.apply and env.step; we fuse them so the whole step compiles).
        ka, sk = jax.random.split(key)
        ac_in = (obs_sym[None, None, :], done_flag[None, None])
        hidden, logits, _ = network.apply(params, hidden, ac_in)
        logits = logits[0, 0]
        action = jnp.argmax(logits) if greedy else jax.random.categorical(ka, logits)
        obs2, state2, reward, done, _ = env.step(sk, state, action, env_params)
        return (hidden, state2, obs2, action, reward, done,
                state2.timestep, state2.achievements, _to_chw(render_fn(state2)))

    records: list[dict] = []
    total_transitions = 0
    total_noop = 0
    deep_episodes = 0
    achievement_totals = np.zeros(N_ACHIEVEMENTS, dtype=np.int64)
    # "deep" = anything past the shallow trio (wood/sapling/stone).
    names = achievement_names()
    deep_idx = [i for i, n in enumerate(names)
                if n.startswith(("place_", "make_")) or n in
                {"collect_coal", "collect_iron", "collect_diamond", "eat_plant",
                 "defeat_skeleton", "wake_up"}]

    for episode in range(int(n_episodes)):
        key = jax.random.PRNGKey(int(seed) + episode)
        key, rk = jax.random.split(key)
        obs_sym, state = env.reset(rk, env_params)
        hidden = ScannedRNN.initialize_carry(1, layer_size)
        done_flag = jnp.zeros((), dtype=bool)

        # Everything accumulates ON DEVICE; the ONLY per-step sync is bool(done)
        # for the Python break (the raw env has no auto-reset, so we must stop at
        # done). Actions/rewards/timesteps/frames are stacked+transferred once.
        frames = [_reset_frame(state)]
        achievements = [jnp.zeros(N_ACHIEVEMENTS, dtype=bool)]
        act_dev, rew_dev, ts_dev = [], [], []
        broke_on_done = False
        for _ in range(int(max_steps)):
            key, k = jax.random.split(key)
            (hidden, state, obs_sym, action, reward, done,
             timestep, ach, pixel) = _advance(hidden, state, obs_sym, done_flag, k)
            frames.append(pixel)
            achievements.append(ach)
            act_dev.append(action)
            rew_dev.append(reward)
            ts_dev.append(timestep)
            done_flag = done
            if bool(done):          # the only per-step host sync
                broke_on_done = True
                break

        obs_arr = np.asarray(jnp.stack(frames)).astype(np.uint8)
        ach_arr = np.asarray(jnp.stack(achievements)).astype(bool)
        actions = np.asarray(jnp.stack(act_dev)).astype(np.int64)
        timesteps = np.asarray(jnp.stack(ts_dev)).astype(np.int64)
        rewards = np.asarray(jnp.stack(rew_dev)).astype(np.float32)
        # Only the final transition can be terminal (we break on done); a break
        # on death/lava (timestep < max) is absorbing, timeout/truncation is not.
        continues = np.ones(len(actions), dtype=np.float32)
        if broke_on_done and int(timesteps[-1]) < max_timesteps:
            continues[-1] = 0.0
        total_noop += int((actions == 0).sum())
        achievement_totals += ach_arr[-1].astype(np.int64)
        if int(ach_arr[-1][deep_idx].sum()) > 0:
            deep_episodes += 1
        total_transitions += len(actions)
        records.append({
            "obs": torch.from_numpy(obs_arr),
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards),
            "continues": torch.from_numpy(continues),
            "achievements": torch.from_numpy(ach_arr),
            "env_seed": int(seed) + episode,
        })

    buffer = io.BytesIO()
    torch.save(records, buffer)
    data = buffer.getvalue()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)

    manifest = ExpertReplayManifest(
        format=FORMAT,
        policy="ppo_expert",
        reference=__import__("d4_mamba_jepa.expert.ppo_expert", fromlist=["REFERENCE"]).REFERENCE,
        params_path=str(params_path),
        params_sha256=hashlib.sha256(Path(params_path).read_bytes()).hexdigest(),
        n_episodes=len(records),
        n_transitions=total_transitions,
        replay_sha256=hashlib.sha256(data).hexdigest(),
        achievement_names=names,
        mean_achievements=float(achievement_totals.sum() / max(1, len(records))),
        deep_achievement_episodes=deep_episodes,
        noop_fraction=float(total_noop / max(1, total_transitions)),
        greedy=greedy,
        seed=int(seed),
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
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--layer-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    m = generate_expert_replay(
        params_path=args.params, out_path=args.out, n_episodes=args.episodes,
        max_steps=args.max_steps, layer_size=args.layer_size, seed=args.seed,
        greedy=args.greedy)
    print(json.dumps(asdict(m), indent=2))


if __name__ == "__main__":
    _cli()


__all__ = ["FORMAT", "ExpertReplayManifest", "generate_expert_replay"]
