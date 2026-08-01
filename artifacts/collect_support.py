"""Produce the d4mj support corpus: archived PPO expert with epsilon-greedy noise.

The archive (320 near-ceiling episodes, 68 terminals in 696,746 transitions) leaves
the continuation head almost no terminal supervision. This adds failure at several
distances from competent play -- epsilon in {0.1, 0.25, 0.5, 1.0} -- so the world
model sees deaths near good behaviour, not only the early deaths a random policy
produces.

Every rollout is kept, survivors included: acceptance-filtering for death would make
the corpus a death distribution rather than a behaviour distribution.

Episodes are written `uniform_eligible=True, bc_eligible=False`. They belong in the
world-model pool and must never reach behaviour cloning, however many achievements a
noisy rollout happens to unlock.

Structure follows d4_mamba_jepa/expert/generate.py, which is the audited path: one
episode per env slot, sliced at its first `done`, death read from lava/health rather
than inferred from the step cap.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "d4_mamba_jepa"))

from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT, BlockType
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
from craftax.craftax_env import make_craftax_env_from_name
from expert.ppo_expert import ScannedRNN, load_expert

from d4mj.config import Config
from d4mj.data import Episode, save_episodes

LAYER = 512
PARAMS = "d4_mamba_jepa/artifacts/expert/ppo_expert_v2.msgpack"


def rollout_batch(network, params, env, env_params, render_v, key, num_envs, epsilon, limit):
    reset_v = jax.vmap(env.reset, in_axes=(0, None))
    step_v = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def dead(state):
        lava = state.map[state.player_position[0], state.player_position[1]] == BlockType.LAVA.value
        return lava | (state.player_health <= 0)

    dead_v = jax.vmap(dead)
    n_actions = int(env.action_space(env_params).n)

    @jax.jit
    def scan(key):
        key, rk = jax.random.split(key)
        reset_keys = jax.random.split(rk, num_envs)
        obs, state = reset_v(reset_keys, env_params)
        hidden = ScannedRNN.initialize_carry(num_envs, LAYER)

        def body(carry, k):
            hidden, state, obs, done_prev = carry
            ka, kn, ku, sk = jax.random.split(k, 4)
            hidden, logits, _ = network.apply(params, hidden, (obs[None], done_prev[None]))
            greedy = jax.random.categorical(ka, logits[0])
            noise = jax.random.randint(kn, (num_envs,), 0, n_actions)
            take = jax.random.uniform(ku, (num_envs,)) < epsilon
            action = jnp.where(take, noise, greedy)
            obs2, state2, reward, done, _ = step_v(
                jax.random.split(sk, num_envs), state, action, env_params
            )
            return (hidden, state2, obs2, done), (
                render_v(state2), action, reward, done, state2.achievements, dead_v(state2)
            )

        keys = jax.random.split(key, limit)
        _, out = jax.lax.scan(body, (hidden, state, obs, jnp.zeros(num_envs, dtype=bool)), keys)
        return render_v(state), state.achievements, out, reset_keys

    return scan(key)


def to_episodes(frame0, ach0, out, limit):
    frames, actions, rewards, dones, achs, deads = (np.asarray(x) for x in out)
    frame0, ach0 = np.asarray(frame0), np.asarray(ach0)
    episodes = []
    for n in range(frames.shape[1]):
        column = dones[:, n]
        if column.any():
            end = int(np.argmax(column)) + 1
            terminal = bool(deads[end - 1, n])
        else:
            end, terminal = limit, False

        obs = np.concatenate([frame0[n][None], frames[:end, n]], axis=0)
        unlocked = np.concatenate([ach0[n][None], achs[:end, n]], axis=0).sum(-1)
        terminated = np.zeros(end, dtype=bool)
        truncated = np.zeros(end, dtype=bool)
        terminated[-1] = terminal
        truncated[-1] = not terminal
        episodes.append(
            Episode(
                observations=torch.from_numpy(np.ascontiguousarray(obs)),
                actions_taken=torch.from_numpy(actions[:end, n].astype(np.int64)),
                rewards=torch.from_numpy(rewards[:end, n].astype(np.float32)),
                terminated=torch.from_numpy(terminated),
                truncated=torch.from_numpy(truncated),
                events=torch.from_numpy(unlocked[1:] > unlocked[:-1]),
                uniform_eligible=True,
                bc_eligible=False,
            )
        )
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/craftax_support_v1.pt")
    parser.add_argument("--terminals", type=int, default=320)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=Config().seed)
    parser.add_argument("--epsilons", default="0.1,0.25,0.5,1.0")
    args = parser.parse_args()

    epsilons = [float(x) for x in args.epsilons.split(",")]
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    network, params, _ = load_expert(
        PARAMS,
        obs_dim=int(env.observation_space(env_params).shape[0]),
        action_dim=int(env.action_space(env_params).n),
        layer_size=LAYER,
    )
    render = make_craftax_pixel_renderer(int(BLOCK_PIXEL_SIZE_AGENT))

    def frame(state):
        pixels = jnp.clip(jnp.round(render(state)), 0, 255).astype(jnp.uint8)
        return pixels  # HWC, native 63x63, exactly what d4mj.patchify expects

    render_v = jax.vmap(frame)

    collected: list[Episode] = []
    terminals = 0
    start = time.time()
    round_index = 0
    while terminals < args.terminals:
        epsilon = epsilons[round_index % len(epsilons)]
        key = jax.random.PRNGKey(args.seed + round_index)
        frame0, ach0, out, _ = rollout_batch(
            network, params, env, env_params, render_v, key, args.num_envs, epsilon, args.limit
        )
        batch = to_episodes(frame0, ach0, out, args.limit)
        collected.extend(batch)
        terminals += sum(bool(e.terminated.any()) for e in batch)
        round_index += 1
        print(
            f"round {round_index:3d} eps={epsilon:<4} +{len(batch):3d} episodes "
            f"terminals {terminals}/{args.terminals} "
            f"steps {sum(len(e) for e in collected)} "
            f"[{time.time() - start:.0f}s]",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_episodes(out_path, collected)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "format": "d4mj_support_v1",
        "episodes": len(collected),
        "transitions": sum(len(e) for e in collected),
        "terminal_episodes": sum(bool(e.terminated.any()) for e in collected),
        "terminal_transitions": sum(int(e.terminated.sum()) for e in collected),
        "task_events": sum(int(e.events.sum()) for e in collected),
        "mean_length": sum(len(e) for e in collected) / len(collected),
        "epsilons": epsilons,
        "num_envs": args.num_envs,
        "limit": args.limit,
        "seed": args.seed,
        "rounds": round_index,
        "expert_params": PARAMS,
        "expert_sha256": hashlib.sha256(Path(PARAMS).read_bytes()).hexdigest(),
        "replay_sha256": digest,
        "uniform_eligible": True,
        "bc_eligible": False,
        "seconds": time.time() - start,
    }
    out_path.with_suffix(".pt.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
