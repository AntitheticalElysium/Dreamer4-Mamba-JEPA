"""Stream a large, versioned epsilon-support corpus into mmap-able shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import numpy as np
import torch

from artifacts.collect_support import (
    LAYER,
    PARAMS,
    rollout_batch,
)
from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
from craftax.craftax_env import make_craftax_env_from_name
from expert.ppo_expert import load_expert

from d4mj.config import Config
from d4mj.data import (
    STORE_FORMAT,
    Episode,
    atomic_manifest,
    save_episode_shard,
)

VERSION = "d4mj_support_v2"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assigned_split(seed: int, round_index: int, slot: int) -> str:
    value = hashlib.sha256(f"{seed}:{round_index}:{slot}".encode()).digest()
    draw = int.from_bytes(value[:8], "little") % 1000
    return "train" if draw < 800 else "dev" if draw < 900 else "final"


def to_episodes(frame0, ach0, out, limit, *, epsilon, seed, round_index):
    frames, actions, rewards, dones, achs, deads = (np.asarray(x) for x in out)
    frame0, ach0 = np.asarray(frame0), np.asarray(ach0)
    episodes = []
    for slot in range(frames.shape[1]):
        column = dones[:, slot]
        if column.any():
            end = int(np.argmax(column)) + 1
            terminal = bool(deads[end - 1, slot])
        else:
            end, terminal = limit, False
        obs = np.concatenate([frame0[slot][None], frames[:end, slot]], axis=0)
        unlocked = np.concatenate([ach0[slot][None], achs[:end, slot]], axis=0).sum(-1)
        terminated = np.zeros(end, dtype=bool)
        truncated = np.zeros(end, dtype=bool)
        terminated[-1] = terminal
        truncated[-1] = not terminal
        episodes.append(
            Episode(
                observations=torch.from_numpy(np.ascontiguousarray(obs)),
                actions_taken=torch.from_numpy(actions[:end, slot].astype(np.int64)),
                rewards=torch.from_numpy(rewards[:end, slot].astype(np.float32)),
                terminated=torch.from_numpy(terminated),
                truncated=torch.from_numpy(truncated),
                events=torch.from_numpy(unlocked[1:] > unlocked[:-1]),
                uniform_eligible=True,
                bc_eligible=False,
                epsilon=float(epsilon),
                split=assigned_split(seed, round_index, slot),
                episode_id=f"support-v2:{seed}:{round_index}:{slot}",
                terminal_cause="unspecified_death" if terminal else None,
            )
        )
    return episodes


def new_manifest(args, epsilons: list[float]) -> dict:
    params = Path(PARAMS)
    return {
        "format": STORE_FORMAT,
        "kind": VERSION,
        "complete": False,
        "target_terminal_episodes": args.terminals,
        "episodes": 0,
        "transitions": 0,
        "terminal_episodes": 0,
        "terminal_transitions": 0,
        "task_events": 0,
        "shards": [],
        "rounds": 0,
        "epsilons": epsilons,
        "epsilon_episode_counts": {str(value): 0 for value in epsilons},
        "epsilon_terminal_counts": {str(value): 0 for value in epsilons},
        "split_episode_counts": {name: 0 for name in ("train", "dev", "final")},
        "split_terminal_counts": {name: 0 for name in ("train", "dev", "final")},
        "num_envs": args.num_envs,
        "limit": args.limit,
        "seed": args.seed,
        "expert_params": PARAMS,
        "expert_sha256": file_digest(params),
        "collector_sha256": file_digest(Path(__file__)),
        "uniform_eligible": True,
        "bc_eligible": False,
        "split_rule": "sha256(seed:round:slot), 80/10/10 train/dev/final",
        "retention": "every rollout retained; stopping occurs after a complete env batch",
        "started_unix": time.time(),
        "seconds": 0.0,
    }


def validate_resume(manifest: dict, args, epsilons: list[float]) -> None:
    expected = {
        "format": STORE_FORMAT,
        "kind": VERSION,
        "target_terminal_episodes": args.terminals,
        "epsilons": epsilons,
        "num_envs": args.num_envs,
        "limit": args.limit,
        "seed": args.seed,
        "expert_sha256": file_digest(Path(PARAMS)),
    }
    observed = {key: manifest.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"support-v2 resume contract changed: {observed} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/craftax_support_v2"))
    parser.add_argument("--terminals", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=Config().seed + 10_000)
    parser.add_argument("--epsilons", default="0.1,0.25,0.5,1.0")
    args = parser.parse_args()
    epsilons = [float(value) for value in args.epsilons.split(",")]
    if args.terminals < 1 or args.num_envs < 1 or not epsilons:
        parser.error("terminals, num-envs and the epsilon ladder must be non-empty")

    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        validate_resume(manifest, args, epsilons)
        if manifest["complete"]:
            print(f"already complete: {manifest_path}")
            return
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        manifest = new_manifest(args, epsilons)
        atomic_manifest(manifest_path, manifest)

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
        import jax.numpy as jnp

        return jnp.clip(jnp.round(render(state)), 0, 255).astype(jnp.uint8)

    render_v = jax.vmap(frame)
    start = time.time()
    while manifest["terminal_episodes"] < args.terminals:
        round_index = int(manifest["rounds"])
        epsilon = epsilons[round_index % len(epsilons)]
        key = jax.random.PRNGKey(args.seed + round_index)
        frame0, ach0, out, _ = rollout_batch(
            network,
            params,
            env,
            env_params,
            render_v,
            key,
            args.num_envs,
            epsilon,
            args.limit,
        )
        episodes = to_episodes(
            frame0,
            ach0,
            out,
            args.limit,
            epsilon=epsilon,
            seed=args.seed,
            round_index=round_index,
        )
        shard_path = args.out / f"shard-{round_index:06d}.pt"
        if shard_path.exists():
            raise FileExistsError(f"unregistered support shard exists: {shard_path}")
        record = save_episode_shard(shard_path, episodes)
        record["epsilon"] = epsilon
        manifest["shards"].append(record)
        manifest["rounds"] += 1
        manifest["episodes"] += len(episodes)
        manifest["transitions"] += sum(len(episode) for episode in episodes)
        manifest["terminal_episodes"] += sum(bool(e.terminated.any()) for e in episodes)
        manifest["terminal_transitions"] += sum(int(e.terminated.sum()) for e in episodes)
        manifest["task_events"] += sum(int(e.events.sum()) for e in episodes)
        manifest["epsilon_episode_counts"][str(epsilon)] += len(episodes)
        manifest["epsilon_terminal_counts"][str(epsilon)] += sum(
            bool(e.terminated.any()) for e in episodes
        )
        for episode in episodes:
            manifest["split_episode_counts"][episode.split] += 1
            manifest["split_terminal_counts"][episode.split] += bool(
                episode.terminated.any()
            )
        manifest["seconds"] += time.time() - start
        start = time.time()
        atomic_manifest(manifest_path, manifest)
        print(
            f"round {manifest['rounds']:4d} eps={epsilon:<4} "
            f"episodes {manifest['episodes']} terminals "
            f"{manifest['terminal_episodes']}/{args.terminals} "
            f"steps {manifest['transitions']}",
            flush=True,
        )

    manifest["complete"] = True
    manifest["completed_unix"] = time.time()
    manifest["mean_length"] = manifest["transitions"] / manifest["episodes"]
    atomic_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
