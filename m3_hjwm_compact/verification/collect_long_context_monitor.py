"""Canonical Step-4b monitor collector with 128-frame real prefixes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from crafter_canonical import (  # noqa: E402
    canonical_snapshot,
    canonicalize,
    chw,
    run_branches_canonical,
)
from fork_oracle_v2 import _task_signature, sha256_file  # noqa: E402


OUT = REPO_ROOT / "data" / "long_context_monitor_111_114.pt"
MANIFEST = REPO_ROOT / "reviews" / "artifacts" / \
    "long_context_monitor_111_114.manifest.json"
SEEDS = (111, 112, 113, 114)
PREFIX, SUFFIX = 128, 8
DAY_QUOTA, NIGHT_QUOTA = 4, 2
BRANCHES = 3
SUFFIX_NAMES = ("true", "alt0", "alt1", "alt2")


def collect(seeds=SEEDS, day_quota=DAY_QUOTA, night_quota=NIGHT_QUOTA,
            verify_repeat=True):
    import crafter

    anchors = []
    for env_seed in seeds:
        env = crafter.Env(seed=env_seed, length=100_000)
        rng = np.random.default_rng(env_seed)
        observation = env.reset()
        canonicalize(env)
        observations, actions = [chw(observation)], []
        day_left, night_left = day_quota, night_quota
        done, since_anchor = False, 0
        while day_left or night_left:
            if done:
                observation = env.reset()
                canonicalize(env)
                observations, actions = [chw(observation)], []
                done, since_anchor = False, 0
            is_night = float(env._world.daylight) < 0.5
            ready = (len(observations) >= PREFIX
                     and len(actions) >= PREFIX and since_anchor >= 16)
            wanted = (is_night and night_left) or ((not is_night) and day_left)
            if ready and wanted:
                snapshot = canonical_snapshot(env)
                suffixes = {
                    name: [int(rng.integers(env.action_space.n))
                           for _ in range(SUFFIX)]
                    for name in SUFFIX_NAMES
                }
                base_seed = 700_000 + 977 * len(anchors)
                anchor = {
                    "env_seed": int(env_seed),
                    "daylight": float(env._world.daylight),
                    "night": bool(is_night),
                    "player_pos": np.asarray(env._player.pos, dtype=np.int64),
                    "obs_hist": np.stack(observations[-PREFIX:]).astype(np.uint8),
                    "act_hist": np.asarray(actions[-PREFIX:], dtype=np.int64),
                    "suffixes": suffixes,
                    "branches": {},
                }
                for name, suffix in suffixes.items():
                    frames, outcomes, positions = run_branches_canonical(
                        snapshot, suffix, base_seed, BRANCHES, SUFFIX,
                        _task_signature, verify_repeat=verify_repeat)
                    anchor["branches"][name] = {
                        "frames": frames,
                        "outcomes": outcomes,
                        "positions": positions,
                    }
                live_done = False
                for action in suffixes["true"]:
                    observation, _, live_done, _ = env.step(action)
                    observations.append(chw(observation))
                    actions.append(action)
                    if live_done:
                        break
                anchors.append(anchor)
                done = live_done
                if is_night:
                    night_left -= 1
                else:
                    day_left -= 1
                since_anchor = 0
                del snapshot
                continue

            action = int(rng.integers(env.action_space.n))
            observation, _, done, _ = env.step(action)
            observations.append(chw(observation))
            actions.append(action)
            observations = observations[-(PREFIX + 1):]
            actions = actions[-(PREFIX + 1):]
            since_anchor += 1
        del env
        print(f"[long monitor] seed {env_seed} done ({len(anchors)} anchors)",
              flush=True)
    return anchors


def write_bundle(anchors):
    torch.save(anchors, OUT)
    manifest = {
        "protocol": "reviews/2026-07-16-long-context-scale-protocol.md",
        "bundle": str(OUT),
        "sha256": sha256_file(OUT),
        "anchors": len(anchors),
        "night": int(sum(anchor["night"] for anchor in anchors)),
        "seeds": list(SEEDS),
        "prefix": PREFIX,
        "suffix": SUFFIX,
        "branches": BRANCHES,
        "canonical_collector": True,
        "live_env_canonical": True,
        "verify_repeat": True,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    manifest = write_bundle(collect())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
