"""Step-4 final evaluation set: seeds 79-94, canonical deterministic collector.

Per reviews/2026-07-14-step4-protocol.md: equal branches (3), TRUE common RNG
across suffixes (canonical forking makes this hold; every branch is verified
bit-exact on repeat), 8 day / 4 night anchors per seed, hashed on completion.
"""
from __future__ import annotations

import copy
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
    canonical_snapshot, canonicalize, chw, run_branches_canonical)
from fork_oracle_v2 import PREFIX, SUFFIX, _task_signature, sha256_file  # noqa: E402

OUT = REPO_ROOT / "data" / "final_bundle_79_94.pt"
SEEDS = tuple(range(79, 95))
DAY_QUOTA, NIGHT_QUOTA = 8, 4
BRANCHES = 3
SUFFIX_NAMES = ("true", "alt0", "alt1", "alt2")


def collect(seeds=SEEDS, day_quota=DAY_QUOTA, night_quota=NIGHT_QUOTA,
            verify_repeat=True):
    """Deterministic anchor collection. The LIVE env is canonicalized after
    every reset (2026-07-15 companion critical finding: canonicalizing only
    the branch snapshots leaves anchor DISCOVERY dependent on identity-set
    iteration order, so two runs picked different anchors)."""
    import crafter

    anchors = []
    for env_seed in seeds:
        env = crafter.Env(seed=env_seed, length=100_000)
        rng = np.random.default_rng(env_seed)
        obs = env.reset()
        canonicalize(env)
        obs_hist, act_hist = [chw(obs)], []
        day_left, night_left = day_quota, night_quota
        done, since = False, 0
        while day_left or night_left:
            if done:
                obs = env.reset()
                canonicalize(env)
                obs_hist, act_hist, done, since = [chw(obs)], [], False, 0
            daylight = float(env._world.daylight)
            is_night = daylight < 0.5
            ready = len(obs_hist) >= PREFIX and len(act_hist) >= PREFIX and since >= 10
            wanted = (is_night and night_left) or ((not is_night) and day_left)
            if ready and wanted:
                snapshot = canonical_snapshot(env)
                suffixes = {
                    name: [int(rng.integers(env.action_space.n)) for _ in range(SUFFIX)]
                    for name in SUFFIX_NAMES
                }
                base = 500_000 + 977 * len(anchors)   # SAME base for all suffixes
                anchor = {
                    "env_seed": env_seed, "daylight": daylight, "night": is_night,
                    "player_pos": np.asarray(env._player.pos, dtype=np.int64),
                    "obs_hist": np.stack(obs_hist[-PREFIX:]).astype(np.uint8),
                    "act_hist": np.asarray(act_hist[-PREFIX:], dtype=np.int64),
                    "suffixes": suffixes, "branches": {},
                }
                for name, suf in suffixes.items():
                    fr, oc, pos = run_branches_canonical(
                        snapshot, suf, base, BRANCHES, SUFFIX,
                        _task_signature, verify_repeat=verify_repeat)
                    anchor["branches"][name] = {
                        "frames": fr, "outcomes": oc, "positions": pos}
                live_done = False
                for a in suffixes["true"]:
                    obs, _, live_done, info = env.step(a)
                    obs_hist.append(chw(obs))
                    act_hist.append(a)
                    if live_done:
                        break
                anchors.append(anchor)
                done = live_done
                if is_night:
                    night_left -= 1
                else:
                    day_left -= 1
                since = 0
                del snapshot
                continue
            a = int(rng.integers(env.action_space.n))
            obs, _, done, _ = env.step(a)
            obs_hist.append(chw(obs))
            act_hist.append(a)
            obs_hist = obs_hist[-(PREFIX + 1):]
            act_hist = act_hist[-(PREFIX + 1):]
            since += 1
        del env
        print(f"[79-94] seed {env_seed} done ({len(anchors)} anchors)", flush=True)
    return anchors


def main():
    anchors = collect()
    torch.save(anchors, OUT)
    manifest = {
        "bundle": str(OUT), "sha256": sha256_file(OUT),
        "anchors": len(anchors),
        "night": int(sum(a["night"] for a in anchors)),
        "seeds": list(SEEDS), "branches": BRANCHES,
        "canonical_collector": True, "live_env_canonical": True,
        "verify_repeat": True,
    }
    (REPO_ROOT / "reviews" / "artifacts" / "final_bundle_79_94.manifest.json"
     ).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
