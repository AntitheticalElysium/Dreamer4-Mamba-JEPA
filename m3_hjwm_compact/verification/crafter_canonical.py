"""Deterministic Crafter forking (2026-07-14 companion finding #2).

Pinned Crafter stores per-chunk objects in identity-hashed sets
(engine.py: `_chunks = defaultdict(set)`), iterated during creature balancing
(env.py:162). Deep-copied forks therefore order creatures differently despite
identical RNG state, so "common simulator RNG across suffixes" did not hold.

Fix (wrapper-level; third_party stays pristine): replace chunk sets with a set
subclass whose iteration order is sorted by object position — positions are
unique per object (engine.World.add asserts one object per cell) and identical
across copies — plus defensive rebinding of every object's `.random` to the
fork's world RNG. `run_branches_canonical` additionally verifies bit-exact
repeatability when asked.
"""
from __future__ import annotations

import copy

import numpy as np


class DeterministicChunk(set):
    """Set with copy-stable iteration order (by object position, then type)."""

    def __iter__(self):
        items = list(super().__iter__())
        items.sort(key=lambda obj: (int(obj.pos[0]), int(obj.pos[1]),
                                    type(obj).__name__))
        return iter(items)


def canonicalize(env) -> None:
    """Make an env (or snapshot) fork-deterministic, in place."""
    world = env._world
    for key in list(world._chunks):
        world._chunks[key] = DeterministicChunk(world._chunks[key])
    world._chunks.default_factory = DeterministicChunk
    for obj in world.objects:
        obj.random = world.random


def canonical_snapshot(env):
    snapshot = copy.deepcopy(env)
    canonicalize(snapshot)
    return snapshot


def chw(x):
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def run_branches_canonical(snapshot, suffix, base_seed, branches, suffix_len,
                           task_signature, verify_repeat=False):
    """Reseeded continuations from a canonicalized snapshot. With
    `verify_repeat`, every branch is executed twice and must be bit-exact."""
    frames, outcomes, positions = [], [], []
    for b in range(branches):
        runs = []
        for _ in range(2 if verify_repeat else 1):
            fork = copy.deepcopy(snapshot)
            canonicalize(fork)
            fork._world.random.seed(base_seed + b)
            obs_seq, pos_seq, reward_sum, terminated = [], [], 0.0, False
            info = {}
            for a in suffix:
                obs, r, done, info = fork.step(a)
                obs_seq.append(chw(obs))
                pos_seq.append(np.asarray(info["player_pos"], dtype=np.int64))
                reward_sum += float(r)
                if done:
                    terminated = True
                    while len(obs_seq) < suffix_len:
                        obs_seq.append(obs_seq[-1])
                        pos_seq.append(pos_seq[-1])
                    break
            runs.append((np.stack(obs_seq), np.stack(pos_seq), reward_sum,
                         terminated, task_signature(info)))
            del fork
        if verify_repeat:
            same = (np.array_equal(runs[0][0], runs[1][0])
                    and runs[0][2] == runs[1][2] and runs[0][3] == runs[1][3]
                    and runs[0][4] == runs[1][4])
            if not same:
                raise RuntimeError(
                    f"non-repeatable branch (base_seed={base_seed}, b={b}) "
                    "after canonicalization")
        obs_seq, pos_seq, reward_sum, terminated, sig = runs[0]
        frames.append(obs_seq)
        positions.append(pos_seq)
        outcomes.append({"reward_sum": reward_sum, "terminated": terminated, **sig})
    return np.stack(frames).astype(np.uint8), outcomes, np.stack(positions)
