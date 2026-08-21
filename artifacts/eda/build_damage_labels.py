"""Factual damage labels for every TRAIN transition, from the rendered status bar.

Health is drawn as one 7x7 glyph -- exactly one patch under `Config.patch = 7` -- so
the label needs no simulator at scale. The glyph templates are fitted against
simulator ground truth, checked exactly on held-out episodes, and the run asserts
that no frame in the corpus fails to decode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import corpus
import replay

BAR = slice(49, 56)
COLUMN = {"health": slice(0, 7), "food": slice(7, 14),
          "drink": slice(14, 21), "energy": slice(21, 28)}
FIT = [(0, 0), (0, 5), (1, 3), (2, 7), (3, 11), (100, 2), (200, 7), (300, 19),
       (419, 23), (50, 4)]
CHECK = [(7, 1), (77, 17), (177, 5), (377, 20)]


def trajectory_vitals(shard: int, slot: int) -> np.ndarray:
    """(T+1, 4) health/food/drink/energy from the simulator, via one compiled scan."""
    import jax
    import jax.numpy as jnp

    env, params, _, _, _ = replay.env_and_render()
    reset_key, env_keys = replay._slot_keys(shard, slot)
    actions = replay.episode_fields(shard, slot)["actions_taken"].numpy()
    n = len(actions)

    def vitals(state):
        return jnp.stack([state.player_health, state.player_food,
                          state.player_drink, state.player_energy]).astype(jnp.float32)

    def run(reset_key, keys, acts):
        _, state = env.reset(reset_key, params)

        def body(carry, xs):
            key, action = xs
            _, nxt, _, _, _ = env.step(key, carry, action, params)
            return nxt, vitals(nxt)

        _, trace = jax.lax.scan(body, state, (keys, acts))
        return jnp.concatenate([vitals(state)[None], trace])

    return np.asarray(jax.jit(run)(reset_key, env_keys[:n], jnp.asarray(actions[:n])))


def fit_templates() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    seen: dict[str, dict[bytes, float]] = {field: {} for field in COLUMN}
    for shard, slot in FIT:
        frames = replay.episode_fields(shard, slot)["observations"].numpy()
        truth = trajectory_vitals(shard, slot)
        for i in range(0, len(frames), 3):
            for j, (field, column) in enumerate(COLUMN.items()):
                key = np.ascontiguousarray(frames[i, BAR, column]).tobytes()
                value = float(truth[i, j])
                if seen[field].get(key, value) != value:
                    raise AssertionError(f"glyph collision in {field}")
                seen[field][key] = value
    prepared = {}
    for field, entries in seen.items():
        prepared[field] = (
            np.stack([np.frombuffer(k, dtype=np.uint8) for k in entries]),
            np.array(list(entries.values()), dtype=np.float32),
        )
        print(f"  {field}: {len(entries)} glyph templates", flush=True)
    return prepared


def decode(frames: np.ndarray, field: str, prepared) -> np.ndarray:
    templates, values = prepared[field]
    window = np.ascontiguousarray(frames[:, BAR, COLUMN[field]])
    match = (window.reshape(len(window), -1)[:, None, :] == templates[None]).all(-1)
    found = match.any(1)
    out = np.full(len(frames), np.nan, dtype=np.float32)
    out[found] = values[match[found].argmax(1)]
    return out


def main() -> None:
    prepared = fit_templates()
    for shard, slot in CHECK:
        frames = replay.episode_fields(shard, slot)["observations"].numpy()
        truth = trajectory_vitals(shard, slot)
        for j, field in enumerate(COLUMN):
            got = decode(frames, field, prepared)
            assert not np.isnan(got).any(), f"undecoded {field} at {shard}:{slot}"
            assert np.array_equal(got, truth[:, j]), f"decode mismatch in {field}"
    print(f"held-out check: exact on {len(CHECK)} episodes, all four fields", flush=True)

    rows = corpus.train_rows()
    corpus.verify_order(rows)
    off, foff = corpus.offsets(rows), corpus.frame_offsets(rows)
    health = np.full(foff[-1], np.nan, dtype=np.float32)

    manifest = json.loads((corpus.SUPPORT / "manifest.json").read_text())
    archive = torch.load(corpus.ARCHIVE, weights_only=False, mmap=True)
    by_shard: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        by_shard.setdefault(row["shard"], []).append(i)

    done = 0
    for shard, indices in sorted(by_shard.items()):
        payload = (torch.load(corpus.SUPPORT / manifest["shards"][shard]["file"],
                              weights_only=False, mmap=True) if shard >= 0 else None)
        for i in indices:
            row = rows[i]
            frames = (payload["episodes"][row["slot"]]["observations"].numpy()
                      if payload is not None
                      else archive[row["slot"]]["obs"][:, :, :63, :63]
                      .permute(0, 2, 3, 1).numpy())
            health[foff[i] : foff[i + 1]] = decode(frames, "health", prepared)
            done += 1
        del payload
        if done % 1000 < 24:
            print(f"  decoded {done}/{len(rows)} episodes", flush=True)
    assert not np.isnan(health).any(), "undecoded frames remain"

    delta = np.zeros(off[-1], dtype=np.float32)
    for i in range(len(rows)):
        series = health[foff[i] : foff[i + 1]]
        delta[off[i] : off[i + 1]] = series[1:] - series[:-1]
    damage = delta < 0
    print(f"TRAIN transitions {off[-1]:,}: damaging {int(damage.sum()):,} "
          f"({damage.mean():.3%}), healing {int((delta > 0).sum()):,} "
          f"({(delta > 0).mean():.3%})")
    np.savez(HERE / "damage_labels.npz", offsets=off, frame_offsets=foff,
             health=health, delta=delta, damage=damage)
    print("wrote damage_labels.npz", flush=True)


if __name__ == "__main__":
    main()
