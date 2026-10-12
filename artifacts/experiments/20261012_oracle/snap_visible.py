"""CPU-only exact visible facts for the existing oracle panel, with resumable rows.

The 1,534-field schema is observe.state_features, not Craftax's native 1,345-field
get_obs: tiles (63*17), mobs (63*7; z/cow/skeleton/arrows U/D/L/R), HUD /9,
facing one-hot, sleep/light, inventory /9. No image encoder is evaluated.

Usage: .venv/bin/python artifacts/experiments/20261012_oracle/snap_visible.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "artifacts/eda"))
import replay as RP  # noqa: E402

OUT = ROOT / "artifacts/eda/oracle_v1"
INVENTORY = ("wood", "stone", "coal", "iron", "diamond", "sapling",
             "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword",
             "stone_sword", "iron_sword")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_save(value, path: Path):
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def visible(state):
    """JAX equivalent of observe.state_features's visible vector."""
    pad = 9
    top = state.player_position - jnp.array([3, 4]) + pad
    view = jax.lax.dynamic_slice(jnp.pad(state.map, pad, constant_values=1), top, (7, 9))
    tiles = jax.nn.one_hot(view, 17).reshape(-1)
    mobs = jnp.zeros((63, 7), dtype=jnp.float32)
    for ch, group in enumerate((state.zombies, state.cows, state.skeletons)):
        loc = group.position - state.player_position + jnp.array([3, 4])
        on = group.mask & (loc >= 0).all(-1) & (loc < jnp.array([7, 9])).all(-1)
        cell = jnp.clip(loc[:, 0] * 9 + loc[:, 1], 0, 62)
        mobs = mobs.at[cell, ch].add(on.astype(jnp.float32))
    loc = state.arrows.position - state.player_position + jnp.array([3, 4])
    on = state.arrows.mask & (loc >= 0).all(-1) & (loc < jnp.array([7, 9])).all(-1)
    directions = state.arrow_directions
    ch = jnp.where((directions == jnp.array([1, 0])).all(-1), 4,
                   jnp.where((directions == jnp.array([0, -1])).all(-1), 5,
                             jnp.where((directions == jnp.array([0, 1])).all(-1), 6, 3)))
    cell = jnp.clip(loc[:, 0] * 9 + loc[:, 1], 0, 62)
    mobs = mobs.at[cell, ch].add(on.astype(jnp.float32))
    hud = jnp.array([state.player_health, state.player_food, state.player_drink,
                     state.player_energy], dtype=jnp.float32) / 9
    facing = jax.nn.one_hot(state.player_direction - 1, 4)
    flags = jnp.array([state.is_sleeping, state.light_level], dtype=jnp.float32)
    inventory = jnp.array([getattr(state.inventory, k) for k in INVENTORY], dtype=jnp.float32) / 9
    return jnp.concatenate([tiles, mobs.reshape(-1), hud, facing, flags, inventory])


@jax.jit
def window(state, keys, actions):
    env, params, _, _, _ = RP.env_and_render()
    first = (visible(state), env.get_obs(state))

    def body(state, ka):
        key, action = ka
        _, nxt, _, _, _ = env.step(key, state, action, params)
        return nxt, (visible(nxt), env.get_obs(nxt))

    _, seq = jax.lax.scan(body, state, (keys, actions))
    return tuple(jnp.concatenate([f[None], q]) for f, q in zip(first, seq))


def native_checks(v, native):
    if native.shape != (32, 1345):
        raise ValueError(f"Unexpected native symbolic shape {native.shape}")
    ntile = native[:, :1323].reshape(32, 63, 21)[..., :17]
    assert np.array_equal(ntile, v[:, :1071].reshape(32, 63, 17)), "native terrain mismatch"
    assert np.array_equal(native[:, 1339:1343], v[:, 1516:1520]), "native facing mismatch"
    assert np.array_equal(native[:, 1343:1345], v[:, [1521, 1520]]), "native light/sleep mismatch"
    # Native symbolic counters/inventory use fp16 /10, whereas observe uses fp32 /9.
    inv = (np.rint(v[:, 1522:1534] * 9).astype(np.float16) / np.float16(10)).astype(np.float32)
    hud = (np.rint(v[:, 1512:1516] * 9).astype(np.float16) / np.float16(10)).astype(np.float32)
    error = max(float(np.abs(inv - native[:, 1323:1335]).max()),
                float(np.abs(hud - native[:, 1335:1339]).max()))
    assert error <= 0.001, f"native HUD/inventory mismatch {error}"
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=OUT / "panel.pt")
    parser.add_argument("--out", type=Path, default=OUT / "snap_visible.pt")
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; use a separate --out")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if any(d.platform != "cpu" for d in jax.devices()):
        raise RuntimeError("This helper must run only on CPU")
    panel = torch.load(args.panel, map_location="cpu", weights_only=False, mmap=True)
    roots = panel["roots"][:args.limit or None]
    if panel["layouts"].shape[1:] != (32, 63, 4):
        raise ValueError("Expected the existing 32-frame oracle panel")
    import craftax.craftax_classic.renderer as renderer
    import craftax.craftax_classic.game_logic as game_logic
    import craftax.craftax_classic.world_gen as world_gen
    import craftax.craftax_classic.constants as constants
    from craftax.craftax_classic.envs import craftax_symbolic_env
    source_paths = [Path(__file__), Path(RP.__file__), Path(renderer.__file__),
                    Path(game_logic.__file__), Path(world_gen.__file__), Path(constants.__file__),
                    Path(craftax_symbolic_env.__file__),
                    ROOT / "artifacts/experiments/20260921_readout_ladder/observe.py"]
    contract = {"format": "oracle_snap_visible_v1", "panel_sha256": sha256(args.panel),
                "manifest_sha256": sha256(RP.STORE / "manifest.json"), "roots": roots,
                "sources": {str(p): sha256(p) for p in source_paths},
                "visible_shape": [len(roots), 32, 1534], "native_shape": [len(roots), 32, 1345],
                "jax_version": jax.__version__, "cpu_only": True}
    input_hash = digest(contract)
    chunks = args.out.parent / (args.out.stem + "_chunks")
    chunks.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        done = torch.load(args.out, map_location="cpu", weights_only=False)
        if done["input_hash"] != input_hash:
            raise ValueError("Existing final output has a different input contract")
        print(json.dumps({"status": "already_complete", "roots": len(roots), "path": str(args.out)}), flush=True)
        return
    _, _, _, _, frame = RP.env_and_render()
    check_roots = {0, len(roots) // 2, len(roots) - 1}
    rows, ledger = [], []
    started = time.monotonic()
    for i, root in enumerate(roots):
        path = chunks / f"root-{i:06d}.pt"
        fields = RP.episode_fields(root["shard"], root["slot"])
        assert fields["episode_id"] == root["episode_id"], (root, fields["episode_id"])
        assert fields["split"] == "dev", (root, fields["split"])
        a, b = root["t"] - 15, root["t"] + 16
        actions = fields["actions_taken"].numpy()
        assert a >= 0 and b <= len(actions)
        assert np.array_equal(actions[a:b], panel["actions"][i].numpy()), "panel action alignment mismatch"
        item = {**root, "frame_start": a, "frame_stop_inclusive": b,
                "shard_recorded_sha256": RP.manifest()["shards"][root["shard"]]["sha256"],
                "actions_prefix_sha256": hashlib.sha256(actions[:b].tobytes()).hexdigest(),
                "stored_window_sha256": hashlib.sha256(fields["observations"][a:b + 1].numpy().tobytes()).hexdigest()}
        row_hash = digest({"input_hash": input_hash, "ledger": item})
        if path.exists():
            row = torch.load(path, map_location="cpu", weights_only=False)
            if row["row_hash"] != row_hash:
                raise ValueError(f"Resume contract differs for row {i}")
        else:
            state = RP.advance_to(root["shard"], root["slot"], a)
            _, keys = RP._slot_keys(root["shard"], root["slot"])
            vv, native = window(state, keys[a:b], jnp.asarray(actions[a:b], dtype=jnp.int32))
            v, native = np.asarray(vv), np.asarray(native)
            assert v.shape == (32, 1534)
            err = native_checks(v, native)
            mobs = v[:, 1071:1512].reshape(32, 63, 7)
            layout = np.stack([mobs[..., 0], mobs[..., 1], mobs[..., 2], mobs[..., 3:].sum(-1)], -1) > 0
            assert np.array_equal(layout, panel["layouts"][i].numpy() > 0), f"Panel mob mismatch row {i}"
            checks = []
            if i in check_roots:
                for pos in (0, 15, 31):
                    s = state if pos == 0 else RP.advance_to(root["shard"], root["slot"], a + pos)
                    rendered = np.asarray(frame(s)).astype(np.int32)
                    stored = fields["observations"][a + pos].numpy().astype(np.int32)
                    diff = int(np.abs(rendered - stored).max())
                    assert diff == 0, f"Pixel replay mismatch {root} frame {a + pos}: {diff}"
                    assert np.array_equal(np.asarray(visible(s)), v[pos]), "scan/state agreement mismatch"
                    checks.append({"frame": a + pos, "max_pixel_difference": diff})
            row = {"row_hash": row_hash, "input_hash": input_hash, "ledger": item,
                   "visible": torch.from_numpy(v.copy()), "native_max_counter_error": err,
                   "mob_mismatch_count": 0, "pixel_checks": checks}
            atomic_save(row, path)
        rows.append(row["visible"])
        ledger.append(row["ledger"])
        if (i + 1) % 10 == 0 or i + 1 == len(roots):
            print(json.dumps({"status": "running", "completed_roots": i + 1, "total_roots": len(roots),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    all_rows = [torch.load(chunks / f"root-{i:06d}.pt", map_location="cpu", weights_only=False)
                for i in range(len(roots))]
    result = {"visible": torch.stack(rows), "roots": roots, "ledger": ledger, "input_hash": input_hash,
              "contract": contract, "checks": {"mob_mismatch_count": 0,
              "native_max_counter_error": max(x["native_max_counter_error"] for x in all_rows),
              "pixel_checks": [c for x in all_rows for c in x["pixel_checks"]]},
              "chunk_sha256": [sha256(chunks / f"root-{i:06d}.pt") for i in range(len(roots))]}
    atomic_save(result, args.out)
    print(json.dumps({"status": "complete", "roots": len(roots), "shape": list(result["visible"].shape),
                      "input_hash": input_hash, "checks": result["checks"], "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
