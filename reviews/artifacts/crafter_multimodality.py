"""Phase C premise check: how stochastic are Crafter transitions, as observed?

Method: run a random policy; at probe states, deepcopy the env into N branches,
reseed each branch's world RNG (all creature behaviour, spawning, and night render
noise flow through the single `world.random` RandomState), step the SAME action,
and compare successors.

Primary metric (render-noise-free): divergence of the *semantic state inside the
local 9x7 view* + player pos/health/inventory. Secondary: pixel divergence in the
world-view image region (rows < 49), reported separately for day vs night because
night rendering adds RNG noise (crafter engine.py:209).

Pre-registered reading: if the fraction of transitions with any in-view state
divergence is small and the diverging-cell count is low, Crafter transitions are
near-deterministic as observed, and a multimodal successor mixture (Phase C)
cannot pay off on this environment regardless of its mechanics.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

N_BRANCHES = 8
N_PROBES = 150
PROBE_EVERY = 5
OUT = Path(__file__).with_name("crafter_multimodality_results.json")


def local_view_crop(semantic: np.ndarray, pos: np.ndarray):
    """9x7 world tiles shown in the observation: player_pos + ([-4..4], [-3..3])."""
    x0, x1 = pos[0] - 4, pos[0] + 5
    y0, y1 = pos[1] - 3, pos[1] + 4
    padded = np.full((9, 7), -1, dtype=np.int64)
    sx0, sx1 = max(0, x0), min(semantic.shape[0], x1)
    sy0, sy1 = max(0, y0), min(semantic.shape[1], y1)
    padded[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = semantic[sx0:sx1, sy0:sy1]
    return padded


def state_signature(obs, info):
    view = local_view_crop(np.asarray(info["semantic"]), np.asarray(info["player_pos"]))
    inv = tuple(sorted(info["inventory"].items()))
    return view, inv, obs


def main():
    import crafter

    env = crafter.Env(seed=3, length=10_000)
    rng = np.random.default_rng(3)
    obs = env.reset()
    obs, _, done, info = env.step(0)

    records = []
    steps_since_probe = 0
    while len(records) < N_PROBES:
        if done:
            obs = env.reset()
            obs, _, done, info = env.step(0)
            continue
        steps_since_probe += 1
        if steps_since_probe >= PROBE_EVERY:
            steps_since_probe = 0
            daylight = float(env._world.daylight)
            for action_kind, action in (("policy", int(rng.integers(env.action_space.n))), ("noop", 0)):
                views, invs, pixels = [], [], []
                for branch in range(N_BRANCHES):
                    fork = copy.deepcopy(env)
                    # In place: creatures hold `self.random = world.random` by
                    # reference (crafter objects.py:12); replacing the attribute
                    # would leave every creature on the old copied RNG.
                    fork._world.random.seed(50_000 + 97 * len(records) + branch)
                    b_obs, _, b_done, b_info = fork.step(action)
                    view, inv, img = state_signature(b_obs, b_info)
                    views.append(view); invs.append(inv); pixels.append(img)
                view_diverged = [not np.array_equal(views[0], v) for v in views[1:]]
                inv_diverged = [invs[0] != v for v in invs[1:]]
                diverging_cells = max(
                    (int((views[0] != v).sum()) for v in views[1:]), default=0
                )
                world_region = [p[:49, :, :] for p in pixels]
                pixel_frac = max(
                    (float((world_region[0] != w).any(-1).mean()) for w in world_region[1:]),
                    default=0.0,
                )
                records.append({
                    "action_kind": action_kind,
                    "daylight": daylight,
                    "night": daylight < 0.5,
                    "any_view_divergence": bool(any(view_diverged)),
                    "n_diverged_branches": int(sum(view_diverged)),
                    "max_diverging_view_cells_of_63": diverging_cells,
                    "any_inventory_divergence": bool(any(inv_diverged)),
                    "max_pixel_divergence_frac": pixel_frac,
                })
        action = int(rng.integers(env.action_space.n))
        obs, _, done, info = env.step(action)

    def agg(rows):
        n = len(rows)
        return {
            "n": n,
            "frac_any_view_divergence": sum(r["any_view_divergence"] for r in rows) / n,
            "frac_any_inventory_divergence": sum(r["any_inventory_divergence"] for r in rows) / n,
            "mean_diverging_cells_when_diverged": (
                float(np.mean([r["max_diverging_view_cells_of_63"] for r in rows if r["any_view_divergence"]]))
                if any(r["any_view_divergence"] for r in rows) else 0.0
            ),
            "mean_pixel_divergence_day_only": (
                float(np.mean([r["max_pixel_divergence_frac"] for r in rows if not r["night"]]))
                if any(not r["night"] for r in rows) else None
            ),
        }

    report = {
        "protocol": {
            "seed": 3, "branches": N_BRANCHES, "probes": len(records) // 2,
            "note": "divergence over reseeded world RNG, same action, deepcopied state",
        },
        "all": agg(records),
        "policy_actions": agg([r for r in records if r["action_kind"] == "policy"]),
        "noop_actions": agg([r for r in records if r["action_kind"] == "noop"]),
        "day": agg([r for r in records if not r["night"]]),
        "night": agg([r for r in records if r["night"]]),
    }
    print(json.dumps(report, indent=2))
    OUT.write_text(json.dumps({"report": report, "records": records}, indent=2))


if __name__ == "__main__":
    main()
