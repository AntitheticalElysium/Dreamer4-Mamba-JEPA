"""P1-F5 step 3 (PLAN.md): oracle gate for the two forecast targets on the change-map gate's pairs and natural states.

TSP forecast: a 3x3 window's pixels of the forecast frame in [-1, 1], averaged over the 49 uniform windows (the F5
schedule's gaze-independent draw). Outcome forecast: reward bin and termination of the forecast transition. G1 at pure
noise and the change target are taken from ../20261111_change_forecast_gate/CHECKS.json for the same frames; the pairs
and natural states are rebuilt and asserted identical. CPU only, no model.

    python -m artifacts.experiments.20261111_p1f5.forecast_gate
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import importlib, json
from pathlib import Path
import numpy as np
import torch
from saccade import data as D, train as TR

P = Path(__file__).resolve().parent
E = P.parent
G = importlib.import_module("artifacts.experiments.20261111_change_forecast_gate.gate")
PREV_PATH = E / "20261111_change_forecast_gate/CHECKS.json"


def windows(frame):
    """[63, 63, 3] uint8 -> [49, 1323]: each uniform window's pixels in [-1, 1] (TSP's faithful target)."""
    y = frame.astype(np.float64) / 127.5 - 1
    return np.stack([y[(r - 1) * 7:(r + 2) * 7, (c - 1) * 7:(c + 2) * 7].ravel() for r, c in G.CENTRES])


def tsp_pair(a, b):
    """Excess squared error of the best tied forecast, mean over the window's numbers, averaged over windows."""
    return float(((windows(a) - windows(b)) ** 2).mean() / 4)


def tsp_keys(frames):
    return float(np.stack([windows(f) for f in frames]).var(0).mean())


def cat_gap(labels):
    """Deterministic outcomes per member/key: the tied forecast's excess log-loss is the entropy of their mixture."""
    _, c = np.unique(np.asarray(labels), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def outcome_gap(reward, terminated, edges):
    """Match G4: categorical reward and Bernoulli termination conditional on the recorded reward."""
    reward = torch.as_tensor(reward, dtype=torch.float32)
    symlog = reward.sign() * reward.abs().log1p()
    bins = len(edges) - 1
    binned = (((symlog - edges[0]) / (edges[-1] - edges[0])) * bins).floor().long().clamp(0, bins - 1)
    _, groups, counts = np.unique(reward.numpy(), return_inverse=True, return_counts=True)
    cond_term = sum(n / len(reward) * cat_gap(np.asarray(terminated)[groups == i]) for i, n in enumerate(counts))
    return cat_gap(binned.numpy()), float(cond_term)


def main():
    jax, jnp, step, render, roll = G.sim()
    L = G.dev_labels()
    prev = json.loads(PREV_PATH.read_text())
    ck = torch.load(G.CKPT, mmap=True, map_location="cpu", weights_only=False)
    edges = ck["model"]["outcome.edges"]

    banks = {"C1": G.hunger_pairs(L, jax, jnp, roll), "C2": G.zombie_pairs(L, jax, jnp, step, render, roll)[0]}
    rows = {}
    for name, bank in banks.items():
        before = {x["e"]: x for x in prev["rows"][name] if "excluded" not in x}
        assert {sc["e"] for sc in bank if "excluded" not in sc} == set(before), name
        rows[name] = []
        for sc in bank:
            if "excluded" in sc:
                continue
            old = before[sc["e"]]
            for k in ("t0", "key"):
                assert k not in old or old[k] == sc[k], (name, sc["e"], k)
            br, q = sc["branch"], sc["q"]
            a, b = br[0, q + 1], br[1, q + 1]
            reward, termination = outcome_gap(sc["reward"][:, q], sc["term"][:, q], edges)
            row = {"e": sc["e"], "change": old["change"], "g1": old["g1"], "tsp": tsp_pair(a, b),
                   "reward": reward, "termination": termination,
                   "rewards": sc["reward"][:, q].tolist(), "controls_tsp": [tsp_pair(br[0, c + 1], br[1, c + 1]) for c in sc["controls"]],
                   "foil_tsp": {}, "foil_g1": {k: v["g1"] for k, v in old["foil"].items()}}
            for amp in (1, 3, 6):
                row["foil_tsp"][str(amp)] = tsp_pair(b, np.minimum(b.astype(np.int16) + amp, 255).astype(np.uint8))
            rows[name].append(row)
            print(name, sc["e"], round(row["tsp"], 6), row["reward"], {k: round(v, 6) for k, v in row["foil_tsp"].items()}, flush=True)

    # Natural floors: the same states and keys as the change-map gate (same seeds and draw order, asserted).
    rng = np.random.default_rng(7)
    eps = rng.choice(np.flatnonzero(L["ep_T"] >= 20), 120, replace=False)
    noise, j = [], 0
    for i in eps:
        fields = G.recorded(L, i)
        frames, acts = fields["observations"].numpy(), fields["actions_taken"].numpy()
        for t in sorted(rng.choice(np.arange(1, len(acts)), 2, replace=False)):
            old = prev["noise"][j]; j += 1
            assert (old["e"], old["t"]) == (int(i), int(t))
            st = D.RP.advance_to(int(L["ep_shard"][i]), int(L["ep_slot"][i]), int(t))
            kk = jax.random.split(jax.random.PRNGKey(7 * 100003 + int(i) * 1009 + int(t)), 8)
            outs = [step(k, st, jnp.asarray(int(acts[t]), jnp.int32)) for k in kk]
            nxt = np.stack([np.asarray(render(o[1])) for o in outs])
            rng.integers(len(G.CENTRES))                                 # the fixation draw the earlier gate consumed
            reward, termination = outcome_gap([float(o[2]) for o in outs], [bool(o[3]) for o in outs], edges)
            noise.append({**old, "tsp": tsp_keys(nxt), "reward": reward, "termination": termination})

    floors = {k: float(np.mean([x[k] for x in noise])) for k in ("change", "g1", "tsp", "reward", "termination")}
    dark = np.array([x["night"] for x in noise])
    floors_day = {k: float(np.mean([x[k] for x, d in zip(noise, dark) if not d])) for k in ("g1", "tsp")}
    summary = {"floors": floors, "floors_day": floors_day, "states": len(noise)}
    boot = np.random.default_rng(0)
    for name, rs in rows.items():
        st, sg = np.array([r["tsp"] for r in rs]), np.array([r["g1"] for r in rs])
        nt, ng = np.array([x["tsp"] for x in noise]), np.array([x["g1"] for x in noise])
        R = (st / nt.mean()) / (sg / ng.mean())
        assert len(noise) == 2 * len(eps)
        # The two natural states in an episode are a single resampling unit.
        bs = [np.median((st[i] / nt[k].mean()) / (sg[i] / ng[k].mean()))
              for i, k in ((boot.integers(len(rs), size=len(rs)),
                            (2 * boot.integers(len(eps), size=(len(eps), 1)) + np.arange(2)).ravel())
                           for _ in range(2000))]
        summary[name] = {
            "pairs": len(rs),
            "event_over_floor": {k: float(np.median([r[k] for r in rs]) / floors[k]) if floors[k] > 0 else None
                                 for k in ("change", "g1", "tsp", "reward", "termination")},
            "cost_median": {k: float(np.median([r[k] for r in rs])) for k in ("change", "g1", "tsp", "reward", "termination")},
            "R_tsp_vs_g1": float(np.median(R)), "R_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "R_tsp_vs_g1_day_floor": float(np.median((st / floors_day["tsp"]) / (sg / floors_day["g1"]))),
            "pairs_with_reward_gap": int(sum(r["reward"] > 0 for r in rs)),
            "controls_tsp_max": float(max(max(r["controls_tsp"]) for r in rs)),
            "foil_over_consequence_max_tsp": {a: float(max(r["foil_tsp"][a] / r["tsp"] for r in rs)) for a in ("1", "3", "6")},
            "scenes_foil_exceeds_tsp": {a: int(sum(r["foil_tsp"][a] >= r["tsp"] for r in rs)) for a in ("1", "3", "6")},
        }
    report = {"plan": "PLAN.md (step 3)", "previous": str(PREV_PATH.relative_to(D.ROOT)), "prev_sha256": TR.file_sha256(PREV_PATH),
              "summary": summary, "rows": rows, "noise": noise}
    (P / "FORECAST_GATE.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
