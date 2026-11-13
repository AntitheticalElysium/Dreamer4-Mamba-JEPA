"""S2 L1 on the terminal-inclusive DEV bank: does the harness pass a correct world and fail broken ones? (S2_PROPOSAL 3.)
CPU only, no model. saccade/eval/oracle.py's protocol and functions, unchanged, on bank.py's sample:

- The simulator oracle (replayed start state, the window's precommitted actions, fresh keys; 16 futures per start:
  A = a perfect model, B = the reference) with every trajectory absorbing on its OWN termination; the recorded future
  absorbs at the recorded death and is padded there (never at a time truncation: such starts are not admitted).
- Sanity: the recorded future is exchangeable with the oracle's (randomised PIT, now including the death step;
  persistence, counts and the termination rate equal).
- Null: A against B on every gate. Power: every corruption of A must be rejected by at least one gate: the seven of
  S0, plus wrong actions (A rolled under another start's plan, judged against its own), fake consequences (unprovoked health loss), immortal
  (no trajectory ever terminates: what a survival-censored reference would teach) and sudden death (termination
  without cause).
Gates: the one-step event family (G.compare), 1-4 step persistence, visible count, and two termination readings
(share of trajectories terminating in the window; mean absorbing step), each rejecting beyond 3 episode-bootstrap
SE. The energy score is reported beside them, never pass / fail.

    JAX_PLATFORMS=cpu python -m artifacts.experiments.20261114_s2_harness.oracle_l1 OUT_JSON [N_STARTS]
"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np

from saccade.eval import gates as G, oracle as O, s1 as S

HERE = Path(__file__).resolve().parent
H, M = O.H, O.M
RP = O.RP
EXTRA = ("wrong_action", "fake_consequences", "immortal", "sudden_death")


def recorded(ep, t0, T):
    """The recorded future of a start, absorbing at the recorded death: states padded with the terminal state."""
    n = min(H, T - t0)
    pad = lambda x, m: np.concatenate([x, np.repeat(x[-1:], m - len(x), 0)]) if len(x) < m else x
    y = {k: pad(ep[k][t0:t0 + n + 1], H + 1) for k in ("grid", "body", "pos")}
    y["reward"] = np.r_[ep["reward"][t0:t0 + n], np.zeros(H - n)]
    y["done"] = np.r_[ep["done"][t0:t0 + n], np.ones(H - n, bool)]
    assert n == H or ep["done"][t0 + n - 1], "a padded window must end in a recorded death"
    return y


def death_step(o):
    return float(np.argmax(o["done"]) + 1) if o["done"].any() else float(H + 1)


def extra(kind, o, rng):
    q = {k: v.copy() for k, v in o.items()}
    if kind == "fake_consequences":
        for t in range(1, len(q["body"])):
            if rng.random() < 0.1:
                q["body"][t:, 0] = np.maximum(q["body"][t:, 0] - 1, 1)
    elif kind == "immortal":
        q["done"][:] = False
        q["body"][:, 0] = np.maximum(q["body"][:, 0], 1)
    elif kind == "sudden_death":
        if rng.random() < 0.1:
            q["done"][rng.integers(H):] = True
    return q


def main():
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    bank = np.load(HERE / "dev_bank_h16.npz")
    n_win = int(sys.argv[2]) if len(sys.argv) > 2 else len(bank["sample"])
    pick = bank["sample"][np.sort(np.random.default_rng(7).permutation(len(bank["sample"]))[:n_win])]
    rows, plans = bank["rows"][pick], bank["plans"][pick].astype(np.int32)
    other = np.roll(np.arange(len(rows)), len(rows) // 2)                    # another start's plan (wrong_action)
    t_off = np.r_[0, np.cumsum(L["ep_T"])]
    rng = np.random.default_rng(20261114)
    samp, eps_list = O.sampler(), G.episode_grids(L)
    Y, Sx, Rs, W = [], [], [], []
    for w, (e, sh, sl, t0, T, _) in enumerate(rows):
        state = RP.advance_to(int(sh), int(sl), int(t0))
        keys = jax.random.split(jax.random.PRNGKey(2 * 10 ** 6 + w), M * H).reshape(M, H, 2)
        R, r, d = jax.tree.map(np.asarray, samp(state, jnp.asarray(plans[w]), keys))
        Rw, rw, dw = jax.tree.map(np.asarray, samp(state, jnp.asarray(plans[other[w]]), keys[:8]))
        Y.append(recorded(eps_list[e], int(t0), int(T)))
        Rs.append([(jax.tree.map(lambda v: v[i], R), r[i], d[i]) for i in range(M)])
        Sx.append([O.traj(*x) for x in Rs[-1]])
        W.append([O.traj(jax.tree.map(lambda v: v[i], Rw), rw[i], dw[i]) for i in range(8)])
        assert (Y[-1]["pos"][0] == Sx[-1][0]["pos"][0]).all() and (Y[-1]["grid"][0] == Sx[-1][0]["grid"][0]).all()
        if (w + 1) % 100 == 0:
            print(f"start {w + 1}/{len(rows)}", flush=True)
    ep_of = rows[:, 0]
    vec = lambda o: G.summary_vector(o["grid"], o["body"], o["reward"], o["done"], o["pos"])
    scale = np.stack([vec(y) for y in Y]).std(0); scale[scale == 0] = 1.0
    V = lambda o: vec(o) / scale
    Yv = [V(y) for y in Y]
    es_of = lambda Ts: np.array([G.energy_score(np.stack([V(o) for o in t]), yv) for t, yv in zip(Ts, Yv)])
    A, B = [s[:8] for s in Sx], [s[8:] for s in Sx]
    agg = lambda Ts: tuple(np.array([np.sum([O.per_traj(o)[i] for o in t], 0) for t in Ts]) for i in range(4))
    term = lambda Ts: (np.array([sum(o["done"].any() for o in t) for t in Ts], float),
                       np.array([sum(death_step(o) for o in t) for t in Ts], float), np.array([len(t) for t in Ts], float))
    since0 = []
    for e, _, _, t0, _, _ in rows:
        drops = np.flatnonzero(np.diff(eps_list[e]["body"][:t0 + 1, 0]) < 0)
        since0.append(int(min(t0 - drops[-1], 10)) if len(drops) else 10)

    def events(Ts, acts):
        parts = [G.traj_pairs(o["grid"][:n], o["body"][:n], o["pos"][:n], acts[w], o["light"][:n], since0[w], ep_of[w])
                 for w, ts in enumerate(Ts) for o in ts for n in [O.n_alive(o)]]
        return G.all_estimates({k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    ev_B, pB, tB, esB = events(B, plans), agg(B), term(B), es_of(B)
    ones = np.ones(len(Y))

    def gates(Ts, acts=plans):
        p, t, es = agg(Ts), term(Ts), es_of(Ts)
        out = {}
        for name, (na, da, nb, db) in {"persistence": (p[0], p[1], pB[0], pB[1]), "count": (p[2], p[3], pB[2], pB[3]),
                                       "terminated": (t[0], t[2], tB[0], tB[2]), "death_step": (t[1], t[2], tB[1], tB[2]),
                                       "energy_score": (es, ones, esB, ones)}.items():
            d, se = O.paired_diff(na, da, nb, db, ep_of, rng)
            rej = bool(np.any(d > 3 * se + 1e-9)) if name == "energy_score" else bool(np.any(np.abs(d) > 3 * se + 1e-9))
            out[name] = {"diff": np.round(d, 5).tolist(), "se": np.round(se, 5).tolist(), "rejects": rej}
        rej, unm = G.compare(ev_B, events(Ts, acts))
        out["event_gates_rejecting"], out["event_gates_unmeasured"] = rej, unm
        out["rejected_by"] = [k for k in ("persistence", "count", "terminated", "death_step") if out[k]["rejects"]] + \
                             (["events"] if rej else [])
        return out

    res = {"starts": len(Y), "episodes": int(len(np.unique(ep_of))), "early_terminal_starts": int((rows[:, 3] + H > rows[:, 4]).sum()),
           "recorded_terminated": float(np.mean([y["done"].any() for y in Y])), "oracle_terminated": float(tB[0].sum() / tB[2].sum()),
           "pit": {}}
    scal = {"health_drops": lambda o: float((np.diff(o["body"][:O.n_alive(o), 0]) < 0).sum()),
            "displacement": lambda o: float(np.abs(o["pos"][O.n_alive(o) - 1] - o["pos"][0]).sum()), "death_step": death_step}
    for name, f in scal.items():
        u = np.array([O.pit(np.array([f(o) for o in s]), f(y), rng) for y, s in zip(Y, Sx)])
        res["pit"][name] = {"mean": float(u.mean()), "quartile_shares": (np.histogram(u, bins=[0, .25, .5, .75, 1])[0] / len(u)).round(4).tolist()}
    rec = [[y] for y in Y]
    pr, tr = agg(rec), term(rec)
    sanity = {}
    for name, (na, da, nb, db) in {"persistence": (pr[0], pr[1], pB[0], pB[1]), "count": (pr[2], pr[3], pB[2], pB[3]),
                                   "terminated": (tr[0], tr[2], tB[0], tB[2]), "death_step": (tr[1], tr[2], tB[1], tB[2])}.items():
        d, se = O.paired_diff(na, da, nb, db, ep_of, rng)
        sanity[name] = {"diff": np.round(d, 5).tolist(), "se": np.round(se, 5).tolist(), "beyond_3se": bool(np.any(np.abs(d) > 3 * se + 1e-9))}
    res["sanity_recorded_vs_oracle"] = sanity
    res["energy_score"] = {"oracle_A": float(es_of(A).mean()), "oracle_B": float(esB.mean())}
    res["null_A_vs_B"] = gates(A)
    print(json.dumps({k: res[k] for k in ("starts", "recorded_terminated", "oracle_terminated", "pit", "sanity_recorded_vs_oracle")}), flush=True)
    print("null", json.dumps(res["null_A_vs_B"]), flush=True)
    res["power"] = {}
    for c in O.CORRUPTIONS + EXTRA:
        if c == "wrong_action":                                              # judged against the COMMANDED plan
            Ts, acts = W, plans
        elif c in EXTRA:
            Ts, acts = [[extra(c, o, rng) for o in t] for t in A], plans
        else:
            Ts, acts = [[O.traj(O.corrupt_rec(c, R, rng), r, dd) for R, r, dd in rs[:8]] for rs in Rs], plans
        res["power"][c] = gates(Ts, acts)
        print(c, res["power"][c]["rejected_by"], len(res["power"][c]["event_gates_rejecting"]), flush=True)
    res["passes"] = {"null_not_rejected": not res["null_A_vs_B"]["rejected_by"],
                     "recorded_exchangeable": not any(v["beyond_3se"] for v in sanity.values()),
                     "every_corruption_rejected": all(bool(v["rejected_by"]) for v in res["power"].values())}
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["passes"]), flush=True)


if __name__ == "__main__":
    main()
