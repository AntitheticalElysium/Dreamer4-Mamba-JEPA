"""S0 step 4 (2026-10-27): a simulator-oracle sampler for the multi-step (S2) gates, their power, and references.

Dev windows of 16 consecutive EXOGENOUS actions (CURRICULUM: only these support distributional scoring; all come from
eps = 1.0 episodes). The replayed start state is stepped 16 times with the recorded actions and FRESH random keys:
true samples from the environment's conditional distribution GIVEN THE FULL STATE (sharper than any belief can be, so
the oracle's scores are a floor, not a target). 16 samples per window: set A (8) plays a perfect model, set B (8) is
the null comparison. Against the recorded future y, per window:
- joint-trajectory energy score (G.summary_vector, components scaled by the recorded futures' std, absorbing terminal convention);
- persistence of entities over 1-4 steps and mean visible entity count, up to termination;
- references: the oracle (B) and a MARGINAL model (8 recorded futures of windows from other episodes, history ignored).
Sanity: the recorded future must be exchangeable with the oracle samples (randomised PIT of three scalars uniform;
recorded persistence and counts equal to the oracle's).
Power: entity-level corruptions of A (world coordinates, from the simulator records) must be rejected:
flicker (each entity missing at single steps, p = 0.2), teleport (an entity jumps (+-2, +-2) and stays displaced,
p = 0.1 per step), phantom drift (a cow appears every 4 steps and stays), vanishing drift (an entity disappears for
good, p = 0.05 per step), frozen time (positions, HUD and entities fixed; reward and termination kept), delayed
consequences (HUD 2 steps late). A gate rejects when the difference to B exceeds 3 episode-bootstrap SE (one-sided
for the energy score, two-sided otherwise).
Usage (project venv): python -m saccade.eval.oracle LABELS_NPZ OUT_JSON [N_WINDOWS]
"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
import replay as RP
from saccade.eval import gates as G

H, M = 16, 16
CORRUPTIONS = ("flicker", "teleport", "phantom_drift", "vanishing_drift", "frozen_time", "delayed_consequences")


def sampler():
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position,
                     "body": jnp.stack([s.player_health, s.player_food, s.player_drink, s.player_energy]),
                     **{f"{m}_p": getattr(s, m).position for m in G.KINDS}, **{f"{m}_m": getattr(s, m).mask for m in G.KINDS}}

    def one(state, actions, keys):
        def body(s, xs):
            k, a = xs
            _, nxt, r, d, _ = env.step(k, s, a, params)
            return nxt, (rec(nxt), r, d)
        _, (seq, r, d) = jax.lax.scan(body, state, (keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), rec(state), seq), r, d
    return jax.jit(jax.vmap(one, in_axes=(None, None, 0)))


def to_obs(R):
    """Simulator record of one trajectory -> what a reader sees: (kind grid, body, pos)."""
    pos = R["pos"].astype(int); grid = np.zeros((len(pos), 7, 9), np.int8)
    for k, m in enumerate(G.KINDS, start=1):
        rel = R[f"{m}_p"].astype(int) - pos[:, None, :]
        vis = R[f"{m}_m"] & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
        t, j = np.nonzero(vis)
        grid[t, rel[t, j, 0] + 3, rel[t, j, 1] + 4] = k
    return grid, R["body"].astype(int), pos


def corrupt_rec(kind, R, rng):
    Q = {k: v.copy() for k, v in R.items()}; n = len(Q["pos"])
    if kind == "flicker":
        for m in G.KINDS:
            Q[f"{m}_m"][1:] &= rng.random(Q[f"{m}_m"][1:].shape) >= 0.2
    elif kind == "teleport":
        for m in G.KINDS:
            P = Q[f"{m}_p"]
            for t in range(1, n):
                jump = rng.random(P.shape[1]) < 0.1
                P[t:, jump] += rng.choice([-2, 2], (int(jump.sum()), 2))
    elif kind == "phantom_drift":
        add_p, add_m = np.zeros((n, 4, 2), int), np.zeros((n, 4), bool)
        for i, t in enumerate(range(4, n, 4)):
            off = np.array([rng.integers(-2, 3), rng.integers(-3, 4)])
            off[1] += off[1] == 0 and off[0] == 0
            add_p[t:, i] = Q["pos"][t] + off; add_m[t:, i] = True
        Q["cows_p"] = np.concatenate([Q["cows_p"], add_p], 1); Q["cows_m"] = np.concatenate([Q["cows_m"], add_m], 1)
    elif kind == "vanishing_drift":
        for m in G.KINDS:
            for t in range(1, n):
                Q[f"{m}_m"][t:, rng.random(Q[f"{m}_m"].shape[1]) < 0.05] = False
    elif kind == "frozen_time":
        for k in Q:
            Q[k][1:] = Q[k][0]
    elif kind == "delayed_consequences":
        Q["body"][1:] = R["body"][np.maximum(np.arange(1, n) - 2, 0)]
    return Q


def traj(R, r, d):
    g, b, p = to_obs(R)
    return {"grid": g, "body": b, "pos": p, "reward": np.asarray(r, float), "done": np.asarray(d, bool)}


def n_alive(o):
    """States up to and including the terminal one (a dead state keeps being stepped by the simulator)."""
    return int(np.argmax(o["done"])) + 2 if o["done"].any() else len(o["pos"])


def per_traj(o):
    """Persistence (num, den per k) and visible-count (sum, steps) up to termination."""
    n = n_alive(o)
    num, den = G.persistence(o["grid"][:n], o["pos"][:n])
    return num, den, float((o["grid"][1:n] > 0).sum()), float(n - 1)


def paired_diff(na, da, nb, db, ep, rng, n_boot=1000):
    """sum(na)/sum(da) - sum(nb)/sum(db) over windows, with an episode-cluster bootstrap SE."""
    u, inv = np.unique(ep, return_inverse=True)

    def agg(x):
        z = np.zeros((len(u),) + np.shape(x)[1:]); np.add.at(z, inv, x)
        return z
    A, Da, B, Db = agg(na), agg(da), agg(nb), agg(db)
    f = lambda i: A[i].sum(0) / np.maximum(Da[i].sum(0), 1e-12) - B[i].sum(0) / np.maximum(Db[i].sum(0), 1e-12)
    boots = np.stack([f(rng.integers(0, len(u), len(u))) for _ in range(n_boot)])
    return f(np.arange(len(u))), boots.std(0)


def pit(xs, y, rng):
    """Randomised PIT of y among samples xs (uniform on (0, 1) when y is exchangeable with them)."""
    return (np.sum(xs < y) + rng.random() * (np.sum(xs == y) + 1)) / (len(xs) + 1)


def main():
    L = dict(np.load(sys.argv[1])); n_win = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    T = L["ep_T"]; t_off = np.r_[0, np.cumsum(T)]
    exo = L["t_exogenous"]; rng = np.random.default_rng(20261028)
    cands = [(e, t0) for e in range(len(T)) for t0 in range(0, T[e] - H + 1, H) if exo[t_off[e] + t0:t_off[e] + t0 + H].all()]
    picks = [cands[i] for i in np.sort(rng.permutation(len(cands))[:n_win])]
    print(f"{len(cands)} exogenous 16-step windows; using {len(picks)}", flush=True)
    samp = sampler(); eps_list = G.episode_grids(L)
    Y, S, Rs = [], [], []
    for w, (e, t0) in enumerate(picks):
        state = RP.advance_to(int(L["ep_shard"][e]), int(L["ep_slot"][e]), t0)
        acts = jnp.asarray(L["t_action"][t_off[e] + t0:t_off[e] + t0 + H].astype(np.int32))
        keys = jax.random.split(jax.random.PRNGKey(10 ** 6 + w), M * H).reshape(M, H, 2)
        R, r, d = jax.tree.map(np.asarray, samp(state, acts, keys))
        ep = eps_list[e]
        Y.append({k: ep[k][t0:t0 + H + 1] for k in ("grid", "body", "pos")} | {k: ep[k][t0:t0 + H] for k in ("reward", "done")})
        Rs.append([(jax.tree.map(lambda v: v[i], R), r[i], d[i]) for i in range(M)])
        S.append([traj(*x) for x in Rs[-1]])
        assert (Y[-1]["pos"][0] == S[-1][0]["pos"][0]).all() and (Y[-1]["grid"][0] == S[-1][0]["grid"][0]).all()
        if (w + 1) % 200 == 0:
            print(f"window {w + 1}/{len(picks)}", flush=True)
    ep_of = np.array([e for e, _ in picks])
    vec = lambda o: G.summary_vector(o["grid"], o["body"], o["reward"], o["done"], o["pos"])
    scale = np.stack([vec(y) for y in Y]).std(0); scale[scale == 0] = 1.0           # data std (recorded futures)
    V = lambda o: vec(o) / scale
    Yv = [V(y) for y in Y]
    # references and sanity
    es = {"oracle_B": np.array([G.energy_score(np.stack([V(o) for o in s[8:]]), yv) for s, yv in zip(S, Yv)]),
          "oracle_A": np.array([G.energy_score(np.stack([V(o) for o in s[:8]]), yv) for s, yv in zip(S, Yv)])}
    marg = []
    for w in range(len(Y)):
        others = np.flatnonzero(ep_of != ep_of[w]); pick = rng.choice(others, 8, replace=False)
        marg.append(G.energy_score(np.stack([Yv[i] for i in pick]), Yv[w]))
    es["marginal"] = np.array(marg)
    scal = {"health_drops": lambda o: float((np.diff(o["body"][:n_alive(o), 0]) < 0).sum()),
            "visible_at_end": lambda o: float((o["grid"][-1] > 0).sum()) if not o["done"].any() else -1.0,
            "displacement": lambda o: float(np.abs(o["pos"][n_alive(o) - 1] - o["pos"][0]).sum())}
    out = {"n_windows": len(Y), "n_candidates": len(cands), "n_episodes": int(len(np.unique(ep_of))), "pit": {}}
    for name, f in scal.items():
        u = np.array([pit(np.array([f(o) for o in s]), f(y), rng) for y, s in zip(Y, S)])
        q = np.histogram(u, bins=[0, .25, .5, .75, 1])[0] / len(u)
        out["pit"][name] = {"mean": float(u.mean()), "quartile_shares": q.round(4).tolist()}
    out["energy_score"] = {k: float(v.mean()) for k, v in es.items()}
    pt = {"recorded": [per_traj(y) for y in Y]}
    pt["oracle_B"] = [[per_traj(o) for o in s[8:]] for s in S]
    agg8 = lambda lst: tuple(np.array([np.sum([x[i] for x in w8], 0) for w8 in lst]) for i in range(4))
    B = agg8(pt["oracle_B"]); Rec = tuple(np.array([x[i] for x in pt["recorded"]]) for i in range(4))
    out["persistence"] = {"oracle_B": (B[0].sum(0) / B[1].sum(0)).round(4).tolist(),
                          "recorded": (Rec[0].sum(0) / Rec[1].sum(0)).round(4).tolist()}
    out["visible_count"] = {"oracle_B": float(B[2].sum() / B[3].sum()), "recorded": float(Rec[2].sum() / Rec[3].sum())}
    # recorded vs oracle: the sampler must reproduce the logged dynamics (8 samples vs 1 recording, scaled per window)
    d, se = paired_diff(Rec[0], Rec[1], B[0], B[1], ep_of, rng)
    out["sanity_recorded_vs_oracle"] = {"persistence_diff": d.round(4).tolist(), "persistence_se": se.round(4).tolist()}
    d, se = paired_diff(Rec[2], Rec[3], B[2], B[3], ep_of, rng)
    out["sanity_recorded_vs_oracle"] |= {"count_diff": float(d), "count_se": float(se)}
    d, se = paired_diff(es["oracle_A"], np.ones(len(Y)), es["oracle_B"], np.ones(len(Y)), ep_of, rng)
    out["null_A_vs_B"] = {"es_diff": float(d), "es_se": float(se)}
    A0 = agg8([[per_traj(o) for o in s[:8]] for s in S])
    d, se = paired_diff(A0[0], A0[1], B[0], B[1], ep_of, rng)
    out["null_A_vs_B"] |= {"persistence_diff": d.round(4).tolist(), "persistence_se": se.round(4).tolist()}
    d, se = paired_diff(A0[2], A0[3], B[2], B[3], ep_of, rng)
    out["null_A_vs_B"] |= {"count_diff": float(d), "count_se": float(se)}
    print("pit", out["pit"], "\nES", out["energy_score"], "\nsanity", out["sanity_recorded_vs_oracle"], "\nnull", out["null_A_vs_B"], flush=True)
    out["power"] = {}
    for c in CORRUPTIONS:
        Ts = [[traj(corrupt_rec(c, R, rng), r, dd) for R, r, dd in rs[:8]] for rs in Rs]
        esc = np.array([G.energy_score(np.stack([V(o) for o in t]), yv) for t, yv in zip(Ts, Yv)])
        C = agg8([[per_traj(o) for o in t] for t in Ts])
        de, see = paired_diff(esc, np.ones(len(Y)), es["oracle_B"], np.ones(len(Y)), ep_of, rng)
        dp, sep = paired_diff(C[0], C[1], B[0], B[1], ep_of, rng)
        dc, sec = paired_diff(C[2], C[3], B[2], B[3], ep_of, rng)
        out["power"][c] = {"es_diff": float(de), "es_se": float(see), "es_rejects": bool(de > 3 * see),
                           "persistence_diff": dp.round(4).tolist(), "persistence_se": sep.round(4).tolist(),
                           "persistence_rejects": bool((np.abs(dp) > 3 * sep).any()),
                           "count_diff": float(dc), "count_se": float(sec), "count_rejects": bool(abs(dc) > 3 * sec)}
        print(c, out["power"][c], flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
