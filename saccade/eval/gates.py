"""S0 step 2 (2026-10-27): observable-level event statistics, dev references, and the gates' power (TEST_SPEC v1.1).

Everything is computed from what a percept reader could output: a 7x9 grid of entity kinds around the player, HUD values
(health, food, drink, energy), the camera shift between frames, action, reward and termination. Simulator labels supply
these for real trajectories; later, a frozen reader supplies them for generated ones, and the same functions run.

Gates (each a conditional event probability, episode-bootstrap standard errors):
  appear_interior / appear_edge / vanish_in_view per kind and light (entity bookkeeping: arrivals, phantoms, losses);
  health drop given an adjacent zombie and time since the last drop (consequence timing, hidden cooldown);
  health drop with no zombie adjacent; arrow appearance given a skeleton at Manhattan 4-5 (coupled events);
  camera moved given a movement action, and direction agreement (self-motion);
  mean visible count per kind.
Power check: references from dev half A; half B (real) must pass; five corruptions of B must be rejected:
  shuffled histories, mismatched actions, duplicated entities, delayed consequences (+2 steps), frozen time.
A gate rejects when |B - A| > 3 joint standard errors. Plus a classifier two-sample test (LightGBM) on matched
(current grid, action, shift, next grid) tuples: A vs B (should be ~0.5) and A vs each corruption.
Usage (scratch venv with numpy + lightgbm): python -m saccade.eval.gates LABELS_NPZ OUT_JSON
"""
import json, sys
import numpy as np

KINDS = ("zombies", "cows", "skeletons", "arrows")
MOVE = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}           # LEFT, RIGHT, UP, DOWN as (row, col) deltas


def build_pairs(L):
    """Labels -> per-transition observable pairs (vectorised over all episodes)."""
    T = L["ep_T"]; s_off = np.r_[0, np.cumsum(T + 1)]; t_off = np.r_[0, np.cumsum(T)]
    pos = L["s_pos"].astype(int)
    grid = np.zeros((len(pos), 7, 9), np.int8)
    for k, m in enumerate(KINDS, start=1):
        P = L[f"s_{m}_p"].astype(int); M = L[f"s_{m}_m"]
        rel = P - pos[:, None, :]
        vis = M & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
        r, j = np.nonzero(vis)
        grid[r, rel[r, j, 0] + 3, rel[r, j, 1] + 4] = k
    i0 = np.concatenate([np.arange(s_off[e], s_off[e] + T[e]) for e in range(len(T))])   # state index of s_t
    ep = np.repeat(np.arange(len(T)), T)
    body = L["s_body"].astype(int)
    since = np.zeros(len(i0), int)                                # steps since the last health drop (capped at 10)
    drop = body[i0 + 1, 0] < body[i0, 0]
    for e in range(len(T)):
        c = 10
        for t in range(t_off[e], t_off[e + 1]):
            since[t] = c
            c = 1 if drop[t] else min(c + 1, 10)
    return {"g0": grid[i0], "g1": grid[i0 + 1], "d": pos[i0 + 1] - pos[i0], "h0": body[i0, 0], "h1": body[i0 + 1, 0],
            "night": L["s_light"][i0] < 0.5, "a": L["t_action"].astype(int), "since": since, "ep": ep}


def _dilate(m):
    p = np.pad(m, ((0, 0), (1, 1), (1, 1)))
    return np.max(np.stack([p[:, 1 + dr:8 + dr, 1 + dc:10 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)]), 0)


def _shift(m, d, inner):
    """out[n, r, c] = m[n, r + d0, c + d1] where in range (and at least `inner` cells inside), else False."""
    out = np.zeros_like(m)
    rr, cc = np.meshgrid(np.arange(7), np.arange(9), indexing="ij")
    for d0 in (-1, 0, 1):
        for d1 in (-1, 0, 1):
            sel = (d[:, 0] == d0) & (d[:, 1] == d1)
            if not sel.any():
                continue
            r2, c2 = rr + d0, cc + d1
            ok = (r2 >= inner) & (r2 < 7 - inner) & (c2 >= inner) & (c2 < 9 - inner)
            src = m[sel][:, np.clip(r2, 0, 6), np.clip(c2, 0, 8)]
            out[sel] = src & ok
    return out


def events(P):
    """Per-transition observable events."""
    g0, g1, d = P["g0"], P["g1"], P["d"]
    E = {}
    edge = np.zeros((7, 9), bool); edge[[0, 6], :] = True; edge[:, [0, 8]] = True
    for k, m in enumerate(KINDS, start=1):
        m0, m1 = g0 == k, g1 == k
        explained = _shift(_dilate(m0), d, 0)                     # t+1 cell has a kind-k entity at t within 1 cell
        new = m1 & ~explained
        E[f"appear_edge_{m}"] = (new & edge).any((1, 2))
        E[f"appear_interior_{m}"] = (new & ~edge).any((1, 2))
        kept = _shift(_dilate(m1), -d, 0)                         # t cell has a kind-k entity at t+1 within 1 cell
        inview = _shift(np.ones_like(m0), -d, 1)                  # its neighbourhood is still inside the t+1 view
        E[f"vanish_{m}"] = (m0 & ~kept & inview).any((1, 2))
        E[f"count_{m}"] = m1.sum((1, 2))
    adj = np.zeros(len(g0), bool)
    for r, c in ((2, 4), (4, 4), (3, 3), (3, 5)):
        adj |= g0[:, r, c] == 1
    E["zombie_adj"] = adj
    E["drop"] = P["h1"] < P["h0"]
    rr, cc = np.meshgrid(np.arange(7) - 3, np.arange(9) - 4, indexing="ij"); man = np.abs(rr) + np.abs(cc)
    E["skel_range"] = ((g0 == 3) & ((man == 4) | (man == 5))).any((1, 2))
    E["arrow_new"] = E["appear_interior_arrows"] | E["appear_edge_arrows"]
    mv = np.isin(P["a"], list(MOVE))
    E["move_action"] = mv
    E["moved"] = np.abs(d).sum(1) > 0
    want = np.array([MOVE.get(a, (9, 9)) for a in P["a"]])
    E["dir_ok"] = (d == want).all(1)
    return E


def stats(P, E):
    """Named conditional probabilities: name -> (numerator mask, denominator mask) or a mean value array."""
    n, night = E["drop"], P["night"]
    S = {}
    for m in KINDS:
        for lab, lm in (("day", ~night), ("night", night)):
            for ev in ("appear_edge", "appear_interior", "vanish"):
                S[f"{ev}_{m}_{lab}"] = (E[f"{ev}_{m}"], lm)
            S[f"count_{m}_{lab}"] = (E[f"count_{m}"].astype(float), lm)
    for b in (1, 2, 3, 4, 5, 6):
        sel = P["since"] == b if b < 6 else P["since"] >= 6
        S[f"drop_given_adj_since{b}"] = (n, E["zombie_adj"] & sel)
    S["drop_given_no_adj"] = (n, ~E["zombie_adj"])
    S["arrow_given_skel_range"] = (E["arrow_new"], E["skel_range"])
    S["arrow_given_no_skel_range"] = (E["arrow_new"], ~E["skel_range"])
    S["moved_given_move"] = (E["moved"], E["move_action"])
    S["dir_ok_given_moved_move"] = (E["dir_ok"], E["move_action"] & E["moved"])
    return S


def estimate(val, cond, ep, n_boot=200, rng=None):
    """Ratio estimate sum(val[cond]) / sum(cond) with an episode-bootstrap standard error."""
    v = val.astype(float) * cond; c = cond.astype(float)
    eps_ids, inv = np.unique(ep, return_inverse=True)
    sv = np.bincount(inv, v, len(eps_ids)); sc = np.bincount(inv, c, len(eps_ids))
    if sc.sum() == 0:
        return None
    est = sv.sum() / sc.sum()
    rng = rng or np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(eps_ids), len(eps_ids))
        if sc[idx].sum() > 0:
            boots.append(sv[idx].sum() / sc[idx].sum())
    return {"est": float(est), "se": float(np.std(boots)), "n_cond": int(sc.sum())}


def all_estimates(P):
    E = events(P); S = stats(P, E)
    return {k: estimate(v, c, P["ep"]) for k, (v, c) in S.items()}


def subset(P, mask):
    return {k: v[mask] for k, v in P.items()}


def corrupt(P, kind, rng):
    Q = {k: v.copy() for k, v in P.items()}
    if kind == "shuffled_history":
        perm = rng.permutation(len(Q["g1"]))
        Q["g1"], Q["h1"], Q["d"] = Q["g1"][perm], Q["h1"][perm], Q["d"][perm]
    elif kind == "mismatched_actions":
        Q["a"] = Q["a"][rng.permutation(len(Q["a"]))]
    elif kind == "duplicated_entities":
        g = Q["g1"]; n, r, c = np.nonzero(g > 0)
        pick = rng.random(len(n)) < 0.3
        dr, dc = rng.integers(-1, 2, len(n)), rng.integers(-1, 2, len(n))
        r2, c2 = np.clip(r + dr, 0, 6), np.clip(c + dc, 0, 8)
        ok = pick & (g[n, r2, c2] == 0) & ~((r2 == 3) & (c2 == 4))
        g[n[ok], r2[ok], c2[ok]] = g[n[ok], r[ok], c[ok]]
    elif kind == "delayed_consequences":
        h0, h1, ep = Q["h0"], Q["h1"], Q["ep"]
        same = np.r_[ep[2:] == ep[:-2], [False, False]]
        dh = h1 - h0
        new = np.zeros_like(dh); new[2:] = np.where(same[:-2], dh[:-2], 0)
        Q["h1"] = h0 + new
    elif kind == "frozen_time":
        Q["g1"], Q["h1"], Q["d"] = Q["g0"].copy(), Q["h0"].copy(), np.zeros_like(Q["d"])
    return Q


def c2st(PA, PB, rng):
    import lightgbm as lgb

    def feats(P):
        oh = lambda g: np.stack([(g == k) for k in range(5)], -1).reshape(len(g), -1)
        a = np.eye(17)[P["a"]]
        return np.concatenate([oh(P["g0"]), oh(P["g1"]), a, P["d"], P["h0"][:, None], P["h1"][:, None]], 1).astype(np.float32)
    n = min(len(PA["a"]), len(PB["a"]), 60000)
    ia, ib = rng.choice(len(PA["a"]), n, replace=False), rng.choice(len(PB["a"]), n, replace=False)
    X = np.concatenate([feats(subset(PA, np.isin(np.arange(len(PA["a"])), ia))), feats(subset(PB, np.isin(np.arange(len(PB["a"])), ib)))])
    y = np.r_[np.zeros(n), np.ones(n)]
    epid = np.r_[PA["ep"][np.sort(ia)], PB["ep"][np.sort(ib)] + 10 ** 6]
    u = np.unique(epid); test_eps = rng.choice(u, len(u) // 3, replace=False); te = np.isin(epid, test_eps)
    m = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 12, "seed": 0, "num_leaves": 31},
                  lgb.Dataset(X[~te], y[~te]), 200)
    acc = float(((m.predict(X[te]) > 0.5) == y[te]).mean())
    return {"accuracy": acc, "n_test": int(te.sum())}


# ---------------------------------------------------------------- multi-step and revisit helpers (S0 step 3)
def episode_grids(L):
    """Per-episode observable sequences: kind grid [T+1, 7, 9], terrain view [T+1, 7, 9], pos, body, reward, done."""
    T = L["ep_T"]; s_off = np.r_[0, np.cumsum(T + 1)]; t_off = np.r_[0, np.cumsum(T)]
    pos = L["s_pos"].astype(int); grid = np.zeros((len(pos), 7, 9), np.int8)
    for k, m in enumerate(KINDS, start=1):
        rel = L[f"s_{m}_p"].astype(int) - pos[:, None, :]; M = L[f"s_{m}_m"]
        vis = M & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
        r, j = np.nonzero(vis)
        grid[r, rel[r, j, 0] + 3, rel[r, j, 1] + 4] = k
    out = []
    for e in range(len(T)):
        a, b = s_off[e], s_off[e + 1]; c, d = t_off[e], t_off[e + 1]
        out.append({"grid": grid[a:b], "view": L["s_view"][a:b], "pos": pos[a:b], "body": L["s_body"][a:b].astype(int),
                    "reward": L["t_reward"][c:d], "done": L["t_terminated"][c:d], "action": L["t_action"][c:d].astype(int)})
    return out


def revisit_events(ep_list, drift=0.0, rng=None, marg=None):
    """Cells entering view that were seen before: (episode, gap, terrain agrees with last sighting). Terrain is only
    recorded and compared where no entity covers the cell (a reader cannot see under an entity). With drift > 0, the
    terrain of each revisited cell is redrawn from the terrain marginal `marg` with that probability (a plausible
    forgetting corruption for the power check; the redraw can coincide with the truth)."""
    rows = []
    rr, cc = np.meshgrid(np.arange(7) - 3, np.arange(9) - 4, indexing="ij")
    for e, ep in enumerate(ep_list):
        last_t, last_v = {}, {}
        prev = set()
        for t in range(len(ep["pos"])):
            wr = ep["pos"][t][0] + rr; wc = ep["pos"][t][1] + cc
            cur = set()
            for i in range(7):
                for j in range(9):
                    w = (int(wr[i, j]), int(wc[i, j])); cur.add(w)
                    free = ep["grid"][t, i, j] == 0
                    v = int(ep["view"][t, i, j])
                    if t > 0 and w not in prev and w in last_t and free and w in last_v:
                        if drift > 0 and rng.random() < drift:
                            v = int(rng.choice(len(marg), p=marg))
                        rows.append((e, t - last_t[w], v == last_v[w]))
                    last_t[w] = t
                    if free:
                        last_v[w] = v
            prev = cur
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def revisit_estimates(R):
    out = {}
    for lo, hi in ((1, 16), (17, 64), (65, 256), (257, 10 ** 9)):
        m = (R[:, 1] >= lo) & (R[:, 1] <= hi)
        out[f"revisit_agree_gap{lo}_{hi if hi < 10 ** 9 else 'inf'}"] = estimate(R[:, 2].astype(bool), m, R[:, 0])
    return out


def summary_vector(grid, body, reward, done, pos):
    """One observable trajectory -> one vector: per step, per kind (count, mean row, mean col), body (4), reward,
    termination, camera shift (2). Terminal convention: after termination, an absorbing step (zeros, termination 1)."""
    H = len(reward); feats = []
    rr, cc = np.meshgrid(np.arange(7) - 3, np.arange(9) - 4, indexing="ij")
    ended = False
    for t in range(H):
        if ended:
            f = np.zeros(20); f[17] = 1.0; feats.append(f); continue
        g = grid[t + 1]; f = []
        for k in range(1, 5):
            m = g == k; n = m.sum()
            f += [n, rr[m].mean() if n else 0.0, cc[m].mean() if n else 0.0]
        d = pos[t + 1] - pos[t]
        f += list(body[t + 1]) + [reward[t], float(done[t]), d[0], d[1]]
        feats.append(np.array(f, float))
        ended = bool(done[t])
    return np.concatenate(feats)


def energy_score(X, y):
    """ES_1 with the unbiased pairwise term: mean ||X_i - y|| - 0.5 mean_{i != j} ||X_i - X_j||."""
    m = len(X)
    a = np.linalg.norm(X - y[None], axis=1).mean()
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    return float(a - 0.5 * D.sum() / (m * (m - 1)))


def persistence(grid, pos, kmax=4):
    """Share of entities at t still explained at every step t+1..t+k (same kind within j cells at t+j), among
    entities whose neighbourhood stays in view; per k = 1..kmax. Returns (numerators, denominators)."""
    num, den = np.zeros(kmax), np.zeros(kmax)
    H = len(grid) - 1
    for t in range(H):
        for k in range(1, 5):
            for (i, j) in zip(*np.nonzero(grid[t] == k)):
                w = pos[t] + np.array([i - 3, j - 4]); alive = True
                for s in range(1, kmax + 1):
                    if t + s > H:
                        break
                    rel = w - pos[t + s]
                    if abs(rel[0]) > 3 - 1 or abs(rel[1]) > 4 - 1:
                        break
                    if alive:
                        r0, c0 = rel[0] + 3, rel[1] + 4
                        win = grid[t + s][max(0, r0 - s):r0 + s + 1, max(0, c0 - s):c0 + s + 1]
                        alive = bool((win == k).any())
                    den[s - 1] += 1; num[s - 1] += alive
    return num, den


def main():
    L = dict(np.load(sys.argv[1]))
    P = build_pairs(L)
    out = {"n_transitions": int(len(P["a"])), "reference_full_dev": all_estimates(P)}
    halfA = P["ep"] % 2 == 0
    PA, PB = subset(P, halfA), subset(P, ~halfA)
    refA = all_estimates(PA)
    rng = np.random.default_rng(20261027)
    cases = {"real_half_B": PB}
    for kind in ("shuffled_history", "mismatched_actions", "duplicated_entities", "delayed_consequences", "frozen_time"):
        cases[kind] = corrupt(PB, kind, rng)
    report = {}
    for name, Q in cases.items():
        est = all_estimates(Q); rej = []
        for k, a in refA.items():
            b = est.get(k)
            if a is None or b is None:
                continue
            se = np.hypot(a["se"], b["se"])
            if se > 0 and abs(b["est"] - a["est"]) > 3 * se:
                rej.append(k)
        report[name] = {"rejected_by": rej, "n_rejecting_gates": len(rej), "c2st": c2st(PA, Q, rng)}
        print(name, len(rej), rej[:8], report[name]["c2st"], flush=True)
    out["power"] = report
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
