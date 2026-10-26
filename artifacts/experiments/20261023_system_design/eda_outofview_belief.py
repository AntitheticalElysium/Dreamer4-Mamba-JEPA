"""A3 (2026-10-25): what does the exact belief about an entity that has left the view look like in Craftax-Classic?
User question: how should the out-of-view belief be represented (samples, point + spread, something else)? This
measures the environment's own answer for a perfect observer: the exact posterior over the entity's position given
its last sighting, the player's real trajectory and the fact that it has not been seen since. Simulator state is used
for evaluation only; nothing is trained.

CLAIM: for the gaps that occur, the belief stays compact and single-peaked, so one centre-plus-spread summary keeps
  nearly all of its predictive power.
COMPETING EXPLANATIONS: (a) negative information (the view sweeping over possible locations) carves holes and splits
  the belief; (b) terrain blocking and the chase rule make it multimodal; (c) despawning makes existence, not position,
  the main uncertainty.
CONTROLS: the exact belief by Monte-Carlo simulation of the game's own mob rules (game_logic.update_mobs, re-implemented
  for one mob with the real map, real player positions and the other mobs' real positions for blocking, in update
  order), from the last-seen position; the same simulation without conditioning on non-detection; a moment-matched
  Gaussian with and without truncation by the view; K-sample approximations (K = 4, 16) with a one-cell kernel; the
  log-likelihood of the entity's TRUE position as the common yardstick; strata by kind and gap; day / night.
OUTCOMES: a truncated single Gaussian within ~0.1 nats of the exact belief at the common gaps, with rare split supports,
  means a point + spread meets Craftax's needs; a large loss or frequent split supports means the representation must
  hold several hypotheses. Neither outcome selects a representation for other environments.
Usage (project venv): eda_outofview_belief.py OUT_JSON N_EPISODES N_PARTICLES
"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP
from craftax.craftax_classic.constants import SOLID_BLOCKS, BlockType

KINDS = ("zombies", "cows", "skeletons")
HMAX = 16
SOLID = np.isin(np.arange(64), np.asarray(SOLID_BLOCKS)) | (np.arange(64) == BlockType.LAVA.value)
CARD = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])


def scan_fn():
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position, "light": s.light_level, "map": s.map.astype(jnp.int8),
                     **{f"{m}_p": getattr(s, m).position for m in KINDS}, **{f"{m}_m": getattr(s, m).mask for m in KINDS}}

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, rec(nxt)
        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), rec(s0), seq)
    return jax.jit(run)


def visible(rel):
    return (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)


def blockers(R, s, kind, j):
    """Occupied cells when mob (kind, j) moves during step s -> s+1: mobs updated before it at their new positions, after
    it at their old positions (update order zombies, cows, skeletons; index order within a kind)."""
    occ = []
    order = list(KINDS)
    for m in KINDS:
        for i in range(R[f"{m}_p"].shape[1]):
            if m == kind and i == j:
                continue
            before = order.index(m) < order.index(kind) or (m == kind and i < j)
            t = s + 1 if before else s
            if R[f"{m}_m"][t, i]:
                occ.append(R[f"{m}_p"][t, i])
    return np.array(occ, int).reshape(-1, 2)


def step_particles(x, kind, R, s, rng):
    """Move particles x [N, 2] (world cells) one step with the game's rule for `kind`; returns new x and a despawn mask."""
    player = R["pos"][s + 1].astype(int)                          # move_player runs before update_mobs
    mp = R["map"][s + 1]
    N = len(x)
    if kind == "zombies":
        rand = CARD[rng.integers(0, 4, N)]
        d = player - x; ad = np.abs(d)
        tie = ad[:, 0] == ad[:, 1]
        axis = np.where(tie, rng.integers(0, 2, N), (ad[:, 1] > ad[:, 0]).astype(int))
        chase = np.zeros_like(x); chase[np.arange(N), axis] = np.sign(d[np.arange(N), axis])
        close = (ad.sum(1) < 10) & (rng.random(N) < 0.75)
        prop = x + np.where(close[:, None], chase, rand)
    elif kind == "cows":
        dirs = np.concatenate([CARD, np.zeros((4, 2), int)])
        prop = x + dirs[rng.integers(0, 8, N)]
    else:                                                          # skeletons: approach / flee / random (game_logic)
        d = player - x; dist = np.abs(d).sum(1)
        rand = CARD[rng.integers(0, 4, N)]
        ad = np.abs(d); axis = (ad[:, 1] > ad[:, 0]).astype(int)
        toward = np.zeros_like(x); toward[np.arange(N), axis] = np.sign(d[np.arange(N), axis])
        mv = np.where((dist >= 10)[:, None], toward, np.where((dist <= 3)[:, None], -toward, rand))
        mv = np.where((rng.random(N) < 0.85)[:, None], rand, mv)
        prop = x + mv
    inb = (prop >= 0).all(1) & (prop < mp.shape[0]).all(1)
    pc = np.clip(prop, 0, mp.shape[0] - 1)
    ok = inb & ~SOLID[mp[pc[:, 0], pc[:, 1]].astype(int)]
    occ = blockers(R, s, kind, R["_j"])
    if len(occ):
        ok &= ~(prop[:, None, :] == occ[None]).all(-1).any(1)
    ok &= ~(prop == player).all(1)
    despawn = np.abs(x - player).sum(1) >= 14                      # tested on the old position
    return np.where(ok[:, None], prop, x), despawn


def stats(x, w, true, player, rng):
    """Belief statistics for weighted particles x [N, 2] (w in {0, 1}) and log-likelihoods of the true cell."""
    xs = x[w]
    cells, counts = np.unique(xs, axis=0, return_counts=True)
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    mu = xs.mean(0); cov = np.cov(xs.T) + np.eye(2) * 0.25          # +1/4: a cell has unit width
    # 95% support components (8-connected)
    order = np.argsort(-p); keep = order[: int(np.searchsorted(np.cumsum(p[order]), 0.95) + 1)]
    sup = {tuple(c) for c in cells[keep]}
    comps, seen = 0, set()
    for c in sup:
        if c in seen:
            continue
        comps += 1; stack = [c]; seen.add(c)
        while stack:
            a = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    b = (a[0] + dr, a[1] + dc)
                    if b in sup and b not in seen:
                        seen.add(b); stack.append(b)
    # log-likelihoods of the true cell on a window around it
    R_ = 24
    gr = np.stack(np.meshgrid(np.arange(true[0] - R_, true[0] + R_ + 1), np.arange(true[1] - R_, true[1] + R_ + 1),
                              indexing="ij"), -1).reshape(-1, 2)
    ti = len(gr) // 2
    eps = 1e-5                                                     # per-cell floor, identical for every estimate
    floor = lambda q: (q / q.sum() + eps) / (1 + eps * len(q))
    lut = {tuple(c): q for c, q in zip(cells, p)}
    exact = floor(np.array([lut.get(tuple(c), 0.0) for c in gr]))
    inv = np.linalg.inv(cov); dd = gr - mu
    g0 = np.exp(-0.5 * np.einsum("ni,ij,nj->n", dd, inv, dd))
    g, gt = floor(g0), floor(g0 * ~visible(gr - player))
    out = {"ent": ent, "ncells": float(np.exp(ent)), "spread": float(np.sqrt(np.trace(cov - np.eye(2) * 0.25))),
           "components": comps, "ll_exact": float(np.log(exact[ti])), "ll_gauss": float(np.log(g[ti])),
           "ll_gauss_trunc": float(np.log(gt[ti]))}
    for K in (4, 16):
        smp = xs[rng.integers(0, len(xs), K)]
        kd = np.zeros(len(gr))
        for sx in smp:
            dist = np.abs(gr - sx).sum(1)
            kd += np.where(dist == 0, 0.6, np.where(dist == 1, 0.1, 0.0))
        kd = floor(kd)
        out[f"ll_k{K}"] = float(np.log(kd[ti]))
    return out


def main():
    out_path, n_eps, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(20261025)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps_ = RP._shard(sh)["episodes"]
        for i in rng.permutation([i for i, e in enumerate(eps_) if e["split"] == "train" and len(e["actions_taken"]) >= 64])[:2].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    print("replay check (max pixel diff):", RP.verify(*picks[0], checks=4), flush=True)
    run = scan_fn(); rows = []
    for e, (sh, slot) in enumerate(picks):
        acts = RP.episode_fields(sh, slot)["actions_taken"].numpy().astype(np.int32); T = len(acts)
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        R = {k: v[: T + 1] for k, v in R.items()}
        pos = R["pos"].astype(int)
        for kind in ("zombies", "cows"):                               # skeleton shooting needs cooldowns: not simulated
            P_, M_ = R[f"{kind}_p"].astype(int), R[f"{kind}_m"]
            vis = M_ & visible(P_ - pos[:, None, :])
            for j in range(P_.shape[1]):
                for t0 in range(T - 1):
                    if not (vis[t0, j] and M_[t0 + 1, j] and not vis[t0 + 1, j] and np.abs(P_[t0 + 1, j] - P_[t0, j]).sum() <= 1):
                        continue
                    R["_j"] = j
                    x = np.repeat(P_[t0, j][None], N, 0); p_gone = 0.0   # posterior that it despawned
                    for k in range(1, HMAX + 1):
                        s = t0 + k - 1
                        if s + 1 > T:
                            break
                        x, desp = step_particles(x, kind, R, s, rng)
                        true_alive = M_[s + 1, j] and np.abs(P_[s + 1, j] - P_[s, j]).sum() <= 1
                        if not true_alive:
                            rows.append({"kind": kind, "k": k, "event": "true_gone", "p_exist": 1 - p_gone})
                            break
                        seen = visible(x - pos[s + 1][None]) & ~desp
                        if vis[s + 1, j]:                      # true re-entry: predictive of the true cell before rejection
                            hit = ~desp & (x == P_[s + 1, j]).all(1)
                            rows.append({"kind": kind, "k": k, "event": "reentry", "ll_reentry": float(np.log((1 - p_gone) * hit.mean() + 1e-5)),
                                         "p_visible": float((1 - p_gone) * seen.mean()), "light": float(R["light"][s + 1])})
                            break
                        f_d, f_s = float(desp.mean()), float(seen.mean())
                        stay, gone = (1 - p_gone) * (1 - f_d - f_s), p_gone + (1 - p_gone) * f_d
                        p_gone = gone / max(stay + gone, 1e-12)
                        w = ~desp & ~seen
                        if w.sum() < 16:
                            rows.append({"kind": kind, "k": k, "event": "depleted"})
                            break
                        st = stats(x, w, P_[s + 1, j], pos[s + 1], rng)
                        st.update({"kind": kind, "k": k, "event": "out", "removed_frac": f_s / max(1 - f_d, 1e-12),
                                   "p_exist": 1 - p_gone, "light": float(R["light"][s + 1])})
                        rows.append(st)
                        x = x[rng.choice(np.nonzero(w)[0], N)]          # resample the surviving hypotheses
        print(f"episode {e + 1}/{len(picks)} T={T} rows={len(rows)}", flush=True)
        json.dump(rows, open(out_path, "w"))
    print("saved", out_path)


if __name__ == "__main__":
    main()
