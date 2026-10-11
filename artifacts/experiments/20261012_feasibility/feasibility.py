"""10-12 feasibility study (NOTEBOOK "10-12 FEASIBILITY STUDY", predeclared before this code): is the whiteboard's hypothesis
promising? Simulator only (Craftax-Classic, JAX on CPU), no learned model.
Roots: support-v2 episodes replayed exactly from their stored actions and keys (artifacts/eda/replay.py), one root per episode,
each from a different shard; root t uniform in [64, len - 17]; seed 20261012.
  roots   pick the roots, replay each to t, check the render against the stored frame -> roots.pt
  q1      history: replay each prefix with per-step counters; a source-derived candidate tracker per counter (hunger, thirst,
          fatigue, recover) started L transitions before the root (L in 1, 4, 15, 32, 64, full). Observable inputs: HUD values,
          the sleep flag, eat / drink events (DO on a visible cow / ripe plant / water), damage. Invisible: threshold resets
          while the stat is clipped (food / drink / energy 0, energy or health 9). Checks: the equations reproduce the
          simulator's counters at every step, and the true value is never eliminated. Reports per counter and L: identified
          (one candidate), and the next-16-step VISIBLE HUD event schedule determined (all candidates agree; an event at a
          clipped stat shows nothing and is not counted). Recover assumes damage events are observable.
  q2      one-step innovation: 256 keys x 17 actions from each root; outcomes hashed on the symbolic visible observation and on
          the full world (all mobs incl. off-screen, arrows, map, plants, inventory, player). Plug-in and Miller-Madow
          entropy (K = 256, and the first 64 keys as a bias check), Good-Turing missing mass, outcomes for 95 / 99% coverage.
  q3      decision value at horizon 16 with the recorded continuation: per root x 17 first actions, a 32-key selection bank and
          an independent 32-key judgment bank (common keys across actions within a bank), and a coherent deterministic
          mode-world (each step: the most frequent next visible observation of 64 forks, continue from that state). Expected
          safe (1 - P_judge(dead by 16)) and native 16-step return for: uniform, mode-world, S = 1 / 4 / 16 / 32 samples.
Usage: feasibility.py roots|q1|q2|q3 [--roots N]  -> artifacts/eda/feasibility_v1/
"""
import os
import sys
import time
import json
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT / "artifacts/eda"))
import replay as RP  # noqa: E402

OUT = ROOT / "artifacts/eda/feasibility_v1"
SEED = 20261012
H = 16
LS = (1, 4, 15, 32, 64, None)                                              # None = full prefix from reset
DO = 5


def is_dead(s):
    from craftax.craftax_classic.constants import BlockType
    return (s.map[s.player_position[0], s.player_position[1]] == BlockType.LAVA.value) | (s.player_health <= 0)


def pick_roots(n):
    rng = np.random.default_rng(SEED)
    shards = rng.permutation(len(RP.manifest()["shards"]))
    roots = []
    for sh in shards.tolist():
        eps = RP._shard(sh)["episodes"]
        slots = [i for i, e in enumerate(eps) if len(e["actions_taken"]) >= 100]
        if not slots:
            continue
        slot = int(rng.choice(slots))
        n_act = len(eps[slot]["actions_taken"])
        t = int(rng.integers(64, n_act - 17 + 1))
        roots.append({"shard": sh, "slot": slot, "t": t, "len": n_act})
        if len(roots) == n:
            break
    return roots


def cmd_roots(n):
    _, _, _, _, frame = RP.env_and_render()
    roots = pick_roots(n)
    worst = 0
    for r in roots:
        s = RP.advance_to(r["shard"], r["slot"], r["t"])
        stored = RP.episode_fields(r["shard"], r["slot"])["observations"][r["t"]].numpy()
        diff = int(np.abs(np.asarray(frame(s)).astype(np.int32) - stored.astype(np.int32)).max())
        r["render_maxdiff"] = diff
        worst = max(worst, diff)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(roots, OUT / "roots.pt")
    print(json.dumps({"roots": len(roots), "render_maxdiff_worst": worst,
                      "episode_len_median": int(np.median([r["len"] for r in roots]))}), flush=True)


def load_roots(n):
    roots = torch.load(OUT / "roots.pt", weights_only=False)
    return roots[:n] if n else roots


def root_state(r):
    return RP.advance_to(r["shard"], r["slot"], r["t"])


# ---------------------------------------------------------------------------------------------------------------- Q1
@jax.jit
def _prefix_scan(reset_key, env_keys, actions, stop):
    env, params, _, _, _ = RP.env_and_render()
    _, state = env.reset(reset_key, params)

    def rec(s):
        return {"food": s.player_food, "drink": s.player_drink, "energy": s.player_energy, "health": s.player_health,
                "sleep": s.is_sleeping, "hunger": s.player_hunger, "thirst": s.player_thirst,
                "fatigue": s.player_fatigue, "recover": s.player_recover}

    def body(carry, xs):
        s, i = carry
        k, a = xs
        _, nxt, _, _, _ = env.step(k, s, a, params)
        live = i < stop
        nxt = jax.tree.map(lambda x, y: jnp.where(live, x, y), nxt, s)
        return (nxt, i + 1), rec(nxt)

    first = rec(state)
    _, seq = jax.lax.scan(body, (state, 0), (env_keys, actions))
    return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), first, seq)


def prefix_records(r, upto):
    reset_key, env_keys = RP._slot_keys(r["shard"], r["slot"])
    actions = RP.episode_fields(r["shard"], r["slot"])["actions_taken"].numpy()
    bucket = next(b for b in RP.BUCKETS if b >= upto)
    pad = np.zeros(bucket, dtype=np.int32)
    pad[:min(len(actions), bucket)] = actions[:bucket]
    rec = _prefix_scan(reset_key, env_keys[:bucket], jnp.asarray(pad), upto)
    return {k: np.asarray(v)[:upto + 1].astype(np.float64) for k, v in rec.items()}     # index i = state after i actions


COUNTERS = {                     # name: (grid, stat, threshold-up, threshold-down)
    "hunger": (np.arange(0, 26.0, 0.5), "food"),
    "thirst": (np.arange(0, 21.0, 0.5), "drink"),
    "fatigue": (np.arange(-10, 31.0, 1.0), "energy"),
    "recover": (np.arange(-15, 26.0, 0.5), "health"),
}


def step_counter(name, c, rec, i):
    """candidate values c (array) at state i -> (values at state i+1, predicted visible stat change), using the inputs of the
    transition i -> i+1 (sleep flag of the state after the step, as craftax's update reads it)"""
    sl = rec["sleep"][i + 1] > 0.5
    if name in ("hunger", "thirst"):
        stat = "food" if name == "hunger" else "drink"
        thr = 25.0 if name == "hunger" else 20.0
        ate = rec[f"_{name}_reset"][i]
        v = np.where(ate, 0.0, c) + (0.5 if sl else 1.0)
        cross = v > thr
        v = np.where(cross, 0.0, v)
        return v, -cross.astype(np.float64)                                    # stat change from the counter: -1 per crossing
    if name == "fatigue":
        v = np.minimum(c - 1, 0) if sl else c + 1
        up, down = v > 30, v < -10
        return np.where(up | down, 0.0, v), np.where(up, -1.0, np.where(down, 1.0, 0.0))
    nec = (rec["food"][i + 1] > 0) & (rec["drink"][i + 1] > 0) & ((rec["energy"][i + 1] > 0) | sl)
    add = (2.0 if sl else 1.0) if nec else (-0.5 if sl else -1.0)
    v = c + add
    up, down = v > 25, v < -15
    return np.where(up | down, 0.0, v), np.where(up, 1.0, np.where(down, -1.0, 0.0))


def annotate(rec):
    """derive the observable event inputs from the replayed truth: eat / drink resets and damage"""
    n = len(rec["food"]) - 1
    for name in ("hunger", "thirst"):
        rs = np.zeros(n, dtype=bool)
        for i in range(n):
            sl = rec["sleep"][i + 1] > 0.5
            inc = 0.5 if sl else 1.0
            nxt = rec[name][i + 1]
            # reset by eating / drinking: the next value equals the increment while the plain update (increment, or the
            # threshold reset to 0 when it crosses) does not. Eating ON a crossing step resets to the increment, not 0
            # (fixed 10-12 after the full run's equation check: max error 1.0, 47 violations).
            plain = 0.0 if rec[name][i] + inc > (25 if name == "hunger" else 20) else rec[name][i] + inc
            stat = "food" if name == "hunger" else "drink"
            # food / drink rise only by eating / drinking (a visible HUD increase); this also catches eating at counter 0,
            # where the counter alone cannot show the reset (fixed 10-12: 21 violations)
            rs[i] = (abs(nxt - inc) < 1e-6 and abs(plain - inc) > 1e-6) or rec[stat][i + 1] > rec[stat][i]
        rec[f"_{name}_reset"] = rs
    # stat change explained by the counters; damage = the rest of the health change
    rec["_damage"] = np.zeros(n)
    for i in range(n):
        _, dh = step_counter("recover", np.array([rec["recover"][i]]), rec, i)
        expected = min(rec["health"][i] + dh[0], 9) if dh[0] > 0 else rec["health"][i] + dh[0]
        rec["_damage"][i] = expected - rec["health"][i + 1]
    return rec


def check_equations(rec):
    """the source-derived update must reproduce the simulator's counters at every step"""
    worst = {}
    for name in COUNTERS:
        err = 0.0
        for i in range(len(rec["food"]) - 1):
            v, _ = step_counter(name, np.array([rec[name][i]]), rec, i)
            err = max(err, abs(v[0] - rec[name][i + 1]))
        worst[name] = err
    return worst


def observed_change(name, rec, i):
    """the visible stat change attributable to the counter at transition i, or None when clipped (invisible)"""
    stat = COUNTERS[name][1]
    a, b = rec[stat][i], rec[stat][i + 1]
    if name in ("hunger", "thirst"):
        if rec[f"_{name}_reset"][i]:
            return 0.0                                                          # eating / drinking: no crossing this step
        return None if a == 0 else b - a
    if name == "fatigue":
        # energy -1 is invisible at 0, energy +1 invisible at 9; report the change only when both directions are visible
        if a == 0 or a == 9:
            return None
        return b - a
    d = b - a + rec["_damage"][i]                                              # health change net of damage
    return None if a == 9 else d


def track(name, rec, t, L):
    grid = COUNTERS[name][0]
    start = 0 if (L is None or L >= t) else t - L
    c = np.array([rec[name][0]]) if start == 0 else grid.copy()
    violations = 0
    for i in range(start, t):
        v, pred = step_counter(name, c, rec, i)
        obs = observed_change(name, rec, i)
        if obs is not None:
            keep = np.isclose(pred, obs)
            v = v[keep]
        c = np.unique(np.round(v, 3))
        if not np.any(np.isclose(c, rec[name][i + 1])):
            violations += 1
            c = np.unique(np.append(c, rec[name][i + 1]))
    # next-16-step event schedule per candidate, under the factual inputs
    scheds = set()
    for v0 in c:
        x = np.array([v0]); ev = []
        for i in range(t, min(t + H, len(rec["food"]) - 1)):
            x, pred = step_counter(name, x, rec, i)
            e = int(pred[0]); stat = rec[COUNTERS[name][1]][i]                # count only VISIBLE HUD events (factual stat)
            visible = (e < 0 and stat > 0) or (e > 0 and stat < 9)
            ev.append(e if visible else 0)
        scheds.add(tuple(ev))
    return {"n": int(len(c)), "identified": len(c) == 1, "schedule_determined": len(scheds) == 1, "violations": violations,
            "true_has_event": any(e != 0 for e in next(iter(scheds))) if len(scheds) == 1 else None}


def cmd_q1(n):
    roots = load_roots(n)
    rows, eq_worst = [], {k: 0.0 for k in COUNTERS}
    for ri, r in enumerate(roots):
        rec = annotate(prefix_records(r, r["t"] + H))
        for k, v in check_equations(rec).items():
            eq_worst[k] = max(eq_worst[k], v)
        for name in COUNTERS:
            for L in LS:
                res = track(name, rec, r["t"], L)
                rows.append({"root": ri, "counter": name, "L": "full" if L is None else L, **res})
        if ri % 40 == 0:
            print(f"q1 root {ri}/{len(roots)}", json.dumps(eq_worst), flush=True)
    summary = {"equation_check_max_abs_error": eq_worst, "violations_total": int(sum(x["violations"] for x in rows)), "by": {}}
    for name in COUNTERS:
        for L in LS:
            lk = "full" if L is None else L
            sel = [x for x in rows if x["counter"] == name and x["L"] == lk]
            summary["by"][f"{name}/L={lk}"] = {
                "identified": round(float(np.mean([x["identified"] for x in sel])), 4),
                "schedule_determined": round(float(np.mean([x["schedule_determined"] for x in sel])), 4),
                "mean_candidates": round(float(np.mean([x["n"] for x in sel])), 2)}
    (OUT / "q1.json").write_text(json.dumps(summary, indent=1))
    torch.save(rows, OUT / "q1_rows.pt")
    print(json.dumps(summary), flush=True)


# ---------------------------------------------------------------------------------------------------------------- Q2 / Q3
@jax.jit
def _fork(state, keys, actions):                                           # one state, [K] keys, [A] actions -> [A, K]
    env, params, _, _, _ = RP.env_and_render()

    def one(k, a):
        obs, nxt, rew, _, _ = env.step(k, state, a, params)
        return obs, nxt, rew, is_dead(nxt)
    return jax.vmap(lambda a: jax.vmap(lambda k: one(k, a))(keys))(actions)


@jax.jit
def _fork_many(states, keys, actions):                                     # [A] states, [K] keys, [A] actions -> [A, K]
    env, params, _, _, _ = RP.env_and_render()

    def one(s, k, a):
        obs, nxt, rew, _, _ = env.step(k, s, a, params)
        return obs, nxt, rew, is_dead(nxt)
    return jax.vmap(lambda s, a: jax.vmap(lambda k: one(s, k, a))(keys))(states, actions)


def world_vec(nxt):
    """full-world semantic outcome per fork: all mob arrays (masked), arrows, map, plants, inventory, player"""
    m = lambda mob: [jnp.where(mob.mask[..., None], mob.position, -1).reshape(*mob.mask.shape[:-1], -1),
                     jnp.where(mob.mask, mob.health, -1), jnp.where(mob.mask, mob.attack_cooldown, -99)]
    parts = m(nxt.zombies) + m(nxt.cows) + m(nxt.skeletons) + m(nxt.arrows)
    parts += [jnp.where(nxt.arrows.mask[..., None], nxt.arrow_directions, -9).reshape(*nxt.arrows.mask.shape[:-1], -1),
              nxt.map.reshape(*nxt.map.shape[:-2], -1), nxt.growing_plants_age, nxt.growing_plants_mask.astype(jnp.int32),
              jnp.stack(jax.tree.leaves(nxt.inventory), -1), nxt.player_position, nxt.player_direction[..., None],
              jnp.stack([nxt.player_health, nxt.player_food, nxt.player_drink, nxt.player_energy], -1),
              nxt.is_sleeping[..., None].astype(jnp.int32)]
    return jnp.concatenate([p.astype(jnp.int32).reshape(*p.shape[:2], -1) for p in parts], -1)


def ent_stats(rows):
    """rows [K, D] -> plug-in / Miller-Madow entropy (bits), outcomes, singletons, coverage counts"""
    _, counts = np.unique(rows, axis=0, return_counts=True)
    K = rows.shape[0]
    p = np.sort(counts)[::-1] / K
    h = float(-(p * np.log2(p)).sum())
    cum = np.cumsum(p)
    return {"H": h, "H_mm": h + (len(p) - 1) / (2 * K * np.log(2)), "m": int(len(p)), "singletons": int((counts == 1).sum()),
            "cov95": int(np.searchsorted(cum, 0.95 - 1e-9) + 1), "cov99": int(np.searchsorted(cum, 0.99 - 1e-9) + 1),
            "top": float(p[0])}


def cmd_q2(n):
    roots = load_roots(n)
    acts = jnp.arange(17)
    rows = []
    t0 = time.time()
    for ri, r in enumerate(roots):
        s = root_state(r)
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(SEED + 2), ri), 256)
        obs, nxt, _, dead = _fork(s, keys, acts)
        obs = np.asarray(obs); wv = np.asarray(world_vec(nxt)); dead = np.asarray(dead)
        for a in range(17):
            vis, vis64 = ent_stats(obs[a]), ent_stats(obs[a, :64])
            mapp, rest = ent_stats(obs[a, :, :1323]), ent_stats(obs[a, :, 1323:])
            wor = ent_stats(wv[a])
            rows.append({"root": ri, "action": a, "dead": float(dead[a].mean()),
                         **{f"vis_{k}": v for k, v in vis.items()}, "vis64_H_mm": vis64["H_mm"], "vis64_m": vis64["m"],
                         "map_H_mm": mapp["H_mm"], "nonmap_H_mm": rest["H_mm"],
                         **{f"world_{k}": v for k, v in wor.items()}})
        if ri % 40 == 0:
            print(f"q2 root {ri}/{len(roots)} {time.time() - t0:.0f}s", flush=True)
    q = lambda key, f: round(float(f([x[key] for x in rows])), 4)
    summary = {"root_actions": len(rows)}
    for key in ("vis_H_mm", "vis64_H_mm", "map_H_mm", "nonmap_H_mm", "world_H_mm", "vis_m", "vis_cov95", "vis_cov99",
                "vis_singletons", "world_m", "world_cov95"):
        v = np.array([x[key] for x in rows], dtype=float)
        summary[key] = {"mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
                        "p90": round(float(np.quantile(v, 0.9)), 4), "p99": round(float(np.quantile(v, 0.99)), 4),
                        "max": round(float(v.max()), 4)}
    summary["good_turing_missing_mass_vis_mean"] = round(float(np.mean([x["vis_singletons"] / 256 for x in rows])), 4)
    summary["share_cov95_le8"] = round(float(np.mean([x["vis_cov95"] <= 8 for x in rows])), 4)
    summary["share_vis_deterministic"] = round(float(np.mean([x["vis_m"] == 1 for x in rows])), 4)
    summary["readings"] = {"compact": summary["vis_H_mm"]["median"] <= 2 and summary["vis_H_mm"]["p90"] <= 4 and summary["share_cov95_le8"] >= 0.9}
    (OUT / "q2.json").write_text(json.dumps(summary, indent=1))
    torch.save(rows, OUT / "q2_rows.pt")
    print(json.dumps(summary), flush=True)


@jax.jit
def _roll(state, action_seqs, keys):                                       # [A,16] actions, [K] keys -> dead_by [A,K], ret [A,K]
    env, params, _, _, _ = RP.env_and_render()

    def one(seq, key):
        def body(c, xs):
            s, dead, ret = c
            a, k = xs
            _, nxt, rew, _, _ = env.step(k, s, a, params)
            ret = ret + jnp.where(dead, 0.0, rew)
            return (nxt, dead | is_dead(nxt), ret), None
        (_, dead, ret), _ = jax.lax.scan(body, (state, False, 0.0), (seq, jax.random.split(key, H)))
        return dead, ret
    return jax.vmap(lambda seq: jax.vmap(lambda k: one(seq, k))(keys))(action_seqs)


def mode_world(state, seqs, base_key, kmode=64):
    """per first action: follow the most frequent next VISIBLE observation of kmode forks at every step"""
    A = seqs.shape[0]
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (A,) + x.shape), state)
    dead = np.zeros(A, dtype=bool); ret = np.zeros(A); survived = np.zeros(A, dtype=int)
    for i in range(H):
        keys = jax.random.split(jax.random.fold_in(base_key, i), kmode)
        obs, nxt, rew, dd = _fork_many(states, keys, jnp.asarray(seqs[:, i]))
        obs = np.asarray(obs); rew = np.asarray(rew); dd = np.asarray(dd)
        pick = np.zeros(A, dtype=int)
        for a in range(A):
            u, inv, counts = np.unique(obs[a], axis=0, return_inverse=True, return_counts=True)
            mode = int(np.argmax(counts))
            pick[a] = int(np.flatnonzero(inv.reshape(-1) == mode)[0])
        states = jax.tree.map(lambda x: x[jnp.arange(A), jnp.asarray(pick)], nxt)
        ret += np.where(dead, 0.0, rew[np.arange(A), pick])
        newly = dd[np.arange(A), pick] & ~dead
        dead |= newly
        survived += (~dead).astype(int)
    return dead, ret, survived


def cmd_q3(n):
    roots = load_roots(n)
    rows = []
    t0 = time.time()
    for ri, r in enumerate(roots):
        s = root_state(r)
        cont = RP.episode_fields(r["shard"], r["slot"])["actions_taken"].numpy()[r["t"] + 1:r["t"] + H]
        seqs = np.stack([np.concatenate([[a], cont]) for a in range(17)]).astype(np.int32)
        ks = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(SEED + 3), ri), 32)       # selection bank
        kj = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(SEED + 4), ri), 32)       # judgment bank
        ds, rs = (np.asarray(x) for x in _roll(s, jnp.asarray(seqs), ks))
        dj, rj = (np.asarray(x) for x in _roll(s, jnp.asarray(seqs), kj))
        md, mr, msurv = mode_world(s, seqs, jax.random.fold_in(jax.random.PRNGKey(SEED + 5), ri))
        rows.append({"root": ri, "dead_sel": ds, "ret_sel": rs, "dead_judge": dj, "ret_judge": rj,
                     "mode_dead": md, "mode_ret": mr, "mode_survived": msurv})
        if ri % 20 == 0:
            print(f"q3 root {ri}/{len(roots)} {time.time() - t0:.0f}s", flush=True)
    torch.save(rows, OUT / "q3_rows.pt")
    summarize_q3(rows)


def summarize_q3(rows):
    def value_of(chosen, judge):                                           # mean judged value over the tied chosen set
        return float(judge[chosen].mean())

    def choose_min(est):
        return np.flatnonzero(np.isclose(est, est.min()))

    def choose_max(est):
        return np.flatnonzero(np.isclose(est, est.max()))
    per = {k: [] for k in ("uniform", "mode", "S1", "S4", "S16", "S32")}
    per_r = {k: [] for k in per}
    opp = []
    for r in rows:
        Pj = r["dead_judge"].mean(1); safe = 1 - Pj
        Rj = r["ret_judge"].mean(1)
        Ps = r["dead_sel"]; Rs = r["ret_sel"]
        opp.append(Ps.mean(1).max() > Ps.mean(1).min())
        per["uniform"].append(safe.mean()); per_r["uniform"].append(Rj.mean())
        per["mode"].append(value_of(choose_max(r["mode_survived"].astype(float)), safe))
        per_r["mode"].append(value_of(choose_max(r["mode_ret"]), Rj))
        for S in (1, 4, 16, 32):
            per[f"S{S}"].append(value_of(choose_min(Ps[:, :S].mean(1)), safe))
            per_r[f"S{S}"].append(value_of(choose_max(Rs[:, :S].mean(1)), Rj))
    opp = np.array(opp)
    def table(d, mask):
        out = {k: round(float(np.array(v)[mask].mean()), 4) for k, v in d.items()}
        idx = np.flatnonzero(mask)
        draws = np.random.default_rng(SEED).integers(0, len(idx), (2000, len(idx)))      # root bootstrap (one root per episode)
        for a, b in (("S16", "mode"), ("S1", "mode"), ("S32", "mode"), ("mode", "uniform"), ("S16", "S1"), ("S32", "S16")):
            diff = (np.array(d[a]) - np.array(d[b]))[idx]
            boot = diff[draws].mean(1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            out[f"{a}-{b}"] = [round(float(diff.mean()), 4), round(float(lo), 4), round(float(hi), 4), bool(lo > 0 or hi < 0)]
        return out
    summary = {"roots": len(rows), "opportunity_roots": int(opp.sum()),
               "safe16_opportunity": table(per, opp), "safe16_all": table(per, np.ones(len(rows), bool)),
               "return16_all": table(per_r, np.ones(len(rows), bool))}
    summary["readings"] = {"sampling_matters_beyond_coherence": summary["safe16_opportunity"]["S16-mode"][3] and summary["safe16_opportunity"]["S16-mode"][0] > 0}
    (OUT / "q3.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--roots") + 1]) if "--roots" in sys.argv else 0
    OUT.mkdir(parents=True, exist_ok=True)
    RP.env_and_render()                                                    # build (and cache) the env OUTSIDE any jit trace
    {"roots": lambda: cmd_roots(n or 240), "q1": lambda: cmd_q1(n), "q2": lambda: cmd_q2(n), "q3": lambda: cmd_q3(n),
     "q3sum": lambda: summarize_q3(torch.load(OUT / "q3_rows.pt", weights_only=False))}[cmd]()
