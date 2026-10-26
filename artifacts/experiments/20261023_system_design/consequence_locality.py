"""Where on screen do consequences happen, and what glimpse budget would see them? (2026-10-23, design measurement)

Evaluation-only use of simulator state (nothing here trains anything). TRAIN-split support-v2 episodes are replayed exactly
(artifacts/eda/replay.py) and every transition t -> t+1 is scanned for consequential events, placed in the screen
coordinates of frame t+1 (rows 0-6 map, rows 7-8 HUD, 9 columns; player at (3, 4)):
  vitals    health / food / drink / energy changed            -> its HUD cell (7, 0..3)
  inventory an item count changed                             -> its HUD cell (renderer layout)
  block     a block in view changed (collect, place, grow)    -> that cell
  removed   a mob adjacent to the player at t is gone at t+1  -> its cell (eaten cow, killed zombie)
  damage    health fell and a zombie was adjacent / an arrow within 1 -> the attacker's cell (plus the HUD vitals event)
  threat    a zombie becomes adjacent (not adjacent at t)     -> its cell
Watch-list (not events): zombies at Manhattan distance 2-3 at t+1 (anticipation targets).
Reports event frequency, location heatmap, near-player / HUD shares, night static at event cells, and the oracle
coverage of all event cells in a step by k glimpses of g x g cells (exhaustive search over placements, k <= 2; greedy k=3),
plus a fixed allocation (3x3 around the player + the four vitals cells).
"""
import json, os, sys, time
from itertools import combinations
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
ROOT = "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA"
sys.path.insert(0, ROOT + "/artifacts/eda")
import replay as RP

INV = [("wood", (7, 5)), ("stone", (7, 6)), ("coal", (7, 7)), ("iron", (7, 8)), ("diamond", (8, 0)), ("sapling", (7, 4)),
       ("wood_pickaxe", (8, 1)), ("stone_pickaxe", (8, 2)), ("iron_pickaxe", (8, 3)), ("wood_sword", (8, 4)),
       ("stone_sword", (8, 5)), ("iron_sword", (8, 6))]
VIT = [("player_health", (7, 0)), ("player_food", (7, 1)), ("player_drink", (7, 2)), ("player_energy", (7, 3))]


def rec(s):
    inv = s.inventory
    return {"pos": s.player_position, "map": s.map.astype(jnp.int8), "light": s.light_level, "sleep": s.is_sleeping,
            "vit": jnp.stack([getattr(s, n) for n, _ in VIT]).astype(jnp.int32),
            "inv": jnp.stack([getattr(inv, n) for n, _ in INV]).astype(jnp.int32),
            **{f"{m}_p": getattr(s, m).position for m in ("zombies", "cows", "skeletons", "arrows")},
            **{f"{m}_m": getattr(s, m).mask for m in ("zombies", "cows", "skeletons", "arrows")}}


def scan_fn():
    env, params, _, _, _ = RP.env_and_render()

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, rec(nxt)

        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), rec(s0), seq)

    return jax.jit(run)


def noise_cells():
    H, W = 49, 63
    xs, ys = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))
    t = (1 - np.exp(-0.5 * (xs ** 2 + ys ** 2) / 0.25)).T
    return t.reshape(7, 7, 9, 7).mean((1, 3))


def to_screen(world, pos):
    r, c = world[..., 0] - pos[0] + 3, world[..., 1] - pos[1] + 4
    return r, c


def events_for(R, t):
    """Event cells (set of (row, col, kind)) for transition t -> t+1, in frame t+1 screen coordinates."""
    ev = set()
    p0, p1 = R["pos"][t], R["pos"][t + 1]
    for i, (_, cell) in enumerate(VIT):
        if R["vit"][t + 1][i] != R["vit"][t][i]:
            ev.add((*cell, "vitals"))
    for i, (_, cell) in enumerate(INV):
        if R["inv"][t + 1][i] != R["inv"][t][i]:
            ev.add((*cell, "inventory"))
    for r in range(7):
        for c in range(9):
            w = (p1[0] + r - 3, p1[1] + c - 4)
            if 0 <= w[0] < 64 and 0 <= w[1] < 64 and R["map"][t + 1][w] != R["map"][t][w]:
                ev.add((r, c, "block"))
    for m in ("zombies", "cows", "skeletons"):
        P0, M0, M1 = R[f"{m}_p"][t], R[f"{m}_m"][t], R[f"{m}_m"][t + 1]
        d0 = np.abs(P0 - p0).sum(-1)
        for j in np.where(M0 & ~M1 & (d0 <= 1))[0]:
            r, c = to_screen(P0[j], p1)
            if 0 <= r < 7 and 0 <= c < 9:
                ev.add((int(r), int(c), "removed"))
    hit = R["vit"][t + 1][0] < R["vit"][t][0]
    zP0, zM0, zP1, zM1 = R["zombies_p"][t], R["zombies_m"][t], R["zombies_p"][t + 1], R["zombies_m"][t + 1]
    adj0 = zM0 & (np.abs(zP0 - p0).sum(-1) == 1)
    adj1 = zM1 & (np.abs(zP1 - p1).sum(-1) == 1)
    if hit:
        for j in np.where(adj0 | adj1)[0]:
            r, c = to_screen(zP1[j] if adj1[j] else zP0[j], p1)
            if 0 <= r < 7 and 0 <= c < 9:
                ev.add((int(r), int(c), "damage"))
        aP, aM = R["arrows_p"][t + 1], R["arrows_m"][t + 1]
        for j in np.where(aM & (np.abs(aP - p1).sum(-1) <= 1))[0]:
            r, c = to_screen(aP[j], p1)
            if 0 <= r < 7 and 0 <= c < 9:
                ev.add((int(r), int(c), "damage"))
    for j in np.where(adj1 & ~adj0)[0]:
        r, c = to_screen(zP1[j], p1)
        ev.add((int(r), int(c), "threat"))
    watch = []
    d1 = np.abs(zP1 - p1).sum(-1)
    for j in np.where(zM1 & (d1 >= 2) & (d1 <= 3))[0]:
        r, c = to_screen(zP1[j], p1)
        if 0 <= r < 7 and 0 <= c < 9:
            watch.append((int(r), int(c)))
    return ev, watch


def glimpse_masks(g):
    out = []
    for r in range(0, 9 - g + 1):
        for c in range(0, 9 - g + 1):
            m = np.zeros((9, 9), bool); m[r:r + g, c:c + g] = True; out.append(m.reshape(-1))
    return np.array(out)


def main():
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = sys.argv[2]
    rng = np.random.default_rng(20261023)
    man = RP.manifest()
    shards = rng.permutation(len(man["shards"]))
    picks = []
    for sh in shards.tolist():
        eps = RP._shard(sh)["episodes"]
        cand = [i for i, e in enumerate(eps) if e["split"] == "train" and len(e["actions_taken"]) >= 32]
        if cand:
            picks.append((sh, int(rng.choice(cand))))
        if len(picks) == n_eps:
            break
    run = scan_fn()
    G = {g: glimpse_masks(g) for g in (1, 2, 3)}
    fixed = np.zeros((9, 9), bool); fixed[2:5, 3:6] = True; fixed[7, 0:4] = True; fixed = fixed.reshape(-1)
    noise = noise_cells()
    heat = np.zeros((9, 9)); kinds = {}; n_tr = n_ev = 0; ncells = []; near = hud = far = 0
    cover = {(g, k): 0 for g in (1, 2, 3) for k in (1, 2, 3)}; cover_fixed = 0; partial = {(g, k): 0.0 for g in (1, 2, 3) for k in (1, 2, 3)}
    night_map_cells = 0; map_cells = 0; static_at_events = []; watch_heat = np.zeros((9, 9)); n_watch = 0; eps_meta = []
    t0 = time.time()
    for e_i, (sh, slot) in enumerate(picks):
        f = RP.episode_fields(sh, slot)
        acts = f["actions_taken"].numpy().astype(np.int32)
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= len(acts))
        pad = np.zeros(b, np.int32); pad[:len(acts)] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        T = len(acts)
        eps_meta.append({"shard": sh, "slot": slot, "len": T, "epsilon": f["epsilon"]})
        for t in range(T - 1 if bool(f["terminated"][-1]) else T):   # skip the terminal transition's post-death frame
            n_tr += 1
            ev, watch = events_for(R, t)
            for (r, c) in watch:
                watch_heat[r, c] += 1; n_watch += 1
            if not ev:
                continue
            n_ev += 1
            cells = sorted({(r, c) for r, c, _ in ev})
            ncells.append(len(cells))
            for r, c, k in ev:
                kinds[k] = kinds.get(k, 0) + 1
            for r, c in cells:
                heat[r, c] += 1
                if r >= 7: hud += 1
                elif max(abs(r - 3), abs(c - 4)) <= 1: near += 1
                else: far += 1
                if r < 7:
                    map_cells += 1
                    if R["light"][t + 1] < 0.5:
                        night_map_cells += 1
                        static_at_events.append(float(2 * (0.5 - R["light"][t + 1]) * noise[r, c]))
            target = np.zeros(81, bool)
            for r, c in cells: target[r * 9 + c] = True
            nt = target.sum()
            cover_fixed += int((target & ~fixed).sum() == 0)
            for g in (1, 2, 3):
                hits = (G[g] & target).sum(1)                      # per placement
                best1 = hits.max()
                cover[(g, 1)] += int(best1 == nt); partial[(g, 1)] += best1 / nt
                idx = np.where(hits > 0)[0]
                best2 = best1
                for a, bb in combinations(idx.tolist(), 2):
                    best2 = max(best2, ((G[g][a] | G[g][bb]) & target).sum())
                cover[(g, 2)] += int(best2 == nt); partial[(g, 2)] += best2 / nt
                rem = target.copy(); got = 0                         # greedy 3
                for _ in range(3):
                    h = (G[g] & rem).sum(1); j = int(h.argmax()); got += h[j]; rem &= ~G[g][j]
                cover[(g, 3)] += int(got == nt); partial[(g, 3)] += got / nt
        if (e_i + 1) % 50 == 0:
            print(f"{e_i + 1} episodes, {n_tr} transitions, {time.time() - t0:.0f}s", flush=True)
    res = {"episodes": len(picks), "transitions": n_tr, "event_transitions": n_ev, "event_rate": n_ev / n_tr,
           "event_cells_per_event_step": {"mean": float(np.mean(ncells)), "p50": float(np.median(ncells)),
                                          "p90": float(np.percentile(ncells, 90)), "max": int(np.max(ncells))},
           "kinds": kinds, "cell_share": {"hud": hud / (hud + near + far), "near_player_3x3": near / (hud + near + far),
                                           "elsewhere_on_map": far / (hud + near + far)},
           "heat": heat.tolist(), "watch_zombie_heat_dist2to3": watch_heat.tolist(), "watch_count": n_watch,
           "all_events_covered": {f"g{g}_k{k}": cover[(g, k)] / n_ev for (g, k) in cover},
           "mean_fraction_covered": {f"g{g}_k{k}": partial[(g, k)] / n_ev for (g, k) in partial},
           "fixed_allocation_3x3_player_plus_vitals_all_covered": cover_fixed / n_ev,
           "night_share_of_map_event_cells": night_map_cells / max(map_cells, 1),
           "static_mask_at_night_map_event_cells": {"mean": float(np.mean(static_at_events)) if static_at_events else None,
                                                    "p90": float(np.percentile(static_at_events, 90)) if static_at_events else None},
           "episodes_meta": eps_meta}
    with open(out, "w") as fh:
        json.dump(res, fh)
    print(json.dumps({k: v for k, v in res.items() if k not in ("heat", "watch_zombie_heat_dist2to3", "episodes_meta")}, indent=1))
    np.set_printoptions(precision=3, suppress=True)
    print("event-cell heat (share of event cells):"); print(heat / heat.sum())


if __name__ == "__main__":
    main()
