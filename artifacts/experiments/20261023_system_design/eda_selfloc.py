"""S1-S5 (2026-10-23): facts an option-3 memory (learned self-location + location-keyed episodic store) would face in
Craftax-Classic. Simulator replay of TRAIN-split support-v2 episodes; evaluation-only state; nothing trained.

S1  Self-motion: for move actions, P(moved | target cell block, mob at target, asleep) and the accuracy of the best
    rule on that visible context (is path integration determined by the current view?). Plus the rate of successful
    moves whose 7x9 block view is IDENTICAL before and after (a scroll invisible in the terrain).
S2  Aliasing: fraction of frames whose 7x9 block view already occurred earlier in the episode at a DIFFERENT player
    position (content-only "where am I" retrieval would be wrong), vs at the same position.
S3  Staleness: for terrain re-entering view (seen before), P(block changed since last seen), by gap.
S4  Entities: for mobs re-entering view (same entity), Manhattan displacement between last-seen and re-entry positions,
    by gap.
S5  Value of recall vs no recall for re-entering terrain (coverage is not error): plug-in conditional entropy and argmax
    accuracy of the block given (a) nothing, (b) the visible inward neighbours (the 3 cells adjacent to it that are in
    view at entry), (c) memory (the block last seen there), with counts from all events (large-sample plug-in).
"""
import json, os, sys
from collections import Counter, defaultdict
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}       # LEFT RIGHT UP DOWN (DIRECTIONS[1..4])


def scan_fn():
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position, "map": s.map.astype(jnp.int8), "mob": s.mob_map, "sleep": s.is_sleeping,
                     **{f"{m}_p": getattr(s, m).position for m in ("zombies", "cows", "skeletons")},
                     **{f"{m}_m": getattr(s, m).mask for m in ("zombies", "cows", "skeletons")}}

    def run(reset_key, env_keys, actions):
        _, s0 = env.reset(reset_key, params)

        def body(s, xs):
            k, a = xs
            _, nxt, _, _, _ = env.step(k, s, a, params)
            return nxt, rec(nxt)
        _, seq = jax.lax.scan(body, s0, (env_keys, actions))
        return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), rec(s0), seq)
    return jax.jit(run)


def H(counter_by_ctx):
    tot = sum(sum(c.values()) for c in counter_by_ctx.values()); h = 0.0; acc = 0
    for c in counter_by_ctx.values():
        n = sum(c.values()); p = np.array(list(c.values())) / n
        h += n / tot * float(-(p * np.log2(p)).sum()); acc += max(c.values())
    return {"bits": h, "argmax_acc": acc / tot, "n": tot, "contexts": len(counter_by_ctx)}


def view(m, p):
    pad = np.pad(m, 4, constant_values=1)
    return pad[p[0] - 3 + 4:p[0] + 4 + 4, p[1] - 4 + 4:p[1] + 5 + 4]


def main():
    n_eps, out = int(sys.argv[1]), sys.argv[2]
    rng = np.random.default_rng(20261101)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps = RP._shard(sh)["episodes"]
        for i in rng.permutation([i for i, e in enumerate(eps) if e["split"] == "train" and len(e["actions_taken"]) >= 64])[:3].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    run = scan_fn()
    move_ctx = defaultdict(Counter); invisible = [0, 0]
    alias = Counter(); stale = defaultdict(lambda: [0, 0]); disp = defaultdict(list)
    s5 = {"none": defaultdict(Counter), "neighbours": defaultdict(Counter), "memory": defaultdict(Counter),
          "memory+neighbours": defaultdict(Counter)}
    for sh, slot in picks:
        f = RP.episode_fields(sh, slot); acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        pos = R["pos"][:T + 1].astype(int); mp = R["map"]; mob = R["mob"]
        views = [view(mp[t], pos[t]).tobytes() for t in range(T + 1)]
        # S1
        for t in range(T):
            a = int(acts[t])
            if a in MOVES:
                d = MOVES[a]; tgt = (pos[t, 0] + d[0], pos[t, 1] + d[1])
                inb = 0 <= tgt[0] < 64 and 0 <= tgt[1] < 64
                blk = int(mp[t][tgt]) if inb else -1; mb = bool(mob[t][tgt]) if inb else False
                moved = bool((pos[t + 1] != pos[t]).any())
                move_ctx[(blk, mb, bool(R["sleep"][t]))][moved] += 1
                if moved:
                    invisible[1] += 1; invisible[0] += int(views[t + 1] == views[t])
        # S2
        first_pos = {}
        for t in range(T + 1):
            v = views[t]
            if v in first_pos:
                alias["same_pos" if any((pp == pos[t]).all() for pp in first_pos[v]) else "other_pos"] += 1
                first_pos[v].append(pos[t])
            else:
                alias["new_view"] += 1; first_pos[v] = [pos[t]]
        # S3 + S5 terrain re-entries
        last_seen_t = np.full((64, 64), -1); last_seen_b = np.zeros((64, 64), int)
        for t in range(T + 1):
            if t >= 1:
                prev = {(pos[t - 1, 0] + r - 3, pos[t - 1, 1] + c - 4) for r in range(7) for c in range(9)}
                cur_view = {(pos[t, 0] + r - 3, pos[t, 1] + c - 4) for r in range(7) for c in range(9)}
                for r in range(7):
                    for c in range(9):
                        w = (pos[t, 0] + r - 3, pos[t, 1] + c - 4)
                        if w in prev or not (0 <= w[0] < 64 and 0 <= w[1] < 64):
                            continue
                        blk = int(mp[t][w])
                        nb = tuple(int(mp[t][(w[0] + dr, w[1] + dc)]) if (w[0] + dr, w[1] + dc) in cur_view and (w[0] + dr, w[1] + dc) in prev
                                   and 0 <= w[0] + dr < 64 and 0 <= w[1] + dc < 64 else -1
                                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                        s5["none"][0][blk] += 1; s5["neighbours"][nb][blk] += 1
                        if last_seen_t[w] >= 0:
                            g = t - last_seen_t[w]; gb = "le16" if g <= 16 else "le64" if g <= 64 else "le256" if g <= 256 else "gt256"
                            stale[gb][0] += int(last_seen_b[w] != blk); stale[gb][1] += 1
                            s5["memory"][("seen", int(last_seen_b[w]))][blk] += 1
                            s5["memory+neighbours"][("seen", int(last_seen_b[w]), nb)][blk] += 1
                        else:
                            s5["memory"][("unseen",)][blk] += 1
                            s5["memory+neighbours"][("unseen", nb)][blk] += 1
            r0, c0 = pos[t, 0] - 3, pos[t, 1] - 4
            for r in range(7):
                for c in range(9):
                    w = (r0 + r, c0 + c)
                    if 0 <= w[0] < 64 and 0 <= w[1] < 64:
                        last_seen_t[w] = t; last_seen_b[w] = int(mp[t][w])
        # S4 entity displacement during absence
        for m in ("zombies", "cows", "skeletons"):
            P = R[f"{m}_p"][:T + 1].astype(int); M = R[f"{m}_m"][:T + 1]
            vis = M & (np.abs(P[..., 0] - pos[:, None, 0]) <= 3) & (np.abs(P[..., 1] - pos[:, None, 1]) <= 4)
            for j in range(M.shape[1]):
                last = None
                for t in range(1, T + 1):
                    if not M[t, j]:
                        last = None; continue
                    if vis[t, j] and not vis[t - 1, j] and last is not None:
                        g = t - last; gb = "le4" if g <= 4 else "le16" if g <= 16 else "le64" if g <= 64 else "gt64"
                        disp[(m, gb)].append(int(np.abs(P[t, j] - P[last, j]).sum()))
                    if vis[t, j]:
                        last = t
    blocked = {str(k): {"n": sum(v.values()), "p_moved": v[True] / max(sum(v.values()), 1)} for k, v in move_ctx.items()
               if sum(v.values()) >= 200}
    rule_acc = sum(max(v.values()) for v in move_ctx.values()) / sum(sum(v.values()) for v in move_ctx.values())
    res = {"episodes": len(picks),
           "S1_move_outcome_rule_accuracy_from_visible_target": rule_acc,
           "S1_contexts": dict(sorted(blocked.items(), key=lambda kv: -kv[1]["n"])[:14]),
           "S1_successful_moves_with_identical_terrain_view": invisible[0] / max(invisible[1], 1),
           "S2_alias": {k: v / sum(alias.values()) for k, v in alias.items()},
           "S3_stale_by_gap": {k: {"p_changed": v[0] / v[1], "n": v[1]} for k, v in stale.items()},
           "S4_entity_displacement_by_gap": {f"{m}_{g}": {"n": len(v), "p50": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                                                          "zero": float(np.mean(np.array(v) == 0))} for (m, g), v in disp.items() if len(v) >= 30},
           "S5_reentering_terrain": {k: H(v) for k, v in s5.items()}}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
