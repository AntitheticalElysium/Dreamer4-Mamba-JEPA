"""A5 (2026-10-25): after compensating self-motion, does the pixel residual isolate what changed independently?
Flow parsing (Rushton & Warren 2005): estimate the global self-motion component, remove it, and the remainder is
object motion. Candidate grounded target for entity dynamics: a per-cell "changed independently" map computed from
raw consecutive frames after aligning them by the estimated camera shift. Simulator state labels cells for evaluation
only; nothing is trained.

CLAIM: the aligned residual separates cells where something changed independently (an entity arrived, left or
  appeared; a block was edited) from cells that did not, by day and at night after simple pooling.
COMPETING EXPLANATIONS: (a) night static makes every cell's residual large; (b) the global light level changes between
  frames; (c) camera-shift errors misalign the whole frame; (d) an entity that stays put but changes facing / lighting.
CONTROLS: estimated vs true shift; raw pixels vs cells pooled to 3x3 and 1x1 (static is per-pixel i.i.d.); day / night;
  unchanged cells split into empty terrain vs a stationary entity; ROC AUC and recall at 5% false positives.
OUTCOMES: high separability at night after pooling -> the change map is a usable pixel-grounded target that cannot
  be satisfied by discarding entities; poor separability at night -> the target needs a learned or noise-aware
  estimate. It says nothing about how a model would use it.
Usage (project venv): eda_change_residual.py OUT_JSON N_EPISODES STRIDE
"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_correspondence import cells, SHIFTS
from eda_periphery_degradation import pool

KINDS = ("zombies", "cows", "skeletons", "arrows")


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


def auc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r); r = sums[inv] / cnt[inv]
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    out_path, n_eps, stride = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(20261026)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps_ = RP._shard(sh)["episodes"]
        for i in rng.permutation([i for i, e in enumerate(eps_) if e["split"] == "train" and len(e["actions_taken"]) >= 64])[:2].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    run = scan_fn()
    rows = []                                  # (night, shift_kind, res_raw, res_p3, res_p1, label) label 0 empty, 1 still entity, 2 changed
    for e, (sh, slot) in enumerate(picks):
        f = RP.episode_fields(sh, slot); acts = f["actions_taken"].numpy().astype(np.int32); T = len(acts)
        frames = f["observations"].numpy()
        reset_key, env_keys = RP._slot_keys(sh, slot)
        b = next(x for x in RP.BUCKETS if x >= T); pad = np.zeros(b, np.int32); pad[:T] = acts
        R = jax.tree.map(np.asarray, run(reset_key, env_keys[:b], jnp.asarray(pad)))
        pos = R["pos"].astype(int)
        for t in range(0, T, stride):
            c0, c1 = cells(frames[t]), cells(frames[t + 1])
            night = float(R["light"][t + 1]) < 0.5
            true_shift = tuple(-(pos[t + 1] - pos[t]))
            cost = []
            for dr, dc in SHIFTS:
                r0, r1 = max(0, -dr), min(7, 7 - dr); q0, q1 = max(0, -dc), min(9, 9 - dc)
                cost.append(np.abs(c0[r0:r1, q0:q1] - c1[r0 + dr:r1 + dr, q0 + dc:q1 + dc]).mean())
            est_shift = SHIFTS[int(np.argmin(cost))]
            # entity occupancy in world coordinates at t and t+1 (kind, slot) -> world cell
            occ0, occ1 = {}, {}
            for m in KINDS:
                P_, M_ = R[f"{m}_p"].astype(int), R[f"{m}_m"]
                for j in range(P_.shape[1]):
                    if M_[t, j]:
                        occ0[tuple(P_[t, j])] = (m, j)
                    if M_[t + 1, j]:
                        occ1[tuple(P_[t + 1, j])] = (m, j)
            p0 = {r: pool(c0, r) for r in (3, 1)}; p1 = {r: pool(c1, r) for r in (3, 1)}
            M = R["map"]; H = M.shape[1]
            inb = lambda w: 0 <= w[0] < H and 0 <= w[1] < H
            for rr in range(7):
                for qq in range(9):
                    if (rr, qq) == (3, 4):
                        continue
                    sr, sq = rr - true_shift[0], qq - true_shift[1]           # true source cell at t
                    if not (0 <= sr < 7 and 0 <= sq < 9) or (sr, sq) == (3, 4):   # entering cell / player's old cell
                        continue
                    w1 = (pos[t + 1][0] + rr - 3, pos[t + 1][1] + qq - 4)
                    if not inb(w1):
                        continue
                    e0, e1 = occ0.get(w1), occ1.get(w1)
                    edited = M[t][w1] != M[t + 1][w1]
                    if edited or e0 != e1:
                        label = 2                                             # something changed independently
                    elif e0 is None:
                        label = 0                                             # empty terrain, unchanged
                    else:
                        label = 1                                             # the same entity stayed in this cell
                    for shift_kind, (dr, dc) in (("est", est_shift), ("true", true_shift)):
                        ar, aq = rr - dr, qq - dc
                        if not (0 <= ar < 7 and 0 <= aq < 9):
                            continue
                        rows.append((night, shift_kind == "true",
                                     float(np.abs(c1[rr, qq] - c0[ar, aq]).mean()),
                                     float(np.abs(p1[3][rr, qq] - p0[3][ar, aq]).mean()),
                                     float(np.abs(p1[1][rr, qq] - p0[1][ar, aq]).mean()), label))
        print(f"episode {e + 1}/{len(picks)} T={T} rows={len(rows)}", flush=True)
    A = np.array(rows, float)
    res = {}
    for true_ in (0, 1):
        for night in (0, 1):
            sel = A[(A[:, 1] == true_) & (A[:, 0] == night)]
            lab = sel[:, 5]
            key = f"{'true' if true_ else 'est'}_shift|{'night' if night else 'day'}"
            res[key] = {"n_changed": int((lab == 2).sum()), "n_empty": int((lab == 0).sum()), "n_still_entity": int((lab == 1).sum())}
            for ci, name in ((2, "raw"), (3, "pool3"), (4, "pool1")):
                pos_, neg = sel[lab == 2, ci], sel[lab != 2, ci]
                thr = np.percentile(neg, 95)
                res[key][name] = {"auc": auc(pos_, neg), "auc_vs_still_entity": auc(pos_, sel[lab == 1, ci]),
                                  "recall_at_5pct_fp": float((pos_ > thr).mean())}
            print(key, res[key], flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
