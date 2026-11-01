"""A4 (2026-10-25): can correspondence targets be read off Craftax-Classic pixels without labels?
Correspondence (where each cell's content went between consecutive frames: camera shift + entity moves + changes) is an
agreed candidate perception objective, described as pixel-derived. This measures whether plain pixel matching recovers
it, by day and at night. Simulator state is used for evaluation only; nothing is trained.

CLAIM: block matching of consecutive stored frames recovers (a) the camera shift and (b) each visible entity's screen
  displacement nearly always by day, so correspondence targets are cheap and label-free.
COMPETING EXPLANATIONS: (a) night static corrupts matching; (b) repeated terrain (grass) makes many candidate matches
  equally good (aperture problem), so the target must be a distribution, not one displacement; (c) identical sprites of
  one kind create ambiguous entity matches; (d) an entity moving onto different terrain changes its patch.
CONTROLS: true player and entity displacements from replayed state; day / night strata; ambiguity = best and
  second-best candidate within 5% of the best cost; terrain cells and entity cells counted separately.
OUTCOMES: camera shift and entity displacement recovered by plain matching at night as by day -> correspondence targets
  can be computed from pixels; failure at night or frequent ambiguity -> targets must be learned or distributional.
  This says nothing about how a learned model would use the targets.
Usage (project venv): eda_correspondence.py OUT_JSON N_EPISODES STRIDE
"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP

KINDS = ("zombies", "cows", "skeletons", "arrows")
SHIFTS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]                 # content shift in cells (row, col)


def scan_fn():
    env, params, _, _, _ = RP.env_and_render()
    rec = lambda s: {"pos": s.player_position, "light": s.light_level,
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


def cells(frame):
    """[63, 63, 3] uint8 -> map cells [7, 9, 7, 7, 3] float."""
    return frame[:49].astype(np.float32).reshape(7, 7, 9, 7, 3).transpose(0, 2, 1, 3, 4)


def ambiguous(costs):
    s = np.sort(costs)
    return len(s) > 1 and s[1] <= s[0] * 1.05 + 1e-6


def main():
    out_path, n_eps, stride = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(20261025)
    man = RP.manifest(); picks = []
    for sh in rng.permutation(len(man["shards"])).tolist():
        eps_ = RP._shard(sh)["episodes"]
        for i in rng.permutation([i for i, e in enumerate(eps_) if e["split"] == "train" and len(e["actions_taken"]) >= 64])[:2].tolist():
            picks.append((sh, int(i)))
        if len(picks) >= n_eps:
            break
    run = scan_fn()
    cam, ent, ter = [], [], []
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
            true_shift = tuple(-(pos[t + 1] - pos[t]))                  # content moves opposite to the player
            # (a) camera shift: mean |diff| over the overlap of the 7x9 cell grid for each candidate content shift
            cost = []
            for dr, dc in SHIFTS:
                r0, r1 = max(0, -dr), min(7, 7 - dr); q0, q1 = max(0, -dc), min(9, 9 - dc)
                a = c0[r0:r1, q0:q1]; bb = c1[r0 + dr:r1 + dr, q0 + dc:q1 + dc]
                cost.append(np.abs(a - bb).mean())
            cost = np.array(cost); best = SHIFTS[int(np.argmin(cost))]
            cam.append({"night": night, "moved": true_shift != (0, 0), "ok": best == true_shift, "amb": ambiguous(cost)})
            # (b) entities visible at t and t+1 (same slot, moved <= 1 cell in the world)
            occupied = set()
            for m in KINDS:
                P_, M_ = R[f"{m}_p"].astype(int), R[f"{m}_m"]
                for j in range(P_.shape[1]):
                    if not M_[t, j]:
                        continue
                    r0_ = P_[t, j] - pos[t]
                    if abs(r0_[0]) <= 3 and abs(r0_[1]) <= 4:
                        occupied.add((r0_[0] + 3, r0_[1] + 4))
                    if not (M_[t + 1, j] and np.abs(P_[t + 1, j] - P_[t, j]).sum() <= 1):
                        continue
                    r1_ = P_[t + 1, j] - pos[t + 1]
                    if not (abs(r0_[0]) <= 3 and abs(r0_[1]) <= 4 and abs(r1_[0]) <= 3 and abs(r1_[1]) <= 4):
                        continue
                    src = c0[r0_[0] + 3, r0_[1] + 4]
                    cand, cc = [], []
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            rr, qq = r0_[0] + 3 + dr, r0_[1] + 4 + dc
                            if 0 <= rr < 7 and 0 <= qq < 9:
                                cand.append((dr, dc)); cc.append(np.abs(src - c1[rr, qq]).mean())
                    cc = np.array(cc); got = cand[int(np.argmin(cc))]
                    true_d = tuple(r1_ - r0_)
                    ent.append({"kind": m, "night": night, "moved": true_d != (0, 0), "ok": got == true_d, "amb": ambiguous(cc)})
            # (c) terrain cells: unique best match equal to the true content shift, within the 3x3 neighbourhood
            for rr in range(7):
                for qq in range(9):
                    if (rr, qq) in occupied or (rr, qq) == (3, 4):
                        continue
                    tr, tq = rr + true_shift[0], qq + true_shift[1]
                    if not (0 <= tr < 7 and 0 <= tq < 9):
                        continue
                    cand, cc = [], []
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            r2, q2 = rr + dr, qq + dc
                            if 0 <= r2 < 7 and 0 <= q2 < 9:
                                cand.append((dr, dc)); cc.append(np.abs(c0[rr, qq] - c1[r2, q2]).mean())
                    cc = np.array(cc); got = cand[int(np.argmin(cc))]
                    ter.append((night, got == true_shift, ambiguous(cc)))
        print(f"episode {e + 1}/{len(picks)} T={T}", flush=True)
    ter = np.array(ter, bool)
    res = {}
    for name, rows in (("camera", cam), ("entities", ent)):
        for night in (False, True):
            for moved in (False, True):
                sel = [r for r in rows if r["night"] == night and r["moved"] == moved]
                if sel:
                    res[f"{name}|{'night' if night else 'day'}|{'moved' if moved else 'still'}"] = {
                        "n": len(sel), "ok": float(np.mean([r["ok"] for r in sel])), "ambiguous": float(np.mean([r["amb"] for r in sel]))}
    for kind in KINDS:
        for night in (False, True):
            sel = [r for r in ent if r["kind"] == kind and r["night"] == night]
            if sel:
                res[f"entities_{kind}|{'night' if night else 'day'}"] = {"n": len(sel), "ok": float(np.mean([r["ok"] for r in sel])),
                                                                          "ambiguous": float(np.mean([r["amb"] for r in sel]))}
    for night in (False, True):
        sel = ter[ter[:, 0] == night]
        if len(sel):
            res[f"terrain|{'night' if night else 'day'}"] = {"n": int(len(sel)), "ok": float(sel[:, 1].mean()), "ambiguous": float(sel[:, 2].mean())}
    for k, v in res.items():
        print(k, v)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
