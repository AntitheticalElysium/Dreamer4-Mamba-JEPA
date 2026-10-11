"""10-12 feasibility, factorization check (POST HOC, prompted by q2's breakdown: visible entropy grows ~linearly with visible mobs,
~1.4 bits each). Does the one-step innovation factorize over entities? For every root with >= 2 mobs visible at the root
(inside the 7 x 9 view), 256 keys x 17 actions, one step: each visible mob's next (position, alive) from the simulator's mob arrays
(identity = array slot). Per root-action: the sum of per-mob marginal entropies, the joint entropy of all visible mobs' tuples,
total correlation = sum - joint (0 = independent), and the visible-observation entropy (which also contains spawns and new arrows).
Plug-in entropies with K = 256 (so joint entropies near 8 bits are lower bounds).
Usage: factorize.py  -> artifacts/eda/feasibility_v1/factorize.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feasibility as FE  # noqa: E402


def H(rows):
    _, c = np.unique(rows, axis=0, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    FE.RP.env_and_render()
    roots = FE.load_roots(0)
    acts = jnp.arange(17)
    out = []
    for ri, r in enumerate(roots):
        s = FE.root_state(r)
        p = np.asarray(s.player_position)
        vis = []
        for kind in ("zombies", "cows", "skeletons"):
            mob = getattr(s, kind)
            pos, mask = np.asarray(mob.position), np.asarray(mob.mask)
            for j in np.flatnonzero(mask & (np.abs(pos[:, 0] - p[0]) <= 3) & (np.abs(pos[:, 1] - p[1]) <= 4)):
                vis.append((kind, int(j)))
        if len(vis) < 2:
            continue
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(FE.SEED + 6), ri), 256)
        obs, nxt, _, _ = FE._fork(s, keys, acts)
        obs = np.asarray(obs)
        per = [np.concatenate([np.asarray(getattr(nxt, k).position)[:, :, j], np.asarray(getattr(nxt, k).mask)[:, :, j, None]], -1)
               for k, j in vis]                                                                           # each [17, 256, 3]
        for a in range(17):
            marg = [H(m[a]) for m in per]
            joint = H(np.concatenate([m[a] for m in per], -1))
            out.append({"root": ri, "action": a, "n_mobs": len(vis), "sum_marginals": sum(marg), "joint": joint,
                        "total_correlation": sum(marg) - joint, "vis_obs": H(obs[a])})
    v = lambda k: np.array([x[k] for x in out])
    res = {"roots": len(set(x["root"] for x in out)), "root_actions": len(out),
           "sum_marginals_mean": round(float(v("sum_marginals").mean()), 3), "joint_mean": round(float(v("joint").mean()), 3),
           "total_correlation_mean": round(float(v("total_correlation").mean()), 4),
           "total_correlation_p90": round(float(np.quantile(v("total_correlation"), 0.9)), 4),
           "share_of_sum_lost_to_dependence": round(float(v("total_correlation").sum() / v("sum_marginals").sum()), 4),
           "vis_obs_minus_joint_mean": round(float((v("vis_obs") - v("joint")).mean()), 3),
           "bits_per_mob_marginal": round(float(v("sum_marginals").sum() / v("n_mobs").sum()), 3)}
    (FE.OUT / "factorize.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
