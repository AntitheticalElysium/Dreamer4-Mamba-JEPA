"""A6 (2026-10-25): where and how do never-seen entities first appear in the agent's view?
For the generator's birth model: SQAIR samples entering objects from a learned discovery prior; what must ours cover?
Reads the A1 rollouts (eda_signal_incentives.py, simulator state); nothing is trained.

CLAIM: first sightings of zombies happen at the view edge (they spawn 10-13 cells away and walk in), while cows and
  arrows can appear inside the view (cows spawn 4-13 away; skeletons shoot arrows from within view).
COMPETING EXPLANATION: the player's own movement brings entities in at the edge for every kind, so interior births are
  negligible. CONTROL: classify each entity's first visible step as spawned-in-view, entered-at-edge, or present at
  episode start; edge band = outermost row / column of the 7x9 view.
OUTCOME: a non-negligible interior share for a kind means the birth prior must cover interior cells (and, for arrows,
  be conditioned on other records), not only the view boundary.
Usage: eda_births.py A1_NPZ OUT_JSON
"""
import json, sys
import numpy as np

KINDS = ("zombies", "cows", "skeletons", "arrows")


def main():
    D = np.load(sys.argv[1]); ep = D["episode"]
    same = np.r_[False, ep[1:] == ep[:-1]]
    out = {}
    for kind in KINDS:
        rel = D[kind + "_rel"].astype(int); msk = D[kind + "_mask"]
        vis = msk & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
        edge = (np.abs(rel[..., 0]) == 3) | (np.abs(rel[..., 1]) == 4)
        world = rel + D["pos"].astype(int)[:, None, :]
        prev_world = np.r_[world[:1], world[:-1]]
        cont = same[:, None] & np.r_[msk[:1] & False, msk[:-1]] & msk & (np.abs(world - prev_world).sum(-1) <= 1)
        new = msk & ~cont
        counts = {"spawned_in_view": 0, "entered_at_edge": 0, "entered_interior": 0, "at_episode_start": 0, "never_seen": 0}
        spawn_dist = []
        for j in range(msk.shape[1]):
            seg = np.cumsum(new[:, j]) * msk[:, j]                  # segment id per row (0 = no entity)
            starts = np.nonzero(new[:, j])[0]                         # segment k (1-based) starts at starts[k-1]
            vrows = np.nonzero(vis[:, j] & (seg > 0))[0]
            ids, first = np.unique(seg[vrows], return_index=True)
            first_vis = dict(zip(ids.tolist(), vrows[first].tolist()))
            for k, s0 in enumerate(starts, start=1):
                if not same[s0]:
                    counts["at_episode_start"] += 1
                    continue
                spawn_dist.append(int(np.abs(rel[s0, j]).sum()))
                fv = first_vis.get(k)
                if fv is None:
                    counts["never_seen"] += 1
                elif fv == s0:
                    counts["spawned_in_view"] += 1
                elif edge[fv, j]:
                    counts["entered_at_edge"] += 1
                else:
                    counts["entered_interior"] += 1
        seen = counts["spawned_in_view"] + counts["entered_at_edge"] + counts["entered_interior"]
        counts["share_of_first_sightings_spawned_in_view"] = counts["spawned_in_view"] / max(seen, 1)
        counts["spawn_distance_quantiles"] = np.percentile(spawn_dist, [5, 50, 95]).tolist() if spawn_dist else None
        out[kind] = counts
        print(kind, counts, flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
