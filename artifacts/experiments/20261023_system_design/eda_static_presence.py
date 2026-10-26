"""A7 (2026-10-25): how much visible entity presence carries no recent motion signal?
A residual-change target (A5) only marks entities when they move. CLAIM: most visible entity time is within a few
steps of a move, so motion-grounded targets cover most entity presence. COMPETING EXPLANATION: cows idle (50% stay) and
attacking zombies stand still, so a large share of presence is static. CONTROL: steps since the entity's last world
move, per kind, over visible steps. OUTCOME: a large static share means static entities must be carried by other
signals (fovea / TSP, consequences, appearance shared with moving instances). Reads the A1 rollouts.
Usage: eda_static_presence.py A1_NPZ OUT_JSON
"""
import json, sys
import numpy as np


def main():
    D = np.load(sys.argv[1]); ep = D["episode"]; pos = D["pos"].astype(int)
    same = np.r_[False, ep[1:] == ep[:-1]]
    out = {}
    for kind in ("zombies", "cows", "skeletons"):
        rel = D[kind + "_rel"].astype(int); msk = D[kind + "_mask"]; world = rel + pos[:, None, :]
        vis = msk & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
        moved = np.zeros_like(msk)
        moved[1:] = same[1:, None] & msk[1:] & msk[:-1] & (np.abs(world[1:] - world[:-1]).sum(-1) == 1)
        since = np.zeros(msk.shape, int); c = np.zeros(msk.shape[1], int)
        for t in range(len(ep)):
            c = np.where(moved[t] | ~msk[t] | (not same[t]), 0, c + 1); since[t] = c
        v = since[vis]
        out[kind] = {"visible_steps": int(vis.sum()), "share_moved_this_step": float(moved[vis].mean()),
                     "share_static_ge4": float((v >= 4).mean()), "share_static_ge8": float((v >= 8).mean()),
                     "share_static_ge16": float((v >= 16).mean())}
        print(kind, out[kind], flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
