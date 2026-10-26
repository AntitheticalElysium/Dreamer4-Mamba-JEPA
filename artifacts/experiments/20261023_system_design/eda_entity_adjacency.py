"""A11 (2026-10-26): how often are two visible entities within one cell of each other?
Encoder-context question: a percept with a one-ring (3x3-patch) receptive field mixes the evidence of entities that
are adjacent. CLAIM: adjacent visible entity pairs are rare, so bounded context rarely mixes two entities' evidence.
COMPETING EXPLANATION: zombies converge on the player at night and arrows fly next to skeletons, so mixing is common
exactly where it matters. CONTROLS: Chebyshev distance 1 and 2; per kind; day / night; the player excluded (its cell
is always occupied). OUTCOME: the share of visible entity-steps with another visible entity within distance 1 / 2.
Reads the A1 rollouts (simulator state); nothing is trained.
Usage: eda_entity_adjacency.py A1_NPZ OUT_JSON
"""
import json, sys
import numpy as np

KINDS = ("zombies", "cows", "skeletons", "arrows")


def main():
    D = np.load(sys.argv[1]); light = D["light"].astype(np.float32)
    rels, viss, kinds = [], [], []
    for k, m in enumerate(KINDS):
        rel = D[m + "_rel"].astype(np.int16); msk = D[m + "_mask"]
        vis = msk & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)
        rels.append(rel); viss.append(vis); kinds += [k] * rel.shape[1]
    rel = np.concatenate(rels, 1); vis = np.concatenate(viss, 1); kinds = np.array(kinds)
    S = rel.shape[1]; eye = np.eye(S, dtype=bool)
    out = {}
    for name, lm in (("day", light >= 0.5), ("night", light < 0.5)):
        r, v = rel[lm], vis[lm]
        d = np.abs(r[:, :, None, :] - r[:, None, :, :]).max(-1)                # Chebyshev [N, S, S]
        both = v[:, :, None] & v[:, None, :] & ~eye[None]
        near1 = ((d <= 1) & both).any(2); near2 = ((d <= 2) & both).any(2)
        for k, m in enumerate(KINDS):
            sel = kinds == k; vv = v[:, sel]; n = int(vv.sum())
            out[f"{m}|{name}"] = {"visible_entity_steps": n,
                                  "other_within_1": float(near1[:, sel][vv].mean()) if n else None,
                                  "other_within_2": float(near2[:, sel][vv].mean()) if n else None}
            print(m, name, out[f"{m}|{name}"], flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
