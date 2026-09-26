"""D6. How often does the training data show a BLOCKED move? (the transition's counterfactual coverage)

A successful move scrolls the 7x9-tile map view by one 7-px tile opposite to the move; a blocked move
(solid tile, mob) leaves it in place. Test per move transition in the corpus frames, map area only (rows
0-48), player tile masked: moved if the shifted comparison matches better than the unshifted one by a
margin; blocked if the unshifted matches better; ambiguous otherwise (uniform terrain, night static).
Validated first on the observe store's roots, whose moved/blocked truth is known from the simulator
state (decision.py's rule).
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def classify(before, after, action, margin=0.25):
    """before/after uint8 [n, 63, 63, 3]; returns +1 moved, -1 blocked, 0 ambiguous."""
    a, b = before[:, :49].float(), after[:, :49].float()
    out = torch.zeros(len(a), dtype=torch.long)
    for act, (dr, dc) in MOVES.items():
        m = action == act
        if not m.any():
            continue
        x, y = a[m], b[m]
        # moved: after[r, c] = before[r + 7dr, c + 7dc]
        r0, r1 = max(0, -7 * dr), 49 - max(0, 7 * dr)
        c0, c1 = max(0, -7 * dc), 63 - max(0, 7 * dc)
        shifted = (y[:, r0:r1, c0:c1] - x[:, r0 + 7 * dr:r1 + 7 * dr, c0 + 7 * dc:c1 + 7 * dc]).abs()
        still = (y[:, r0:r1, c0:c1] - x[:, r0:r1, c0:c1]).abs()
        mask = torch.ones(r1 - r0, c1 - c0, 1)
        for rr, cc in ((21 - r0, 28 - c0), (21 - r0 - 7 * dr, 28 - c0 - 7 * dc)):   # player tile, both frames
            if 0 <= rr < r1 - r0 and 0 <= cc < c1 - c0:
                mask[max(0, rr):rr + 7, max(0, cc):cc + 7] = 0
        e_shift = (shifted * mask).flatten(1).mean(1)
        e_still = (still * mask).flatten(1).mean(1)
        rel = (e_still - e_shift) / (e_still + e_shift + 1e-6)
        out[m] = torch.where(rel > margin, 1, torch.where(rel < -margin, -1, 0))
    return out


def validate():
    from choices import move_table
    rows = [r for f in sorted((ROOT / "artifacts/eda/observe_fresh_v6").glob("seed-*.pt"))[:400]
            for r in torch.load(f, weights_only=False)]
    vis = torch.stack([r["visible"].float() for r in rows])
    cat, _ = move_table(vis)
    before, after, acts, truth = [], [], [], []
    for i, r in enumerate(rows):
        for a in MOVES:
            if cat[i, a] <= 1:
                before.append(r["frames"][-1]); after.append(r["successors"][a]); acts.append(a)
                truth.append(1 if cat[i, a] == 0 else -1)
    got = classify(torch.stack(before), torch.stack(after), torch.tensor(acts))
    truth = torch.tensor(truth)
    decided = got != 0
    return {"n": len(truth), "decided_share": float(decided.float().mean()),
            "accuracy_when_decided": float((got[decided] == truth[decided]).float().mean()),
            "true_blocked_share": float((truth == -1).float().mean()),
            "recall_blocked": float((got[truth == -1] == -1).float().mean())}


def corpus(path, shards):
    counts = {"moves": 0, "moved": 0, "blocked": 0, "ambiguous": 0}
    for s in sorted(Path(path).glob("shard-*.pt"))[:shards]:
        for e in torch.load(s, weights_only=False, mmap=True)["episodes"]:
            obs, act = e["observations"], e["actions_taken"]
            m = (act >= 1) & (act <= 4)
            idx = m.nonzero().flatten()
            for j in range(0, len(idx), 4096):
                k = idx[j:j + 4096]
                got = classify(obs[k], obs[k + 1], act[k])
                counts["moves"] += len(k)
                counts["moved"] += int((got == 1).sum()); counts["blocked"] += int((got == -1).sum())
                counts["ambiguous"] += int((got == 0).sum())
    decided = counts["moved"] + counts["blocked"]
    counts["blocked_share_of_decided"] = counts["blocked"] / max(decided, 1)
    return counts


if __name__ == "__main__":
    out = {"validation_on_simulator_truth": validate()}
    print(json.dumps(out), flush=True)
    out["expert_v1"] = corpus(ROOT / "artifacts/craftax_expert_store_v1", 6)
    print(json.dumps(out["expert_v1"]), flush=True)
    out["support_v2"] = corpus(ROOT / "artifacts/craftax_support_v2", 40)
    print(json.dumps(out["support_v2"]), flush=True)
    (HERE / "coverage.json").write_text(json.dumps(out, indent=2) + "\n")
