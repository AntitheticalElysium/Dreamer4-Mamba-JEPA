"""Locate neighbor-passability loss: 81 tokens -> 4x4 pooled grid -> PCA 192.

Previously inspected 55k-58k roots only. Fit on 55k+56k, test on 57k+58k.
Labels are passable move targets excluding lava (which is traversable but fatal).
The same standardized ridge procedure is used for each representation.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "artifacts/experiments/20260921_readout_ladder"),
                str(ROOT / "artifacts/experiments/20260926_diagnosis"), str(HERE)]
import interface as I
from choices import move_table
from decision import unpack, C, MOVES, LAVA
from transition_corrected import ridge_auc

STORES = {"55k": "observe_fresh_v6", "56k": "observe_fresh_v7",
          "57k": "observe_fresh_v8", "58k": "observe_fresh_v9"}


def main():
    torch.set_num_threads(6)
    pool = torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True)
    encoder, _ = I.load_bridge()
    encoder.to("cuda").eval()
    data = {}
    for block, store in STORES.items():
        meta = torch.load(ROOT / "artifacts/eda/diagnosis_dump_v1" / f"{block}_meta.pt")
        rows = [r for f in sorted((ROOT / "artifacts/eda" / store).glob("seed-*.pt"))
                for r in torch.load(f, weights_only=False)]
        rows = [r for r in rows if r["p_death1"].max() > r["p_death1"].min()]
        assert len(rows) == len(meta["visible"])
        frames = torch.stack([r["frames"][-1] for r in rows])
        grids, tokens = [], []
        with torch.no_grad():
            for i in range(0, len(frames), 64):
                _, _, t, _, _ = encoder._hidden(frames[i:i + 64, None].cuda())
                t = t.float()
                side = int(t.shape[1] ** .5)
                assert side * side == t.shape[1]
                g = torch.nn.functional.adaptive_avg_pool2d(
                    t.transpose(1, 2).reshape(len(t), t.shape[2], side, side), 4)
                grids.append(g.flatten(2).transpose(1, 2).flatten(1).cpu())
                tokens.append(t.flatten(1).cpu())
        grid = torch.cat(grids)
        u = I.project(pool["pca"], grid)
        cat, _ = move_table(meta["visible"])
        lava = torch.zeros_like(cat, dtype=torch.bool)
        for row, v in enumerate(meta["visible"]):
            tiles, _, _, _ = unpack(v)
            for a, (dr, dc) in MOVES.items():
                lava[row, a] = int(tiles[C[0] + dr, C[1] + dc]) == LAVA
        data[block] = {"grid": grid, "u": u, "tokens": torch.cat(tokens), "cat": cat, "lava": lava}
        print("encoded", block, len(rows), flush=True)
    out = {}
    for a in (1, 2, 3, 4):
        x, y = {}, {}
        for partition, blocks in (("train", ("55k", "56k")), ("test", ("57k", "58k"))):
            xs = {k: [] for k in ("grid", "u", "tokens")}
            ys = []
            for b in blocks:
                d = data[b]
                sel = (d["cat"][:, a] <= 1) & ~d["lava"][:, a]
                ys.append((d["cat"][sel, a] == 0).float())
                for k in xs:
                    xs[k].append(d[k][sel])
            x[partition] = {k: torch.cat(v) for k, v in xs.items()}
            y[partition] = torch.cat(ys)
        out[str(a)] = {"n_fit": len(y["train"]), "n_test": len(y["test"]),
                       "passable_fit": float(y["train"].mean()),
                       **{k: ridge_auc(x["train"][k], y["train"], x["test"][k], y["test"],
                                      dual=(k != "u")) for k in ("u", "grid", "tokens")}}
        print(a, out[str(a)], flush=True)
    out["mean_auc"] = {k: sum(out[str(a)][k] for a in (1, 2, 3, 4)) / 4
                       for k in ("u", "grid", "tokens")}
    (HERE / "pool_vs_pca.json").write_text(json.dumps(out, indent=2) + "\n")
    print("mean", out["mean_auc"], flush=True)


if __name__ == "__main__":
    main()
