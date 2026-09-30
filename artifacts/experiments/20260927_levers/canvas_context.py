"""Paired fmamba/fcanvas error by recent observed scroll, on the same roots.

The only difference between the checkpoints is the canvas alignment of the
per-position temporal mixer. Context scroll is estimated from TRUE encoded
frames (one-step simulator accuracy 99.35%). Walk seed is the bootstrap unit.
"""
import json
from pathlib import Path

import torch

import teval as T
from scroll import estimate

HERE = Path(__file__).parent
A = "corrt_raw_suffix_s7_fmamba"
B = "corrt_raw_suffix_s7_fcanvas"


def metric(data, rows, moved=False):
    m = data["class"][rows] == 0 if moved else torch.ones_like(data["class"][rows], dtype=torch.bool)
    one = float(data["onestep_err"][rows][m].sum() / data["onestep_copy"][rows][m].sum())
    live = data["alive"][rows, 15]
    depth16 = float(data["gen_err"][rows, 15][live].mean() / data["V"])
    return one, depth16


def main():
    meta, _, _ = T.split()
    cache = T.build_cache("raw", torch.device("cuda"))
    ctx = cache["ctx"].float()
    recent = (estimate(ctx[:, :-1], ctx[:, 1:]) != 0).sum(1)
    da, db = (torch.load(HERE / "evals" / f"{name}_per_root.pt", weights_only=False) for name in (A, B))
    assert torch.equal(da["seed"], db["seed"]) and torch.equal(meta["seed"], da["seed"])
    seeds = da["seed"]
    groups = [torch.where(seeds == seed)[0] for seed in seeds.unique()]
    rng = torch.Generator().manual_seed(20260929)
    out = {}
    for name, mask in (("no_recent_scroll", recent == 0), ("one_recent_scroll", recent == 1),
                       ("two_or_three_recent_scrolls", recent >= 2)):
        idx = torch.where(mask)[0]
        a, b = metric(da, idx), metric(db, idx)
        diffs = []
        for _ in range(2000):
            chosen = torch.randint(len(groups), (len(groups),), generator=rng)
            resampled = torch.cat([groups[j] for j in chosen])
            rows = resampled[mask[resampled]]
            if not len(rows):
                continue
            x, y = metric(da, rows), metric(db, rows)
            diffs.append([y[j] - x[j] for j in range(2)])
        d = torch.tensor(diffs)
        out[name] = {"roots": len(idx), "walk_seeds": int(seeds[idx].unique().numel()),
                     "fmamba": {"one_step_all_x_copy": a[0], "depth16_over_V": a[1]},
                     "fcanvas": {"one_step_all_x_copy": b[0], "depth16_over_V": b[1]},
                     "fcanvas_minus_fmamba": {"one_step_all_x_copy": b[0] - a[0],
                                              "depth16_over_V": b[1] - a[1]},
                     "ci95": {"one_step_all_x_copy": [float(q) for q in d[:, 0].quantile(torch.tensor([.025,.975]))],
                              "depth16_over_V": [float(q) for q in d[:, 1].quantile(torch.tensor([.025,.975]))]}}
    (HERE / "canvas_context.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
