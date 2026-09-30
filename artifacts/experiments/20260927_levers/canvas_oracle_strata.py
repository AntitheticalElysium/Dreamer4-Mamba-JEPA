"""Paired oracle-scroll effect by recent true context scroll, TEST seeds."""
import json
from pathlib import Path

import torch
import teval as T
from scroll import estimate

HERE = Path(__file__).parent


def main():
    p = torch.load(HERE / "canvas_oracle_per_root.pt", weights_only=False)
    cache = T.build_cache("raw", torch.device("cuda"))
    ix = p["root_index"]
    ctx = cache["ctx"][ix].float()
    recent = (estimate(ctx[:, :-1], ctx[:, 1:]) != 0).sum(1)
    seeds = p["seed"]
    groups = [torch.where(seeds == seed)[0] for seed in seeds.unique()]
    rng = torch.Generator().manual_seed(20260929)
    out = {}
    for label, mask in (("none", recent == 0), ("one", recent == 1), ("two_or_three", recent >= 2)):
        rows = torch.where(mask & p["alive"][:, 15])[0]
        a = float(p["native_err"][rows, 15].mean() / p["V"])
        b = float(p["true_scroll_err"][rows, 15].mean() / p["V"])
        draws = []
        for _ in range(2000):
            chosen = torch.randint(len(groups), (len(groups),), generator=rng)
            sample = torch.cat([groups[j] for j in chosen])
            m = mask[sample] & p["alive"][sample, 15]
            if not m.any():
                continue
            x = float(p["native_err"][sample, 15][m].mean() / p["V"])
            y = float(p["true_scroll_err"][sample, 15][m].mean() / p["V"])
            draws.append(y - x)
        d = torch.tensor(draws)
        out[label] = {"alive_roots": len(rows), "walk_seeds": int(seeds[rows].unique().numel()),
                      "native": a, "oracle_scroll": b, "difference": b - a,
                      "difference_ci95": [float(q) for q in d.quantile(torch.tensor([.025, .975]))]}
    (HERE / "canvas_oracle_strata.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
