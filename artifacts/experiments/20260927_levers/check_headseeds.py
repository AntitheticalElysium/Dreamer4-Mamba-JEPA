"""Decision check (2026-10-02): is the budget worlds' gen1 regression next to zombies (dpanel 36k - 18k: -0.034 [-0.051, -0.017]
s7, -0.036 [-0.053, -0.019] s8) larger than HEAD-FITTING noise? dpanel averages 3 head seeds per root and bootstraps roots only,
so head-seed variance is outside its interval; its per-seed overall gen1 spreads are 0.008-0.045, as large as the effect.
check_branches found the 36k imagined branches near zombies BETTER aligned with the real ones (0.928 -> 0.939, 0.927 -> 0.939)
with lower error -- the imagined successors do not explain a regression.
dpanel's own data (opened blocks v6 + v7), imagination and fit_head, K = 8 gen1 head seeds per world (seeds 0-7; dpanel used
0-2), per-seed per-root expected safe saved (evals/headseeds/<world>.pt). Two-level bootstrap of (36k - 18k) mean safe on H1
opportunity roots, overall and zombie-adjacent: 4,000 draws resampling head seeds (with replacement, per world) and episode-seed
clusters of roots (shared across the pair).
Reading, declared before running:
  zombie_regression_real   the two-level 95% interval of the zombie-adjacent difference excludes 0 below at BOTH seeds
  within_head_noise        it includes 0 at either seed (dpanel's -0.034 / -0.036 were head-fitting noise plus roots)
Usage: check_headseeds.py <world.pt> ...   (then --analyse s7_18k:s7_36k ...)
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import dpanel as D  # noqa: E402
T = D.T
K = 8
OUT = Path("artifacts/experiments/20260927_levers/evals/headseeds")


def fit_all(paths):
    from d4mj.config import config_from_dict
    from confirm import seeds_for
    from d4mj.lewm_diagnostics import FORK_STORE
    from observability import expected_safe
    from frozen_ladder import strata
    import spatial as S
    device = torch.device("cuda")
    log = lambda **kw: print(json.dumps(kw), flush=True)
    partition = json.loads((D.LADDER / "evidence/root_partition.json").read_text())
    fit_seeds, _ = seeds_for(partition, FORK_STORE)
    dev_seeds = sorted(partition["fit_dev"]["seeds"])
    data = D.token_cache({"fit": (fit_seeds, None), "dev": (dev_seeds, None),
                          "judge": (None, ["observe_fresh_v6", "observe_fresh_v7"])}, device, log)
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    judge = data["judge"]
    rows_j = torch.arange(len(judge["seed"]))
    strat = strata(judge["visible"])
    OUT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        world, st = T.load_world(Path(path), device)
        name = st["name"]
        if (OUT / f"{name}.pt").exists():
            log(world=name, status="exists")
            continue
        tag = D.CACHE / f"hs_{name}"
        gen1 = {split: D.imagine(world, config, data[split], 5, f"{tag}_{split}", device)[0] for split in ("fit", "dev", "judge")}
        runs = []
        for seed in range(K):
            model, trace = D.fit_head(gen1["fit"], data["fit"]["p1"], gen1["dev"], data["dev"]["p1"], device, seed,
                                      Path(f"{tag}_gen1_s{seed}.pt"))
            runs.append(expected_safe(D.judge_risk(model, gen1["judge"], rows_j, device), judge["p1"])[0])
            log(world=name, seed=seed, dev_safe=trace["dev_safe"])
        opp = judge["p1"].amax(1) > judge["p1"].amin(1)
        torch.save({"seed": judge["seed"], "safe": torch.stack(runs), "opp": opp, "zombie": strat["zombie_adjacent"]}, OUT / f"{name}.pt")
        for f in D.CACHE.glob(f"hs_{name}_*"):
            f.unlink()
        del world
        torch.cuda.empty_cache()


def analyse(pairs):
    res = {}
    g = torch.Generator().manual_seed(20261002)
    for pair in pairs:
        a_name, b_name = pair.split(":")
        a, b = torch.load(OUT / f"{a_name}.pt"), torch.load(OUT / f"{b_name}.pt")
        assert torch.equal(a["seed"], b["seed"])
        r = {}
        for stratum, m in (("overall", a["opp"]), ("zombie_adjacent", a["opp"] & a["zombie"])):
            sa, sb = a["safe"][:, m], b["safe"][:, m]                      # [K, roots]
            seeds = a["seed"][m]
            groups = [torch.where(seeds == s)[0] for s in seeds.unique()]
            draws = []
            for _ in range(4000):
                ka = torch.randint(K, (K,), generator=g); kb = torch.randint(K, (K,), generator=g)
                pick = torch.cat([groups[i] for i in torch.randint(len(groups), (len(groups),), generator=g)])
                draws.append(float(sb[kb][:, pick].mean() - sa[ka][:, pick].mean()))
            lo, hi = torch.tensor(draws).quantile(torch.tensor([0.025, 0.975])).tolist()
            r[stratum] = {"roots": int(m.sum()), "per_seed_a": [round(float(x), 4) for x in sa.mean(1)],
                          "per_seed_b": [round(float(x), 4) for x in sb.mean(1)],
                          "difference": float(sb.mean() - sa.mean()), "interval": [lo, hi], "excludes_zero": lo > 0 or hi < 0}
        res[pair] = r
        print(json.dumps({pair: r}), flush=True)
    z = [res[p]["zombie_adjacent"] for p in res]
    res["readings"] = {"zombie_regression_real": all(x["interval"][1] < 0 for x in z),
                       "within_head_noise": any(x["interval"][0] <= 0 <= x["interval"][1] for x in z)}
    (OUT / "analysis.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["readings"]))


if __name__ == "__main__":
    if sys.argv[1] == "--analyse":
        analyse(sys.argv[2:])
    else:
        fit_all(sys.argv[1:])
