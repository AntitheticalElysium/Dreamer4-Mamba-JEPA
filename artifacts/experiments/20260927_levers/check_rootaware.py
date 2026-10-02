"""Item 1 check (2026-10-03): does the 36k gen1 regression next to zombies disappear when the head can see WHERE THE ZOMBIE WAS?
(check_zombie_cue: the 36k worlds draw stay-when-attacking / chase-otherwise like reality, stayed cue 0.67 / 0.68 vs real 0.62; the
18k worlds' successors carried a successor-only shortcut, beside cue 0.47 / 0.48 vs real 0.07. dpanel's gen head reads each branch's
successor ALONE; check_headseeds: 36k - 18k gen1 zombie-adjacent -0.030 [-0.052, -0.008] s7, -0.035 [-0.060, -0.013] s8.)
dpanel's data and protocol (opened blocks v6 + v7; fit / dev / judge; step-1 imagination as dpanel; fit_head, 8 head seeds as
check_headseeds), with a ROOT-AWARE input per branch: the imagined successor's tokens concatenated per cell with the root's tokens
aligned by the world's OWN imagined view shift (scroll.estimate(root, successor); map cells shifted, out-of-view cells zero, root HUD):
[81, 384]. Nothing real enters the input. Saved per root and head seed (evals/rootaware/<world>.pt); the real successors with the
same construction (real view shift) as a reference, 3 head seeds.
Readings, declared before running:
  rootaware_regression_gone   the root-aware zombie-adjacent (36k - 18k) two-level 95% interval (check_headseeds.analyse) does not
                              lie below 0 at either seed
  rootaware_gain_36k          per seed, (root-aware - successor-only) zombie-adjacent mean is larger for the 36k world than for the
                              18k world (reported)
Usage: check_rootaware.py <world.pt> ...   then   check_rootaware.py --analyse s7_18k:s7_36k ...
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_headseeds as HS  # noqa: E402
import dpanel as D  # noqa: E402
T, N, K = D.T, D.N, HS.K
OUT = Path("artifacts/experiments/20260927_levers/evals/rootaware")


def align(root, succ):
    """root [B,81,192], succ [B,17,81,192] -> [B,17,81,384]: succ ++ root shifted by the view shift from root to succ."""
    from scroll import SHIFTS, estimate
    B = len(root)
    r = root.float()[:, None].expand(B, N, 81, 192).clone()
    s = estimate(r.flatten(0, 1), succ.float().flatten(0, 1)).view(B, N)
    m = r[:, :, :63].unflatten(2, (7, 9))
    out = torch.zeros_like(m)
    for i, (dr, dc) in enumerate(SHIFTS):
        sel = s == i
        rs, re_, cs, ce = max(0, -dr), 7 - max(0, dr), max(0, -dc), 9 - max(0, dc)
        if sel.any() and rs < re_ and cs < ce:
            out[:, :, rs:re_, cs:ce][sel] = m[:, :, rs + dr:re_ + dr, cs + dc:ce + dc][sel]
    r[:, :, :63] = out.flatten(2, 3)
    return torch.cat([succ.float(), r], -1).half()


def build(world, config, data, path, device):
    """[R,17,81,384] fp16 memmap of root-aware inputs (world None = the real successors)."""
    R = len(data["seed"])
    mm = D.memmap(path, (R, N, 81, 384), "w+")
    with torch.no_grad():
        for i in range(0, R, 16):
            idx = torch.arange(i, min(i + 16, R)); b = len(idx)
            if world is None:
                succ = data["real1"][idx].float()
            else:
                fan = data["ctx"][idx].float().repeat_interleave(N, 0)
                past = data["acts"][idx, 1:].repeat_interleave(N, 0)
                a = torch.arange(N).repeat(b)[:, None]
                succ = T.step(world, fan, torch.cat([past, a], 1), device, config).view(b, N, 81, 192).cpu()
            mm[i:i + b] = align(data["ctx"][idx, -1], succ).numpy()
    mm.flush()
    return torch.from_numpy(D.memmap(path, (R, N, 81, 384)))


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
    opp = judge["p1"].amax(1) > judge["p1"].amin(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in [None] + list(paths):
        world, name = None, "real"
        if path is not None:
            world, st = T.load_world(Path(path), device)
            name = st["name"]
        if (OUT / f"{name}.pt").exists():
            log(world=name, status="exists")
            continue
        tag = D.CACHE / f"ra_{name}"
        x = {split: build(world, config, data[split], f"{tag}_{split}.f16", device) for split in ("fit", "dev", "judge")}
        runs = []
        for seed in range(3 if world is None else K):
            model, trace = D.fit_head(x["fit"], data["fit"]["p1"], x["dev"], data["dev"]["p1"], device, seed, Path(f"{tag}_s{seed}.pt"))
            runs.append(expected_safe(D.judge_risk(model, x["judge"], rows_j, device), judge["p1"])[0])
            log(world=name, seed=seed, dev_safe=trace["dev_safe"], judge_zombie=round(float(runs[-1][opp & strat["zombie_adjacent"]].mean()), 4))
        torch.save({"seed": judge["seed"], "safe": torch.stack(runs), "opp": opp, "zombie": strat["zombie_adjacent"]}, OUT / f"{name}.pt")
        for f in D.CACHE.glob(f"ra_{name}_*"):
            f.unlink()
        del world
        torch.cuda.empty_cache()


def analyse(pairs):
    HS.OUT = OUT
    HS.analyse(pairs)                                                    # writes evals/rootaware/analysis.json, two-level bootstrap
    res = json.loads((OUT / "analysis.json").read_text())
    gains = {}
    for name in sorted({x for p in pairs for x in p.split(":")}):
        a, b = torch.load(OUT / f"{name}.pt"), torch.load(Path("artifacts/experiments/20260927_levers/evals/headseeds") / f"{name}.pt")
        m = a["opp"] & a["zombie"]
        gains[name] = {"rootaware": float(a["safe"][:, m].mean()), "successor_only": float(b["safe"][:, m].mean()),
                       "gain": float(a["safe"][:, m].mean() - b["safe"][:, m].mean())}
    real = torch.load(OUT / "real.pt")
    res["zombie_adjacent_means"] = gains | {"real": {"rootaware": float(real["safe"][:, real["opp"] & real["zombie"]].mean())}}
    z = [res[p]["zombie_adjacent"] for p in pairs]
    res["readings"] = {"rootaware_regression_gone": all(x["interval"][1] >= 0 for x in z),
                       "rootaware_gain_36k": {p: gains[p.split(":")[1]]["gain"] > gains[p.split(":")[0]]["gain"] for p in pairs}}
    (OUT / "analysis.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["readings"] | {"means": res["zombie_adjacent_means"]}))


if __name__ == "__main__":
    if sys.argv[1] == "--analyse":
        analyse(sys.argv[2:])
    else:
        fit_all(sys.argv[1:])
