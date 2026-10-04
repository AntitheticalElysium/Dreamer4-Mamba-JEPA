"""Sealed replication of the whitened world's TRAINED system, with a second world seed.

whiten.py (sealed 58k) REPORTED, and whiten_blocks.py confirmed post hoc on 55k/56k/57k: the whitened
patch-state world's own trained continuation head beats DOWN overall in all four blocks (+0.065 to
+0.085, each resolved), beats U's trained head on zombie roots in all four (resolved), and is above
DOWN on zombie roots in all four (+0.039 to +0.053, none resolved alone). One W seed.

Run: W seed 2 = interface.py's recipe on the whitened pool (whiten.whitened_pool) with replicate.py's
seed-2 seeds (init 8, phase-1 order 12, heads +3, phase-2 order 18, depths 14). Seed-1 W and both U
worlds are reused, unchanged.
Judge: a NEW block, `observe.py collect --seed-start 59000 --target-opportunity 800 --max-seeds 1500
--out artifacts/eda/observe_fresh_v10`, collected after this commit, read once. Each world's OWN
trained continuation head on its imagined successors (4 observed frames, one advance per action) --
nothing fitted except the actions_only control (frozen_ladder harness, FIT-train fit / FIT-dev
selection, 3 seeds). DOWN = the FIT action prior.

DECLARED READINGS (committed before collection and before seed-2 training):
  primary    W = the per-root mean of trained_W_s1 and trained_W_s2 (the recipe's expected performance)
             trained_system_passes iff W - DOWN overall, W - actions_only overall, and W - DOWN on
             zombie-adjacent roots are all resolved > 0 (paired seed-clustered 95%)
  secondary  W - U (per-root mean of trained_U_s1 and trained_U_s2) on zombie roots resolved > 0
             -> whitening_helps_trained_system
Reported: each seed alone against the same controls; night and lava strata; SLEEP choices.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

import interface as I  # noqa: E402

N = 17
JUDGE = ROOT / "artifacts/eda/observe_fresh_v10"
WHITE = ROOT / "artifacts/eda/interface_worlds_white"
WORLDS = {"W_s1": ("W", WHITE / "W.pt"), "W_s2": ("W", WHITE / "W_s2.pt"),
          "U_s1": ("U", ROOT / "artifacts/eda/interface_worlds_v1/U.pt"), "U_s2": ("U", ROOT / "artifacts/eda/interface_worlds_v2/U.pt")}


def train(device, log):
    from replicate import SEED2
    from whiten import whitened_pool
    pool, std = whitened_pool()
    I.SEEDS = dict(SEED2)
    world, heads, history, counts = I.train("W", pool, device, log)
    torch.save({"arm": "W", "world": world.state_dict(), "heads": heads.state_dict(), "history": history,
                "depth_counts": counts, "seeds": I.SEEDS, "std": std, "script_sha256": _sha256(HERE / "interface.py"),
                "whiten_rep_sha256": _sha256(Path(__file__)),
                "pool_sha256": json.loads((I.POOL / "pool.json").read_text())["pool_sha256"]}, WHITE / "W_s2.pt")
    log(status="train_complete", arm="W_s2", depth_counts=counts)


def score(device, log):
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_heads import dev_rows
    from frozen_ladder import scores, standardize, strata, train as probe_train
    from ladder import paired
    from observability import expected_safe, load
    from whiten import whitened_pool

    pool, _ = whitened_pool()
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    fit_seeds, unallocated = seeds_for(partition, FORK_STORE)
    dev_seeds = sorted(partition["fit_dev"]["seeds"])
    forbidden = set(partition["reserved_for_gate"]["seeds"]) | set(unallocated)
    fit, dev = load(fit_seeds)["fit"], dev_rows(dev_seeds)
    judge, manifest, files = judge_store(JUDGE)
    new = set(judge["seed"].unique().tolist())
    used = {int(f.stem.split("-")[1]) for k in range(1, 10) for f in (ROOT / f"artifacts/eda/observe_fresh_v{k}").glob("seed-*.pt")}
    if min(new) < 59_000 or new & used or new & (set(fit_seeds) | set(dev_seeds) | forbidden):
        raise SystemExit("judgement seeds are not a new, untouched block")
    pf, pd, pj, seeds = fit["p_death1"], dev["p_death1"], judge["p_death1"], judge["seed"]
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    safe = {"DOWN": expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
    _, opp = expected_safe(pj, pj)
    strat = strata(judge["visible"])
    zombie = opp & strat["zombie_adjacent"]
    p = torch.cat([pf, pd])
    train_rows, hold_rows = torch.arange(len(pf)), torch.arange(len(pf), len(p))
    actions4 = torch.cat([fit["actions"][:, -4:], dev["actions"][:, -4:]]).flatten(1)
    mean, scale = standardize(actions4, train_rows)
    runs = []
    for seed in range(3):
        model, _ = probe_train("vector", actions4.shape[1:], (actions4 - mean) / scale, p, train_rows, hold_rows,
                               seed=seed, device=device, steps=3000)
        runs.append(expected_safe(scores(model, (judge["actions"][:, -4:].flatten(1) - mean) / scale,
                                         torch.arange(len(pj)), device), pj)[0])
    safe["actions_only"] = torch.stack(runs).mean(0)
    encoder, config = I.load_bridge()
    sleep = {}
    for name, (arm, path) in WORLDS.items():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        b = I.world_bundle(config, encoder, device)
        b.world.load_state_dict(stored["world"])
        b.world.eval()
        h = Heads(config).to(device)
        h.load_state_dict(stored["heads"])
        h.eval()
        risk = I.branches(b, h, pool["pca"], arm, encoder, judge["frames"], judge["actions"], device)["p_dead"]
        safe[name] = expected_safe(risk, pj)[0]
        sleep[name] = int((risk[opp].argmin(1) == 6).sum())
        del b, h
        log(world=name, safe=round(float(safe[name][opp].mean()), 4), zombie=round(float(safe[name][zombie].mean()), 4))
    safe["W"] = (safe["W_s1"] + safe["W_s2"]) / 2
    safe["U"] = (safe["U_s1"] + safe["U_s2"]) / 2
    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261024)
    up = lambda r: r["difference"] > 0 and r["excludes_zero"]
    rules = {"W_vs_DOWN": test("W", "DOWN", opp), "W_vs_actions_only": test("W", "actions_only", opp),
             "W_vs_DOWN_zombie": test("W", "DOWN", zombie), "W_vs_U_zombie": test("W", "U", zombie)}
    readings = {"primary": ("trained_system_passes" if up(rules["W_vs_DOWN"]) and up(rules["W_vs_actions_only"])
                            and up(rules["W_vs_DOWN_zombie"]) else "trained_system_fails"),
                "secondary": "whitening_helps_trained_system" if up(rules["W_vs_U_zombie"]) else "no_evidence"}
    reported = {f"{a}_vs_{b}": {"overall": test(a, b, opp), "zombie": test(a, b, zombie), "night": test(a, b, opp & strat["night"])}
                for a in ("W_s1", "W_s2", "W") for b in ("DOWN", "actions_only", "U")}
    evidence = {"schema": "d4mj_whiten_rep_v1", "status": "SEALED: new seed block, rules committed before collection",
                "script_sha256": _sha256(Path(__file__)), "judge_manifest": manifest, "judge_seed_files": files,
                "roots": {"judge": len(pj), "opportunity": int(opp.sum()), "zombie": int(zombie.sum())},
                "rules": rules, "readings": readings, "reported": reported, "sleep_choices": sleep,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean()),
                                      **{s: float(v[m & opp].mean()) for s, m in strat.items() if (m & opp).any()}}
                                  for k, v in safe.items()}}
    (HERE / "evidence/whiten_rep.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="whiten_rep_complete", **readings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "score"))
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    os.chdir(ROOT)
    (train if args.command == "train" else score)(torch.device("cuda"), log)


if __name__ == "__main__":
    raise SystemExit(main())
