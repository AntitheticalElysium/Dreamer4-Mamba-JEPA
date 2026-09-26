"""Training sufficiency: does a longer bridge phase close the u->u world's transition loss?

Replication (INTERFACE.md, 56k): U's transition loses ~5 points against its own root on zombie roots
(-0.046*, -0.055*), and does not resolve over actions_only. Phase-1 loss had plateaued (0.0854 at 5k,
0.0824 at 10k) but phase-2 dynamics loss was still falling at the end (0.239 at 1k, 0.196 at 5k, 0.161
at 9,333). So the one training quantity with visible headroom is the bridge phase.

Run: the seed-1 U recipe exactly (interface.py, SEEDS default), phase 2 extended from 9,333 to 28,000
updates (3x), same schedule (constant LR after warmup), same generators -- so its first 9,333 updates
ARE the seed-1 U run. Checkpoints at 9,333 (1x), 18,666 (2x), 28,000 (3x).

Judge: a NEW block, `observe.py collect --seed-start 57000 --target-opportunity 800 --max-seeds 1500
--out artifacts/eda/observe_fresh_v8`, collected after this commit, read once. FIT-train fit / FIT-dev
selection, frozen_ladder all-action ranking, 3 seeds, 3,000 updates, per-branch head, as interface.py:
  root_U                4 observed states + 3 past actions + candidate action (world-independent)
  gen_{1x,2x,3x}        generated state + candidate token
  state_{1x,2x,3x}      generated state alone (token-free), intact and within-root permuted (5 draws)
  trained_{1x,2x,3x}    the checkpoint's own continuation head
  ref_seed1             generated_U of the saved seed-1 world (reproducibility of 1x)
  DOWN, actions_only, tokens_attn

DECLARED READINGS (committed before collection and before training):
  primary     gen_3x - gen_1x on zombie opportunity roots: resolved > 0 -> longer_training_helps;
              resolved < 0 -> longer_training_hurts; else -> no_evidence_longer_helps
  retention   gen_3x - root_U on zombie roots not resolved < 0 -> retention_restored_at_3x, else
              transition_still_loses
  controls    gen_3x - actions_only overall resolved > 0 -> beats_actions_only_at_3x
  sanity      |gen_1x - ref_seed1| overall < 0.02 expected (same weights up to GPU nondeterminism)
Reported: 2x; trained heads; the token-free probe and its permutation drop; zombie / night / lava.
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

N, DRAWS = 17, 5
LONG = ROOT / "artifacts/eda/interface_worlds_long"
JUDGE = ROOT / "artifacts/eda/observe_fresh_v8"
AT = {"1x": 9_333, "2x": 18_666, "3x": 28_000}


def train(device, log):
    pool = torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True)
    pool_sha = json.loads((I.POOL / "pool.json").read_text())["pool_sha256"]
    LONG.mkdir(parents=True, exist_ok=True)

    def save(world, heads, history, counts, update):
        torch.save({"arm": "U", "update": update, "world": world.state_dict(), "heads": heads.state_dict(),
                    "history": list(history), "depth_counts": dict(counts), "script_sha256": _sha256(HERE / "interface.py"),
                    "longer_sha256": _sha256(Path(__file__)), "pool_sha256": pool_sha}, LONG / f"U_{update}.pt")
        log(stage="checkpoint", update=update)

    I.PHASE2_UPDATES, I.SAVE_AT, I.SAVE = AT["3x"], tuple(AT.values()), save
    I.train("U", pool, device, log)


def score(device, log):
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_heads import dev_rows
    from frozen_ladder import scores, standardize, strata, train as probe_train
    from ladder import paired
    from observability import expected_safe, load

    pool = torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True)
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    fit_seeds, unallocated = seeds_for(partition, FORK_STORE)
    dev_seeds = sorted(partition["fit_dev"]["seeds"])
    forbidden = set(partition["reserved_for_gate"]["seeds"]) | set(unallocated)
    fit, dev = load(fit_seeds)["fit"], dev_rows(dev_seeds)
    judge, manifest, files = judge_store(JUDGE)
    new = set(judge["seed"].unique().tolist())
    used = {int(f.stem.split("-")[1]) for k in range(1, 8) for f in (ROOT / f"artifacts/eda/observe_fresh_v{k}").glob("seed-*.pt")}
    if min(new) < 57_000 or new & used or new & (set(fit_seeds) | set(dev_seeds) | forbidden):
        raise SystemExit("judgement seeds are not a new, untouched block")
    encoder, config = I.load_bridge()
    worlds = {k: LONG / f"U_{v}.pt" for k, v in AT.items()} | {"ref_seed1": ROOT / "artifacts/eda/interface_worlds_v1/U.pt"}
    data = {}
    for name, path in worlds.items():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        bundle = I.world_bundle(config, encoder, device)
        bundle.world.load_state_dict(stored["world"])
        bundle.world.eval()
        heads = Heads(config).to(device)
        heads.load_state_dict(stored["heads"])
        heads.eval()
        data[name] = {s: I.branches(bundle, heads, pool["pca"], "U", encoder, d["frames"], d["actions"], device)
                      for s, d in (("fit", fit), ("dev", dev), ("judge", judge))}
        del bundle, heads
        log(stage="branches", world=name)
    with torch.no_grad():
        for d in (fit, dev, judge):
            d["tokens1"] = torch.cat([encoder._hidden(d["frames"][i:i + 64, -1:].to(device))[2].cpu()
                                      for i in range(0, len(d["frames"]), 64)])
    del encoder
    torch.cuda.empty_cache()

    pf, pd, pj, seeds = fit["p_death1"], dev["p_death1"], judge["p_death1"], judge["seed"]
    p = torch.cat([pf, pd])
    train_rows, hold_rows, judge_rows = torch.arange(len(pf)), torch.arange(len(pf), len(p)), torch.arange(len(pj))
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    safe = {"DOWN": expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
    _, opp = expected_safe(pj, pj)
    zombie = opp & strata(judge["visible"])["zombie_adjacent"]
    strat = strata(judge["visible"])
    g = torch.Generator().manual_seed(20261012)
    perms = [torch.stack([torch.randperm(N, generator=g) for _ in range(len(pj))]) for _ in range(DRAWS)]
    rows = torch.arange(len(pj))[:, None]
    per_seed = {}

    def probe(name, kind, xf, xj, permute=False):
        mean, scale = standardize(xf, train_rows)
        xf, xj = (xf.float() - mean) / scale, (xj.float() - mean) / scale
        runs, pruns = [], []
        for seed in range(3):
            model, _ = probe_train(kind, xf.shape[1:], xf, p, train_rows, hold_rows, seed=seed, device=device, steps=3000)
            runs.append(expected_safe(scores(model, xj, judge_rows, device), pj)[0])
            if permute:
                pruns.append(torch.stack([expected_safe(scores(model, xj[rows, perm], judge_rows, device), pj)[0]
                                          for perm in perms]).mean(0))
            del model
        safe[name] = torch.stack(runs).mean(0)
        per_seed[name] = [float(r[opp].mean()) for r in runs]
        if permute:
            safe[f"{name}_permuted"] = torch.stack(pruns).mean(0)
        log(arm=name, safe=round(float(safe[name][opp].mean()), 4), zombie=round(float(safe[name][zombie].mean()), 4))

    actions4 = torch.cat([fit["actions"][:, -4:], dev["actions"][:, -4:]]).flatten(1)
    probe("actions_only", "vector", actions4, judge["actions"][:, -4:].flatten(1))
    probe("tokens_attn", "tokens_attn", torch.cat([fit["tokens1"], dev["tokens1"]]), judge["tokens1"])
    ref = data["1x"]
    probe("root_U", "branch", torch.cat([ref["fit"]["root"], ref["dev"]["root"]]), ref["judge"]["root"])
    for name, d in data.items():
        probe(f"gen_{name}", "branch", torch.cat([d["fit"]["generated"], d["dev"]["generated"]]), d["judge"]["generated"])
        if name != "ref_seed1":
            probe(f"state_{name}", "branch", torch.cat([d["fit"]["generated"], d["dev"]["generated"]])[..., :192],
                  d["judge"]["generated"][..., :192], permute=True)
            safe[f"trained_{name}"] = expected_safe(d["judge"]["p_dead"], pj)[0]
    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261013)
    up = lambda r: r["difference"] is not None and r["difference"] > 0 and r["excludes_zero"]
    down = lambda r: r["difference"] is not None and r["difference"] < 0 and r["excludes_zero"]
    rules = {"primary_gen_3x_vs_1x_zombie": test("gen_3x", "gen_1x", zombie),
             "retention_3x_zombie": test("gen_3x", "root_U", zombie),
             "gen_3x_vs_actions_only": test("gen_3x", "actions_only", opp),
             "sanity_gen_1x_vs_ref_seed1": test("gen_1x", "gen_ref_seed1", opp)}
    readings = {"primary": ("longer_training_helps" if up(rules["primary_gen_3x_vs_1x_zombie"]) else
                            "longer_training_hurts" if down(rules["primary_gen_3x_vs_1x_zombie"]) else "no_evidence_longer_helps"),
                "retention": "transition_still_loses" if down(rules["retention_3x_zombie"]) else "retention_restored_at_3x",
                "controls": "beats_actions_only_at_3x" if up(rules["gen_3x_vs_actions_only"]) else "not_above_actions_only_at_3x",
                "sanity": abs(rules["sanity_gen_1x_vs_ref_seed1"]["difference"]) < 0.02}
    contrasts = {f"{a}_vs_{b}": {"overall": test(a, b, opp), "zombie": test(a, b, zombie)} for a, b in (
        ("gen_2x", "gen_1x"), ("gen_3x", "gen_2x"), ("gen_1x", "root_U"), ("gen_2x", "root_U"), ("trained_3x", "trained_1x"),
        ("state_1x", "state_1x_permuted"), ("state_3x", "state_3x_permuted"), ("state_3x", "state_1x"), ("gen_3x", "tokens_attn"),
        ("trained_3x", "DOWN"), ("state_3x", "DOWN"))}
    evidence = {"schema": "d4mj_longer_v1", "status": "SEALED: new seed block, rules committed before collection",
                "script_sha256": _sha256(Path(__file__)), "judge_manifest": manifest, "judge_seed_files": files,
                "roots": {"judge": len(pj), "opportunity": int(opp.sum()), "zombie": int(zombie.sum())},
                "training": {k: torch.load(v, map_location="cpu", weights_only=False)["history"][-1] for k, v in worlds.items()},
                "rules": rules, "readings": readings, "contrasts": contrasts, "per_seed": per_seed,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean()),
                                      **{s: float(v[m & opp].mean()) for s, m in strat.items() if (m & opp).any()}}
                                  for k, v in safe.items()}}
    (HERE / "evidence/longer.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="longer_complete", **readings)


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
