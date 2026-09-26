"""Does an ISOTROPIC patch-derived state let the transition learn the fatal tail?

Mechanism found (INTERFACE.md, transition_diag): 94% of U's fatal direction sits in PCA ranks 60-192,
which carry 5% of the action-effect energy and are predicted worst (effect R^2 0.23-0.37 vs 0.83 at the
top). Littwin et al. 2024 (arXiv 2407.03475): gradient-descent learning of linear(ized) predictors is
greedy in feature variance (lambda) -- high-variance directions are learned first; U's coordinates span
a 2,314x variance range, and the loss weights (var^-1/2, range 48x) only partly equalize it. SIGReg keeps
z isotropic (Z's per-coordinate effect R^2 is uniform, ~0.8), but z barely encodes zombies (JEPA's
ρ-greedy bias drops low-predictability features). The candidate that combines the two: the SAME patch
PCA state, whitened per component on TRAIN (u / std), so every direction has unit variance, as SIGReg
makes z; inputs and targets both whitened; the var^-1/2 rule then gives ~uniform weights.

Run: arm W = interface.py's recipe exactly (seed 1 seeds, pool, phases, alias-free bridge, 1x budget),
state W = u / std_TRAIN (std over every frame of the TRAIN main windows). Reference: the saved seed-1 U
world. Frozen-ladder probes standardize their inputs, so root probes cannot tell U from W; every probe
difference is the world's.

Judge: a NEW block, `observe.py collect --seed-start 58000 --target-opportunity 800 --max-seeds 1500
--out artifacts/eda/observe_fresh_v9`, collected after this commit, read once. FIT-train fit / FIT-dev
selection (interface.py's harness). Per arm (W, U): gen (state + token), state (token-free; intact and
within-root permuted), trained (own head); root_U, DOWN, actions_only, tokens_attn. Mechanism, per arm:
the fatal direction fitted on FIT real successors in the arm's space, its generated within-root AUC on
the judge block, and the transition's effect R^2 in U's variance bins (ranks 60-192 = "tail").

DECLARED READINGS (committed before collection and before training):
  primary     gen_W - gen_U on zombie opportunity roots: resolved > 0 -> whitening_helps;
              resolved < 0 -> whitening_hurts; else -> no_evidence_whitening_helps
  controls    gen_W - actions_only overall resolved > 0 -> W_beats_actions_only
  retention   gen_W - root_U on zombie roots not resolved < 0 -> W_no_resolved_loss
  mechanism   tail effect R^2 higher for W than U AND fatal-direction generated AUC higher for W ->
              mechanism_consistent (descriptive)
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
WHITE = ROOT / "artifacts/eda/interface_worlds_white"
JUDGE = ROOT / "artifacts/eda/observe_fresh_v9"
TAIL = (60, 192)


def whitened_pool():
    """The interface pool plus the whitened state 'w' and its weights; installs the W arm in interface."""
    pool = dict(torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True))
    main = ~pool["terminal"]
    std = pool["u"][main].reshape(-1, 192).std(0)
    pool["w"] = pool["u"] / std
    var = pool["w"][main][:, 1:].reshape(-1, 192).var(0)
    lam = var.rsqrt()
    pool["weights"] = dict(pool["weights"]) | {"W": lam / lam.mean()}
    I.KEY["W"] = "w"
    base = I.state_of.__wrapped__ if hasattr(I.state_of, "__wrapped__") else I.state_of

    def state_of(arm, pca, z, grid):
        return base("U", pca, z, grid) / std if arm == "W" else base(arm, pca, z, grid)
    state_of.__wrapped__ = base
    I.state_of = state_of
    return pool, std


def train(device, log):
    pool, std = whitened_pool()
    world, heads, history, counts = I.train("W", pool, device, log)
    WHITE.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": "W", "world": world.state_dict(), "heads": heads.state_dict(), "history": history,
                "depth_counts": counts, "std": std, "script_sha256": _sha256(HERE / "interface.py"),
                "whiten_sha256": _sha256(Path(__file__)),
                "pool_sha256": json.loads((I.POOL / "pool.json").read_text())["pool_sha256"]}, WHITE / "W.pt")
    log(status="train_complete", arm="W", depth_counts=counts,
        weight_range=float(pool["weights"]["W"].max() / pool["weights"]["W"].min()))


def score(device, log):
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from diagnose import within_auc
    from frozen_heads import dev_rows
    from frozen_ladder import scores, standardize, strata, train as probe_train
    from ladder import paired
    from observability import expected_safe, load
    from transition_diag import centre, direction
    from u_world import successors

    pool, std = whitened_pool()
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    fit_seeds, unallocated = seeds_for(partition, FORK_STORE)
    dev_seeds = sorted(partition["fit_dev"]["seeds"])
    forbidden = set(partition["reserved_for_gate"]["seeds"]) | set(unallocated)
    fit, dev = load(fit_seeds)["fit"], dev_rows(dev_seeds)
    fit["successors"] = successors(fit_seeds)["fit"][0]
    judge, manifest, files = judge_store(JUDGE)
    judge["successors"] = torch.stack([r["successors"] for f in sorted(JUDGE.glob("seed-*.pt"))
                                       for r in torch.load(f, weights_only=False)])
    new = set(judge["seed"].unique().tolist())
    used = {int(f.stem.split("-")[1]) for k in range(1, 9) for f in (ROOT / f"artifacts/eda/observe_fresh_v{k}").glob("seed-*.pt")}
    if min(new) < 58_000 or new & used or new & (set(fit_seeds) | set(dev_seeds) | forbidden):
        raise SystemExit("judgement seeds are not a new, untouched block")
    encoder, config = I.load_bridge()
    worlds = {"W": WHITE / "W.pt", "U": ROOT / "artifacts/eda/interface_worlds_v1/U.pt"}
    data, real = {}, {}
    for arm, path in worlds.items():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        bundle = I.world_bundle(config, encoder, device)
        bundle.world.load_state_dict(stored["world"])
        bundle.world.eval()
        heads = Heads(config).to(device)
        heads.load_state_dict(stored["heads"])
        heads.eval()
        data[arm] = {s: I.branches(bundle, heads, pool["pca"], arm, encoder, d["frames"], d["actions"], device)
                     for s, d in (("fit", fit), ("dev", dev), ("judge", judge))}
        del bundle, heads
        real[arm] = {}
        for s, d in (("fit", fit), ("judge", judge)):
            out = []
            for i in range(0, len(d["successors"]), 16):
                z, grid = I.encode(encoder, d["successors"][i:i + 16].flatten(0, 1)[:, None], device)
                out.append(I.state_of(arm, pool["pca"], z, grid)[:, 0].view(-1, N, 192))
            real[arm][s] = torch.cat(out)
        log(stage="branches", arm=arm)
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
    strat = strata(judge["visible"])
    zombie = opp & strat["zombie_adjacent"]
    g = torch.Generator().manual_seed(20261015)
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
    probe("root_U", "branch", torch.cat([data["U"]["fit"]["root"], data["U"]["dev"]["root"]]), data["U"]["judge"]["root"])
    for arm, d in data.items():
        probe(f"gen_{arm}", "branch", torch.cat([d["fit"]["generated"], d["dev"]["generated"]]), d["judge"]["generated"])
        probe(f"state_{arm}", "branch", torch.cat([d["fit"]["generated"], d["dev"]["generated"]])[..., :192],
              d["judge"]["generated"][..., :192], permute=True)
        safe[f"trained_{arm}"] = expected_safe(d["judge"]["p_dead"], pj)[0]

    # mechanism: fatal direction per arm, fitted on FIT real successors in the arm's own space
    var_u = pool["u"][~pool["terminal"]][:, 1:].reshape(-1, 192).var(0)
    tail = var_u.argsort(descending=True)[TAIL[0]:TAIL[1]]
    mechanism = {}
    fatal_f, fatal_j = pf > 0.5, pj > 0.5
    opp_f = fatal_f.any(1) & (~fatal_f).any(1)
    opp_j = fatal_j.any(1) & (~fatal_j).any(1)
    for arm in data:
        keep = torch.ones_like(fatal_f)
        w = direction(centre(real[arm]["fit"], keep)[opp_f].reshape(-1, 192), fatal_f[opp_f].reshape(-1).float())
        gen_j = data[arm]["judge"]["generated"][..., :192]
        rc, gc = centre(real[arm]["judge"], torch.ones_like(fatal_j)), centre(gen_j, torch.ones_like(fatal_j))
        block = {}
        for sname, mask in (("all", opp_j), ("zombie", opp_j & strat["zombie_adjacent"])):
            block[sname] = {"real_auc": within_auc(list((real[arm]["judge"] @ w)[mask]), list(fatal_j[mask])),
                            "generated_auc": within_auc(list((gen_j @ w)[mask]), list(fatal_j[mask])),
                            "effect_corr": float(torch.corrcoef(torch.stack(((gc[mask] @ w).flatten(), (rc[mask] @ w).flatten())))[0, 1])}
        r, gg = rc[opp_j].reshape(-1, 192), gc[opp_j].reshape(-1, 192)
        if arm == "W":        # tail R^2 in U's coordinates, so both arms are measured on the same axes
            r, gg = r * std, gg * std
        block["tail_effect_r2"] = float(1 - (gg[:, tail] - r[:, tail]).square().sum() / r[:, tail].square().sum())
        block["top10_effect_r2"] = float(1 - (gg[:, var_u.argsort(descending=True)[:10]] - r[:, var_u.argsort(descending=True)[:10]]).square().sum()
                                         / r[:, var_u.argsort(descending=True)[:10]].square().sum())
        mechanism[arm] = block
        log(stage="mechanism", arm=arm, gen_auc=round(block["all"]["generated_auc"], 4), tail_r2=round(block["tail_effect_r2"], 4))

    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261016)
    up = lambda r: r["difference"] is not None and r["difference"] > 0 and r["excludes_zero"]
    down = lambda r: r["difference"] is not None and r["difference"] < 0 and r["excludes_zero"]
    rules = {"primary_gen_W_vs_gen_U_zombie": test("gen_W", "gen_U", zombie),
             "gen_W_vs_actions_only": test("gen_W", "actions_only", opp),
             "retention_W_zombie": test("gen_W", "root_U", zombie)}
    readings = {"primary": ("whitening_helps" if up(rules["primary_gen_W_vs_gen_U_zombie"]) else
                            "whitening_hurts" if down(rules["primary_gen_W_vs_gen_U_zombie"]) else "no_evidence_whitening_helps"),
                "controls": "W_beats_actions_only" if up(rules["gen_W_vs_actions_only"]) else "W_not_above_actions_only",
                "retention": "W_transition_loses" if down(rules["retention_W_zombie"]) else "W_no_resolved_loss",
                "mechanism": ("mechanism_consistent" if mechanism["W"]["tail_effect_r2"] > mechanism["U"]["tail_effect_r2"]
                              and mechanism["W"]["all"]["generated_auc"] > mechanism["U"]["all"]["generated_auc"] else "mechanism_not_seen")}
    contrasts = {f"{a}_vs_{b}": {"overall": test(a, b, opp), "zombie": test(a, b, zombie)} for a, b in (
        ("gen_W", "gen_U"), ("state_W", "state_U"), ("trained_W", "trained_U"), ("state_W", "state_W_permuted"),
        ("state_W", "DOWN"), ("gen_U", "root_U"), ("gen_W", "tokens_attn"), ("trained_W", "DOWN"), ("trained_W", "actions_only"))}
    evidence = {"schema": "d4mj_whiten_v1", "status": "SEALED: new seed block, rules committed before collection",
                "script_sha256": _sha256(Path(__file__)), "judge_manifest": manifest, "judge_seed_files": files,
                "roots": {"judge": len(pj), "opportunity": int(opp.sum()), "zombie": int(zombie.sum())},
                "rules": rules, "readings": readings, "mechanism": mechanism, "contrasts": contrasts, "per_seed": per_seed,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean()),
                                      **{s: float(v[m & opp].mean()) for s, m in strat.items() if (m & opp).any()}}
                                  for k, v in safe.items()}}
    (HERE / "evidence/whiten.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="whiten_complete", **readings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "score", "smoke"))
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    os.chdir(ROOT)
    if args.command == "smoke":
        pool, std = whitened_pool()
        I.PHASE1_UPDATES, I.PHASE2_UPDATES = 20, 20
        I.train("W", pool, torch.device("cuda"), log)
        log(status="smoke_complete", std_range=float(std.max() / std.min()),
            w_var=[round(float(v), 3) for v in pool["w"][~pool["terminal"]].reshape(-1, 192).var(0)[[0, 50, 191]]])
        return 0
    (train if args.command == "train" else score)(torch.device("cuda"), log)


if __name__ == "__main__":
    raise SystemExit(main())
