"""Confirm the patch-versus-CLS boundary on new, untouched seeds.

FROZEN_LADDER.md found, on inspected roots, that within-root safe action ranking is readable from
the frozen Raw H2 encoder's patch tokens and not from its CLS token -- the only thing the world
consumes. This is the confirmation, on a seed block nothing has touched, with the arms, metric and
rules fixed here and committed before a single fresh seed is walked.

Judgement data: `observe.py collect --seed-start 51000 --target-opportunity 800 --max-seeds 1500
--out artifacts/eda/observe_fresh_v2`. 51,000+ is disjoint from every earlier range, including the
50,000-50,274 block the ladder inspected; 800 one-step opportunity roots (up from 500) for power in
the zombie stratum. The store is pinned by file hash on first read. Development: the 700 FIT seeds,
exactly as before; selection on the same inner 15% of them.

Arms, one harness (`frozen_ladder`: expected-risk pair ranking over 32-key P, AdamW 1e-3 / 1e-4,
batches of 128 roots, 3,000 updates, inner selection, three probe seeds), frozen Raw H2 checkpoint:

  pixels1       one raw frame, no actions, Nature-DQN CNN     positive control
  tokens_attn   the 81 patch tokens, position-aware attention readout (115k parameters)
  cls           CLS, the same 512-wide head as every arm      (108k)
  cls_wide      CLS, 192 -> 2048 -> 2048 -> 17                capacity control (~4.6M)
  z             projected z, the same head as cls             what the world consumes
  prior         the FIT-root action prior

Declared rules, one-step death, paired episode-seed-clustered 95% intervals on the new roots:

  R0  pixels1 - prior > 0                              else: evaluator_inadequate, read nothing else
  R1  tokens_attn - cls > 0
  R2  tokens_attn - cls_wide > 0
  R3  tokens_attn - cls_wide > 0 on zombie-adjacent roots
  R1, R2, R3 all hold          -> patch_over_cls_confirmed
  R1 and R2 hold, R3 does not  -> patch_over_cls_confirmed_aggregate_only
  otherwise                    -> not_confirmed

Two-step death, day/night and every other stratum are reported and not ruled on.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from confirm import seeds_for  # noqa: E402
from frozen_ladder import Head, materialize, scores, standardize, strata, train  # noqa: E402
from ladder import paired  # noqa: E402
from observability import FRESH, expected_safe, load  # noqa: E402

N = 17
STORE = ROOT / "artifacts/eda/observe_fresh_v2"
ARMS = {"pixels1": ("pixels1", "frames1"), "tokens_attn": ("tokens_attn", "tokens1"),
        "cls": ("vector", "cls1"), "cls_wide": ("wide", "cls1"), "z": ("vector", "z1")}


def judge_store(store=STORE):
    """The new block, in the row format observability.load gives, pinned by file hash."""
    files = sorted(Path(store).glob("seed-*.pt"))
    manifest = hashlib.sha256("".join(f"{f.name}:{_sha256(f)}" for f in files).encode()).hexdigest()
    rows = [r for f in files for r in torch.load(f, weights_only=False)]
    stack = lambda k: torch.stack([r[k] for r in rows])
    seeds = torch.tensor([int(r["seed"]) for r in rows])
    return {"seed": seeds, "frames": stack("frames"),
            "actions": nn.functional.one_hot(stack("led_to_action").clamp(max=N), N + 1).float(),
            "visible": stack("visible").float(), "p_death1": stack("p_death1").float(),
            "p_death2": stack("p_death2").float()}, manifest, len(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical/raw")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--probe-seeds", type=int, default=3)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--name", default="boundary")
    parser.add_argument("--store", type=Path, default=STORE, help="smoke only: a throwaway block")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")

    fit_seeds, _ = seeds_for(json.loads(args.partition.read_text()), FORK_STORE)
    fit = load(fit_seeds)["fit"]
    judge, manifest, files = judge_store(args.store)
    used = {int(s) for f in FRESH.glob("seed-*.pt") for s in [f.stem.split("-")[1]]}
    new = set(judge["seed"].unique().tolist())
    if min(new) < 51_000 or new & used or new & set(fit_seeds):
        raise SystemExit("judgement seeds are not a new, untouched block")
    checkpoint = args.run / "bridge/step-002000.pt"
    if _sha256(checkpoint) != json.loads((HERE / "evidence/confirm.json").read_text())["checkpoint_sha256"]:
        raise SystemExit("checkpoint differs from the one every earlier rung used")

    from d4mj.experiments import _load_bridge_parent
    bundle, heads, _ = _load_bridge_parent(checkpoint)
    bundle.encoder.freeze()
    bundle.world.eval()
    for module in bundle.world.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    for data in (fit, judge):
        data.update(materialize(bundle, data["frames"], data["actions"]))
        data["frames1"] = data["frames"][:, -1:]
    del bundle, heads
    torch.cuda.empty_cache()
    log(stage="materialized", fit=len(fit["seed"]), judge=len(judge["seed"]), judge_files=files)

    groups = fit["seed"].unique()
    held = set(groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(args.seed))
                      [:max(1, round(0.15 * len(groups)))]].tolist())
    inner = torch.tensor([int(s) in held for s in fit["seed"]])
    train_rows, hold_rows = torch.where(~inner)[0], torch.where(inner)[0]
    judge_rows = torch.arange(len(judge["seed"]))
    strat = strata(judge["visible"])
    seeds = judge["seed"]

    report = {}
    for outcome in ("death1", "death2"):
        pf, pj = fit[f"p_{outcome}"], judge[f"p_{outcome}"]
        opp_fit = (pf.amax(1) > pf.amin(1)) & ~inner
        prior = int(pf[opp_fit].mean(0).argmin())
        prior_safe, opp = expected_safe(-nn.functional.one_hot(torch.full((len(pj),), prior), N).float(), pj)
        block = {"judge_opportunity_roots": int(opp.sum()), "prior_action": prior,
                 "prior_expected_safe": float(prior_safe[opp].mean()),
                 "strata_opportunity": {k: int((v & opp).sum()) for k, v in strat.items()},
                 "prior_by_stratum": {k: float(prior_safe[v & opp].mean()) for k, v in strat.items() if (v & opp).any()},
                 "arms": {}}
        per_arm = {}
        for arm, (kind, key) in ARMS.items():
            xf, xj = fit[key], judge[key]
            if kind != "pixels1":
                mean, scale = standardize(xf, train_rows)
                xf, xj = (xf.float() - mean) / scale, (xj.float() - mean) / scale
            shape = (1,) if kind == "pixels1" else xf.shape[1:]
            runs, safes = [], []
            for probe in range(args.probe_seeds):
                model, trace = train(kind, shape, xf, pf, train_rows, hold_rows, seed=probe, device=device,
                                     steps=args.steps)
                safe, _ = expected_safe(scores(model, xj, judge_rows, device), pj)
                fsafe, fopp = expected_safe(scores(model, xf, train_rows, device), pf[train_rows])
                runs.append({"selected_step": trace["selected_step"], "inner_safe": trace["inner_safe"],
                             "parameters": trace["parameters"], "judge_expected_safe": float(safe[opp].mean()),
                             "fit_expected_safe": float(fsafe[fopp].mean())})
                safes.append(safe)
                del model
                torch.cuda.empty_cache()
            per_arm[arm] = torch.stack(safes).mean(0)
            block["arms"][arm] = {"runs": runs, "judge_expected_safe": float(per_arm[arm][opp].mean()),
                                  "vs_prior": paired(per_arm[arm][opp], prior_safe[opp], seeds[opp],
                                                     draws=args.draws, seed=args.seed + 11),
                                  "by_stratum": {k: float(per_arm[arm][v & opp].mean())
                                                 for k, v in strat.items() if (v & opp).any()}}
            log(stage="arm", outcome=outcome, arm=arm, safe=round(block["arms"][arm]["judge_expected_safe"], 4),
                params=runs[0]["parameters"])
        z = opp & strat["zombie_adjacent"]
        test = lambda a, b, mask: paired(per_arm[a][mask], (prior_safe if b == "prior" else per_arm[b])[mask],
                                         seeds[mask], draws=args.draws, seed=args.seed + 13)
        block["rules"] = {"R0_pixels1_vs_prior": test("pixels1", "prior", opp),
                          "R1_tokens_attn_vs_cls": test("tokens_attn", "cls", opp),
                          "R2_tokens_attn_vs_cls_wide": test("tokens_attn", "cls_wide", opp),
                          "R3_tokens_attn_vs_cls_wide_zombie": test("tokens_attn", "cls_wide", z)}
        block["reported"] = {"cls_wide_vs_cls": test("cls_wide", "cls", opp), "z_vs_cls": test("z", "cls", opp),
                             "tokens_attn_vs_prior": test("tokens_attn", "prior", opp),
                             "cls_wide_vs_prior": test("cls_wide", "prior", opp)}
        report[outcome] = block

    holds = {k: bool(v["difference"] > 0 and v["excludes_zero"]) for k, v in report["death1"]["rules"].items()}
    r0, r1, r2, r3 = (holds[k] for k in ("R0_pixels1_vs_prior", "R1_tokens_attn_vs_cls",
                                         "R2_tokens_attn_vs_cls_wide", "R3_tokens_attn_vs_cls_wide_zombie"))
    reading = ("evaluator_inadequate" if not r0 else "patch_over_cls_confirmed" if r1 and r2 and r3 else
               "patch_over_cls_confirmed_aggregate_only" if r1 and r2 else "not_confirmed")
    evidence = {"schema": "d4mj_boundary_v1", "status": "SEALED: new seed block, rules committed before collection",
                "script_sha256": _sha256(Path(__file__)), "frozen_ladder_sha256": _sha256(HERE / "frozen_ladder.py"),
                "checkpoint_sha256": _sha256(checkpoint), "judge_store": str(args.store), "judge_manifest": manifest,
                "judge_seeds": sorted(new), "roots": {"fit": len(fit["seed"]), "judge": len(judge["seed"])},
                "holds": holds, "reading": reading, "outcomes": report}
    (args.out / f"{args.name}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="boundary_complete", reading=reading, holds=holds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
