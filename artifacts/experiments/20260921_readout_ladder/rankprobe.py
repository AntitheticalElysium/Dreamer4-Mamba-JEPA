"""Objective-matched probes on the forks we already own: one- and two-step, 4- and 32-frame.

Every earlier rung in this directory was read by a probe fit on reward CE + continuation BCE and
judged on within-root action choice -- the mismatch the 2026-09-19 localization ladder measured
(`../20260919_localization_ladder/README.md`) and that produced two retracted "never constructed"
headlines. This reruns the question with THAT ladder's instrument, imported unchanged from its
`readout.py`: within-root pair ranking, softplus(score_safe - score_fatal) averaged per root, batches
of whole 17-action fans, FIT-only standardization, mlp128, three probe seeds. BCE-death is fit beside
it on identical rows as the link to the old numbers.

EXPLORATORY. The judgement roots are the 405 seeds already examined post hoc in CONFIRM and
TRANSITION, so nothing here is a sealed result and nothing emits a verdict label. A fresh sealed set
comes only after this fixes the protocol and a contrast worth sealing.

Outcomes, both from rows the fork collector already stored:
  death1  the first action kills
  death2  the first action kills, or it survives and the stored NOOP second step kills
          (`collect_broad_forks.py:191`). This is the only panel on which SLEEP's delayed hazard
          -- the 7-versus-2 zombie damage applies from the step AFTER the flag is set -- is visible.

Rungs, each at a 4- and a 32-frame root context where the context can matter:
  action_only           one-hot action alone: the action prior, rank-fitted
  root_features@ctx     the world's root readout + action: root+action, the bar to beat
  root_pixels           last root frame's CLS + 4x4 pooled patch grid + action (pixel control)
  u@ctx                 Mamba output for (root, a) -- `advanced.history`
  generated_z@ctx       the predicted successor latent
  generated_features@ctx  agent_readout(generated_z, u)
  real_features@ctx     agent_readout(z_true, u)            -- hindsight positive control
  real_z                the encoded real successor          -- hindsight positive control
  successor_pixels      real successor CLS + pooled patches -- hindsight positive control

The hindsight rungs see the realized outcome, so they are positive controls for the INSTRUMENT, not
prediction ceilings. The simulator-state ceiling needs the repeated-key replay.

Read, per rung x context x outcome x objective, on judgement roots offering both a fatal and a safe
action: safe-choice rate per probe seed, within-root AUC, selected-action histogram, fit/judge gap;
and paired, episode-seed-clustered contrasts of the per-root safe rate (mean over probe seeds)
against (a) the FIT-root action prior -- the single action with the best FIT safe rate -- and (b)
root_features at the same context, same objective. For death2, also the restricted NOOP / RIGHT /
SLEEP panel: choose among those three only, against always-RIGHT and the FIT-best of the three.
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
PRIOR = ROOT / "artifacts/experiments/20260919_localization_ladder"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PRIOR))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from confirm import seeds_for  # noqa: E402
from ladder import paired, rows_for  # noqa: E402
from readout import Fit, fit_head, one_hot_actions, scores_of, standardize, summarize  # noqa: E402

NOOP, RIGHT, SLEEP = 0, 2, 6
CONTEXTS = (4, 32)
ROOT_LEVEL = ("action_only", "root_features", "root_pixels")
CONTEXT_FREE = ("action_only", "root_pixels", "real_z", "successor_pixels")
RUNGS = ("action_only", "root_features", "root_pixels", "u", "generated_z", "generated_features",
         "real_features", "real_z", "successor_pixels")
HINDSIGHT = ("real_features", "real_z", "successor_pixels")


@torch.no_grad()
def materialize(bundle, rows, span, pixels, batch=16):
    """Root-level [R, D] and per-action [R, 17, D] features at one context, plus both outcomes."""
    world, encoder, device, count = bundle.world, bundle.encoder, bundle.device, bundle.n_actions
    keep = ("root_features", "u", "generated_z", "generated_features", "real_features")
    if pixels:
        keep += ("root_pixels", "real_z", "successor_pixels")
    out = {name: [] for name in keep}
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        n = len(chunk)
        frames = torch.stack([r["frames"][-span:] for r in chunk]).to(device)
        past = torch.stack([r["led_to_action"][-span + 1:] for r in chunk]).to(device)
        successors = torch.stack([r["successors"] for r in chunk]).to(device)
        state = world.teacher(encoder(frames), past).state
        actions = torch.arange(count, device=device).repeat(n)[:, None]
        fan = bundle.repeat_state(state, count)
        advanced, generated = bundle.advance(fan, actions)
        z_true, cls_true, grid_true = encoder.export(successors.flatten(0, 1).unsqueeze(1))
        _, real = world.observe_latent(fan, actions, z_true)
        per_action = lambda x: x.reshape(n, count, -1).cpu()
        out["root_features"].append(world.features(state)[:, -1, 0].cpu())
        out["u"].append(per_action(advanced.history))
        out["generated_z"].append(per_action(advanced.latent))
        out["generated_features"].append(per_action(generated[:, -1, 0]))
        out["real_features"].append(per_action(real[:, -1, 0]))
        if pixels:
            _, cls_root, grid_root = encoder.export(frames[:, -1:])
            out["root_pixels"].append(torch.cat((cls_root[:, -1], grid_root[:, -1].flatten(1)), -1).cpu())
            out["real_z"].append(per_action(z_true))
            out["successor_pixels"].append(per_action(torch.cat((cls_true[:, 0], grid_true[:, 0].flatten(1)), -1)))
    return {name: torch.cat(value).float() for name, value in out.items()}


def outcomes(rows):
    """death1 and death2 as [R, 17, 1] (the readout's DEATH axis), and the episode seed per root."""
    first = torch.stack([torch.as_tensor(r["terminated"]) for r in rows]).bool()
    valid = torch.stack([torch.as_tensor(r["second_valid"]) for r in rows]).bool()
    second = torch.stack([torch.as_tensor(r["second_terminated"]) for r in rows]).bool()
    two = first | (~first & valid & second)
    seed = torch.tensor([int(r["seed"]) for r in rows])
    unscored = int((~first & ~valid).sum())
    return {"death1": first.float()[..., None], "death2": two.float()[..., None]}, seed, unscored


def design(data, rung, count):
    """The [R, 17, D] rows a head reads: root-level features get the one-hot action appended."""
    if rung == "action_only":
        return one_hot_actions(len(data["root_features"]))
    if rung in ROOT_LEVEL:
        root = data[rung]
        return torch.cat((root[:, None].expand(-1, count, -1), one_hot_actions(len(root))), -1)
    return data[rung]


def per_root_safe(scores, y):
    """Per usable root, 1 when the argmin-score action survives. Usable = has fatal and safe."""
    fatal = y[..., 0].bool()
    usable = fatal.any(1) & (~fatal).any(1)
    chosen = scores.argmin(1)
    return (~fatal).gather(1, chosen[:, None]).squeeze(1).float(), usable


def panel(scores, y, prior_among):
    """NOOP / RIGHT / SLEEP only: roots where death differs among the three."""
    trio = torch.tensor([NOOP, RIGHT, SLEEP])
    fatal = y[:, trio, 0].bool()
    usable = fatal.any(1) & (~fatal).any(1)
    pick = scores[:, trio].argmin(1)
    ok = (~fatal).gather(1, pick[:, None]).squeeze(1).float()
    right = (~fatal[:, 1]).float()
    prior = (~fatal[:, prior_among]).float()
    return ok, right, prior, usable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical/raw")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--fit-roots", type=int, default=8000)
    parser.add_argument("--judge-roots", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=Fit.steps)
    parser.add_argument("--probe-seeds", type=int, default=3)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--name", default="rankprobe")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)

    from d4mj.experiments import _load_bridge_parent
    checkpoint = args.checkpoint or args.run / "bridge/step-002000.pt"
    bundle, heads, _ = _load_bridge_parent(checkpoint)
    bundle.encoder.freeze()
    bundle.world.eval()
    for module in bundle.world.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    count = bundle.n_actions
    device = bundle.device

    partition = json.loads(args.partition.read_text())
    fit_seeds, judge_seeds = seeds_for(partition, FORK_STORE)
    rows = {"fit": rows_for(fit_seeds, max(CONTEXTS), args.fit_roots),
            "judge": rows_for(judge_seeds, max(CONTEXTS), args.judge_roots)}
    labels, seeds, unscored = {}, {}, {}
    for split in rows:
        labels[split], seeds[split], unscored[split] = outcomes(rows[split])
    data = {split: {ctx: materialize(bundle, rows[split], ctx, pixels=(ctx == CONTEXTS[0]))
                    for ctx in CONTEXTS} for split in rows}
    log(stage="materialized", roots={s: len(r) for s, r in rows.items()}, unscored_second_step=unscored)
    identity = {s: hashlib.sha256(repr([(int(r["seed"]), int(r["step"])) for r in rows[s]]).encode())
                .hexdigest() for s in rows}
    del bundle, heads, rows
    torch.cuda.empty_cache()

    spec = Fit(steps=args.steps)
    report, kept = {}, {}
    for outcome in ("death1", "death2"):
        yf, yj = labels["fit"][outcome], labels["judge"][outcome]
        # The FIT-root action prior: the single action with the best safe rate on usable FIT roots.
        usable_fit = yf[..., 0].bool().any(1) & (~yf[..., 0].bool()).any(1)
        per_action = (1.0 - yf[usable_fit, :, 0]).mean(0)
        prior = int(per_action.argmax())
        trio_rate = per_action[[NOOP, RIGHT, SLEEP]]
        prior_trio = int(trio_rate.argmax())
        prior_safe, usable = per_root_safe(-torch.nn.functional.one_hot(
            torch.full((len(yj),), prior), count).float(), yj)
        report[outcome] = {"fit_prior_action": prior, "fit_prior_rates": per_action.tolist(),
                           "fit_prior_judge_safe": float(prior_safe[usable].mean()),
                           "judge_opportunity_roots": int(usable.sum()),
                           "fit_opportunity_roots": int(usable_fit.sum()), "cells": {}}
        references = {}
        for rung in RUNGS:
            for ctx in ((CONTEXTS[0],) if rung in CONTEXT_FREE else CONTEXTS):
                source_ctx = CONTEXTS[0] if rung in CONTEXT_FREE else ctx
                xf = design(data["fit"][source_ctx], rung, count)
                xj = design(data["judge"][source_ctx], rung, count)
                xf, xj = standardize(xf, xj)
                for objective in ("rank", "bce_death"):
                    key = f"{rung}@{ctx}:{objective}"
                    per_seed, judge_scores, fit_rates = [], [], []
                    for probe in range(args.probe_seeds):
                        model, curve, parameters = fit_head(
                            xf, yf, family="mlp128", objective=objective, seed=probe, spec=spec,
                            device=device)
                        sj = scores_of(model, xj, device)
                        per_seed.append(summarize(sj, yj))
                        fit_rates.append(summarize(scores_of(model, xf, device), yf).get("safe_choice_rate"))
                        judge_scores.append(sj)
                    stacked = torch.stack(judge_scores)
                    safe = torch.stack([per_root_safe(s, yj)[0] for s in stacked]).mean(0)
                    cell = {"per_seed": per_seed, "fit_safe_rate": fit_rates,
                            "judge_safe_rate_mean": float(safe[usable].mean()),
                            "parameters": parameters,
                            "vs_fit_prior": paired(safe[usable], prior_safe[usable], seeds["judge"][usable],
                                                   draws=args.draws, seed=args.seed + 11)}
                    if rung == "root_features":
                        references[(ctx, objective)] = safe
                    reference = references.get((ctx, objective))
                    if reference is not None and rung not in ("root_features", "action_only"):
                        cell["vs_root_features"] = paired(safe[usable], reference[usable],
                                                          seeds["judge"][usable], draws=args.draws,
                                                          seed=args.seed + 13)
                    if outcome == "death2":
                        oks, right, pri, u3 = zip(*[panel(s, yj, prior_trio) for s in stacked])
                        ok, u3 = torch.stack(oks).mean(0), u3[0]
                        cell["noop_right_sleep"] = {
                            "roots": int(u3.sum()), "success": float(ok[u3].mean()),
                            "always_right": float(right[0][u3].mean()),
                            "fit_best_of_three": [NOOP, RIGHT, SLEEP][prior_trio],
                            "fit_best_of_three_success": float(pri[0][u3].mean()),
                            "vs_always_right": paired(ok[u3], right[0][u3], seeds["judge"][u3],
                                                      draws=args.draws, seed=args.seed + 17)}
                    report[outcome]["cells"][key] = cell
                    kept[f"{outcome}|{key}"] = stacked
                    test = cell["vs_fit_prior"]
                    log(stage="scored", outcome=outcome, cell=key,
                        safe=round(cell["judge_safe_rate_mean"], 4),
                        vs_prior=round(test["difference"], 4), resolved=test["excludes_zero"])
                del xf, xj

    rows_path = args.out / f"{args.name}_rows.pt"
    torch.save({"labels": labels["judge"], "seed": seeds["judge"], "scores": kept}, rows_path)
    evidence = {"schema": "d4mj_rankprobe_v1", "status": "EXPLORATORY: judged on previously examined roots",
                "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                "partition_sha256": _sha256(args.partition), "script_sha256": _sha256(Path(__file__)),
                "readout_sha256": _sha256(PRIOR / "readout.py"),
                "row_identity": identity, "roots": {s: len(seeds[s]) for s in seeds},
                "unscored_second_step": unscored, "probe": {"family": "mlp128", "steps": spec.steps,
                "batch_roots": spec.batch, "lr": spec.learning_rate, "seeds": args.probe_seeds},
                "judge_rows": {"path": rows_path.name, "sha256": _sha256(rows_path)},
                "outcomes": report}
    (args.out / f"{args.name}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="rankprobe_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
