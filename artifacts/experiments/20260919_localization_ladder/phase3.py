"""Phase 3: cross the viable real-state rung with the learned dynamics.

Phase 1 showed the decision is readable from real successors once the readout is trained for
the decision rather than for six BCE targets. That result is about the ENCODER. It says nothing
about whether the world model's *predicted* successors carry the same signal, and a world model
is only useful if they do.

One structural fact constrains this whole phase: the LeWM world transitions `z` and nothing
else, so generated states exist only in z-space. Whatever the patch grid retains, the predictor
cannot emit it. That is why the generated side of the ladder has fewer rungs than the real side
and it is a finding about the architecture, not a gap in the experiment.

The plan's four-condition table, per rung, all under the rank supervision Phase 1 selected:

  real-fit  -> real   observed-state readout adequacy
  real-fit  -> gen    shared-decoder transfer / semantic compatibility
  gen-fit   -> gen    native generated-state usefulness
  gen-fit   -> real   cross-distribution transfer

plus, because recoverability after generated fitting is not sufficient:

  root+action / history+action   the matched causal controls the generated state must beat
  action_permuted                the generated fan re-labelled with a deranged action map,
                                 which leaves every magnitude intact and destroys only the
                                 action->effect correspondence
  affine_aligned                 one TRAIN-fitted linear map from generated into real space,
                                 so a poor transfer is not read as absence when a rotation
                                 would have sufficed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.m03.cache import resolve_payload
from readout import DEATH, Fit, evaluate, fit_head, one_hot_actions, standardize

SIDECAR = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/sidecar/sidecar.probe_only.pt"
RUNS = {"mamba_raw": ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/features/raw",
        "mamba_tc": ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/features/tc",
        "direct": ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/features/direct_mamba"}
SEEDS = (0, 1, 2)


def features(arm, split):
    return resolve_payload(torch.load(f"{RUNS[arm]}.{split}.pt", map_location="cpu",
                                      weights_only=False))["features"]


def derangement(n, generator):
    """A permutation with no fixed point, so every action is genuinely re-labelled."""
    while True:
        candidate = torch.randperm(n, generator=generator)
        if not bool((candidate == torch.arange(n)).any()):
            return candidate


def fit_affine(source, target):
    """Least-squares map from generated TRAIN space into real TRAIN space, with intercept."""
    a = source.reshape(-1, source.shape[-1]).double()
    b = target.reshape(-1, target.shape[-1]).double()
    a = torch.cat((a, torch.ones(len(a), 1, dtype=a.dtype)), 1)
    solution = torch.linalg.lstsq(a, b).solution
    return solution.float()


def apply_affine(x, solution):
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    flat = torch.cat((flat, torch.ones(len(flat), 1)), 1)
    return (flat @ solution).reshape(*lead, -1)


def score(name, xtr, ytr, xdv, ydv, spec, device, rows, *, family="mlp128"):
    xtr, xdv = standardize(xtr, xdv)
    got = []
    for seed in SEEDS:
        model, _, params = fit_head(xtr, ytr, family=family, objective="rank", seed=seed,
                                    spec=spec, device=device)
        got.append(evaluate(model, xdv, ydv, device))
    row = {"condition": name, "parameters": params,
           "safe_choice": [m["safe_choice"] for m in got],
           "opportunity_roots": got[0]["opportunity_roots"],
           "within_root_auc": [round(m["within_root_auc"], 4) for m in got],
           "mean_safe": float(np.mean([m["safe_choice"] for m in got]))}
    rows.append(row)
    print(json.dumps({"stage": "condition", "condition": name, "safe": row["safe_choice"],
                      "auc": row["within_root_auc"]}), flush=True)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence/phase3")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--arms", default="mamba_raw,mamba_tc,direct")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "dynamics.json"
    if destination.exists():
        print(json.dumps({"status": "cached"}), flush=True)
        return 0

    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    y = {s: side[s]["outcomes"].float() for s in ("train", "dev")}
    spec = Fit(steps=args.steps)
    report = {"schema": "d4mj_phase3_dynamics_v1", "steps": args.steps,
              "note": "the LeWM world transitions z only; generated states exist in z-space alone",
              "arms": {}}

    for arm in args.arms.split(","):
        f = {s: features(arm, s) for s in ("train", "dev")}
        real = {s: f[s]["observed_successor"].float() for s in ("train", "dev")}
        gen = {s: f[s]["generated_successor"].float() for s in ("train", "dev")}
        root = {s: f[s]["projected"].float() for s in ("train", "dev")}
        rows = []
        print(json.dumps({"stage": "arm", "arm": arm}), flush=True)

        # --- the four-condition table -------------------------------------------------------
        score("real_fit_real", real["train"], y["train"], real["dev"], y["dev"], spec, args.device, rows)
        score("real_fit_gen", real["train"], y["train"], gen["dev"], y["dev"], spec, args.device, rows)
        score("gen_fit_gen", gen["train"], y["train"], gen["dev"], y["dev"], spec, args.device, rows)
        score("gen_fit_real", gen["train"], y["train"], real["dev"], y["dev"], spec, args.device, rows)

        # --- matched causal controls the generated state must beat --------------------------
        expand = lambda v: v[:, None].expand(-1, 17, -1)
        score("root_action", torch.cat((expand(root["train"]), one_hot_actions(len(root["train"]))), -1),
              y["train"], torch.cat((expand(root["dev"]), one_hot_actions(len(root["dev"]))), -1),
              y["dev"], spec, args.device, rows)

        # --- action fidelity: same magnitudes, deranged action->effect map -------------------
        generator = torch.Generator().manual_seed(20260919)
        order = derangement(17, generator)
        score("gen_fit_gen_action_permuted", gen["train"][:, order], y["train"],
              gen["dev"][:, order], y["dev"], spec, args.device, rows)

        # --- a rotation is not absence -------------------------------------------------------
        solution = fit_affine(gen["train"], real["train"])
        score("real_fit_gen_affine_aligned", real["train"], y["train"],
              apply_affine(gen["dev"], solution), y["dev"], spec, args.device, rows)

        report["arms"][arm] = {"rows": rows}
        del f, real, gen, root

    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "phase3_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
