"""Phase 3b: the two gaps my first Phase 3 pass left open.

The plan requires more than a z-only verdict on the dynamics:

  "Include native-fit h and [generated z,h]; the final state interface includes memory,
   and a z-only score need not exhaust its capability."

If the recurrent state carries action-conditional information that `z` alone does not, then
"the predictor is not action-conditional" is overstated -- what a planner reads at deployment
is the whole state interface, not z. So h, and [generated z, h], are scored here under the same
supervision and the same derangement test.

  "Evaluate the existing z->z and u->u worlds before training replacements. Obtain real-u and
   generated-u safe choice with both native-fit and transfer-fit readouts."

That lineage matters directly: its `u` is a TRAIN-fitted PCA-192 of the 4x4 pooled patch grid --
a sibling of the `pooled4_pca192` rung that reached 36/36 here. It is the one existing world
model already trained on a spatial export, so it answers whether a spatial export SURVIVES the
dynamics, which is the question findings 2 and 3 raise together.

That lineage is an unsealed diagnostic (patched loader, primary-only, frozen consecutive-TC
encoder shared by both arms, MSE-only worlds). It is labelled as such and is NOT a Raw/TC
comparison.
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
from readout import Fit, evaluate, fit_head, one_hot_actions, standardize

SIDECAR = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/sidecar/sidecar.probe_only.pt"
MEMORY = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/memory/features"
MATCHED = ROOT / "artifacts/lewm_gates_20260918/m03_matched/features"
SEEDS = (0, 1, 2)


def payload(path):
    return resolve_payload(torch.load(path, map_location="cpu", weights_only=False))["features"]


def derangement(n, generator):
    while True:
        candidate = torch.randperm(n, generator=generator)
        if not bool((candidate == torch.arange(n)).any()):
            return candidate


def score(name, xtr, ytr, xdv, ydv, spec, device, rows, family="mlp128"):
    xtr, xdv = standardize(xtr, xdv)
    got = []
    for seed in SEEDS:
        model, _, params = fit_head(xtr, ytr, family=family, objective="rank", seed=seed,
                                    spec=spec, device=device)
        got.append(evaluate(model, xdv, ydv, device))
    row = {"condition": name, "parameters": params, "dim": int(xtr.shape[-1]),
           "safe_choice": [m["safe_choice"] for m in got],
           "mean_safe": round(float(np.mean([m["safe_choice"] for m in got])), 1),
           "opportunity_roots": got[0]["opportunity_roots"],
           "within_root_auc": [round(m["within_root_auc"], 4) for m in got]}
    rows.append(row)
    print(json.dumps({"stage": "cond", "condition": name, "dim": row["dim"],
                      "mean_safe": row["mean_safe"], "safe": row["safe_choice"]}), flush=True)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence/phase3")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "memory_and_u.json"
    if destination.exists():
        print(json.dumps({"status": "cached"}), flush=True)
        return 0

    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    y = {s: side[s]["outcomes"].float() for s in ("train", "dev")}
    spec = Fit(steps=args.steps)
    generator = torch.Generator().manual_seed(20260919)
    order = derangement(17, generator)
    report = {"schema": "d4mj_phase3b_v1", "steps": args.steps, "memory_interface": {}, "u_world": {}}

    # ---- A. does the recurrent state carry what z does not? --------------------------------
    for arm in ("raw", "tc"):
        f = {s: payload(MEMORY / f"{arm}.primary_{s}.pt") for s in ("train", "dev")}
        rows = []
        print(json.dumps({"stage": "memory_arm", "arm": arm}), flush=True)
        real = {s: f[s]["observed_z"].float() for s in ("train", "dev")}
        gen = {s: f[s]["c4_generated_z"].float() for s in ("train", "dev")}
        h = {s: f[s]["c4_next_h"].float() for s in ("train", "dev")}
        joint = {s: torch.cat((gen[s], h[s]), -1) for s in ("train", "dev")}
        score("real_z", real["train"], y["train"], real["dev"], y["dev"], spec, args.device, rows)
        score("generated_z", gen["train"], y["train"], gen["dev"], y["dev"], spec, args.device, rows)
        score("next_h", h["train"], y["train"], h["dev"], y["dev"], spec, args.device, rows)
        score("generated_z_and_h", joint["train"], y["train"], joint["dev"], y["dev"],
              spec, args.device, rows)
        # the derangement test on every generated interface
        score("generated_z__action_permuted", gen["train"][:, order], y["train"],
              gen["dev"][:, order], y["dev"], spec, args.device, rows)
        score("next_h__action_permuted", h["train"][:, order], y["train"],
              h["dev"][:, order], y["dev"], spec, args.device, rows)
        score("generated_z_and_h__action_permuted", joint["train"][:, order], y["train"],
              joint["dev"][:, order], y["dev"], spec, args.device, rows)
        report["memory_interface"][arm] = {"context": "c4", "rows": rows}
        del f, real, gen, h, joint

    # ---- B. the existing world trained on a spatial export ---------------------------------
    report["u_world"]["provenance"] = (
        "artifacts/lewm_gates_20260918/m03_matched -- UNSEALED diagnostic. Its arms are NOT Raw/TC: "
        "'raw' is world_z_z and 'tc' is world_u_u, both on one frozen consecutive-TC encoder with "
        "MSE-only worlds. Its 'u' is TRAIN-fitted PCA-192 of the 4x4 pooled patch grid.")
    for arm, label in (("raw", "world_z_z"), ("tc", "world_u_u")):
        f = {s: payload(MATCHED / f"{arm}.{s}.pt") for s in ("train", "dev")}
        rows = []
        print(json.dumps({"stage": "u_world", "arm": label}), flush=True)
        real = {s: f[s]["observed_successor"].float() for s in ("train", "dev")}
        gen = {s: f[s]["generated_successor"].float() for s in ("train", "dev")}
        root = {s: f[s]["projected"].float() for s in ("train", "dev")}
        expand = lambda v: v[:, None].expand(-1, 17, -1)
        score("real_native_fit", real["train"], y["train"], real["dev"], y["dev"], spec, args.device, rows)
        score("generated_native_fit", gen["train"], y["train"], gen["dev"], y["dev"], spec, args.device, rows)
        score("generated_transfer_fit", real["train"], y["train"], gen["dev"], y["dev"], spec, args.device, rows)
        score("generated__action_permuted", gen["train"][:, order], y["train"],
              gen["dev"][:, order], y["dev"], spec, args.device, rows)
        score("root_action_control",
              torch.cat((expand(root["train"]), one_hot_actions(len(root["train"]))), -1), y["train"],
              torch.cat((expand(root["dev"]), one_hot_actions(len(root["dev"]))), -1), y["dev"],
              spec, args.device, rows)
        report["u_world"][label] = {"rows": rows}
        del f, real, gen, root

    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "phase3b_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
