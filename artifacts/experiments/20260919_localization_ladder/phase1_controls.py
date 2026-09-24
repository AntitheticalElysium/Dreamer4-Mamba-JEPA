"""Phase 1 controls: does the rank-supervised gain survive its matched baselines?

Pair-ranking supervision roughly doubles LeWM safe choice. That is only evidence about the
representation if the same supervision applied to inputs that contain NO successor information
does not reach the same score. An action-only scorer cannot vary within a root -- its argmin is
the same action everywhere -- so it measures exactly the action prior the published head was
accused of having, and it is the control that matters most.

Controls, all under identical supervision, families, seeds and standardization:

  uniform              analytic: expected safe rate of a uniform draw over 17 actions
  best_constant        the single action with the best TRAIN safe rate, applied to DEV
  action_only          rank-fitted on the one-hot action alone
  root_action          rank-fitted on the ROOT encoding + action: knows the present, not the future
  history_action       rank-fitted on the causally encoded prefix + past actions + action
  shuffled_successor   successor features permuted across roots, destroying the root-successor
                       correspondence while preserving every marginal

`root_action` and `history_action` are the baselines a predicted successor must beat: if knowing
the current state and the proposed action is enough, then no successor representation -- real or
generated -- has added anything.
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
from readout import DEATH, Fit, evaluate, fit_head, one_hot_actions, standardize, summarize

SIDECAR = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/sidecar/sidecar.probe_only.pt"
DIRECT = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/features/direct_mamba"
SEEDS = (0, 1, 2)
PRIMARY_ARMS = ("mamba_raw", "mamba_tc", "direct")


def load_taps(name):
    return torch.load(HERE / f"cache/taps.{name}.pt", map_location="cpu", weights_only=False)


def direct_features(split):
    return resolve_payload(torch.load(f"{DIRECT}.{split}.pt", map_location="cpu",
                                      weights_only=False))["features"]


def successor_rung(arm, taps, split, tap="z"):
    if arm == "direct":
        return direct_features(split)["observed_successor"].float()
    value = taps[arm][split]["successor"][tap].float()
    return value.flatten(2) if value.dim() == 4 else value


def root_rung(arm, taps, split, tap="z"):
    if arm == "direct":
        return direct_features(split)["projected"].float()
    value = taps[arm][split]["root"][tap].float()
    return value.flatten(1) if value.dim() == 3 else value


def history_rung(arm, taps, split):
    """Causally encoded prefix plus its past actions, flattened to one vector per root."""
    if arm == "direct":
        return None
    frames = taps[arm][split]["history"]["z"].float()
    actions = taps[arm][split]["history_actions"]
    hot = torch.nn.functional.one_hot(actions.clamp(min=0), 18).float().flatten(1)
    return torch.cat((frames.flatten(1), hot), 1)


def expand(vector):
    return vector[:, None].expand(-1, 17, -1)


def analytic_controls(y_train, y_dev):
    labels_dev = y_dev[..., DEATH].bool()
    usable = labels_dev.any(1) & (~labels_dev).any(1)
    rows = torch.where(usable)[0]
    safe_dev = (~labels_dev)[rows].float()
    uniform = float(safe_dev.mean())
    labels_train = y_train[..., DEATH].bool()
    train_usable = labels_train.any(1) & (~labels_train).any(1)
    per_action_train = (~labels_train)[torch.where(train_usable)[0]].float().mean(0)
    best = int(per_action_train.argmax())
    return {"uniform_over_17": uniform,
            "best_constant_action": {"action": best,
                                     "train_safe_rate": float(per_action_train[best]),
                                     "dev_safe_rate": float(safe_dev[:, best].mean()),
                                     "dev_safe": int(safe_dev[:, best].sum()),
                                     "opportunity_roots": int(usable.sum())},
            "per_action_dev_safe_rate": safe_dev.mean(0).tolist()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence/phase1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--family", default="mlp128")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "controls.json"
    if destination.exists():
        print(json.dumps({"status": "cached"}), flush=True)
        return 0

    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    taps = {n: load_taps(n) for n in ("mamba_raw", "mamba_tc")}
    y = {s: side[s]["outcomes"].float() for s in ("train", "dev")}
    spec = Fit(steps=args.steps)
    report = {"schema": "d4mj_phase1_controls_v1", "family": args.family, "steps": args.steps,
              "analytic": analytic_controls(y["train"], y["dev"]), "fitted": []}
    print(json.dumps({"stage": "analytic", **{k: v for k, v in report["analytic"].items()
                                              if k != "per_action_dev_safe_rate"}}), flush=True)

    def add(name, xtr, xdv, objective="rank"):
        xtr, xdv = standardize(xtr, xdv)
        for seed in SEEDS:
            model, curve, params = fit_head(xtr, y["train"], family=args.family, objective=objective,
                                            seed=seed, spec=spec, device=args.device)
            row = {"control": name, "objective": objective, "seed": seed, "parameters": params,
                   "train": evaluate(model, xtr, y["train"], args.device),
                   "dev": evaluate(model, xdv, y["dev"], args.device)}
            report["fitted"].append(row)
            print(json.dumps({"stage": "control", "control": name, "objective": objective,
                              "seed": seed,
                              "dev_safe": f"{row['dev']['safe_choice']}/{row['dev']['opportunity_roots']}",
                              "auc": round(row["dev"]["within_root_auc"], 3)}), flush=True)

    # Action-only: identical for every arm, so it is fitted once.
    add("action_only", one_hot_actions(len(y["train"])), one_hot_actions(len(y["dev"])))

    for arm in PRIMARY_ARMS:
        root = {s: root_rung(arm, taps, s) for s in ("train", "dev")}
        add(f"{arm}:root_action",
            torch.cat((expand(root["train"]), one_hot_actions(len(root["train"]))), -1),
            torch.cat((expand(root["dev"]), one_hot_actions(len(root["dev"]))), -1))
        history = {s: history_rung(arm, taps, s) for s in ("train", "dev")}
        if history["train"] is not None:
            add(f"{arm}:history_action",
                torch.cat((expand(history["train"]), one_hot_actions(len(history["train"]))), -1),
                torch.cat((expand(history["dev"]), one_hot_actions(len(history["dev"]))), -1))
        # Shuffled-successor: same marginals, no root-successor correspondence.
        generator = torch.Generator().manual_seed(20260919)
        succ = {s: successor_rung(arm, taps, s) for s in ("train", "dev")}
        permuted = {s: succ[s][torch.randperm(len(succ[s]), generator=generator)] for s in ("train", "dev")}
        add(f"{arm}:shuffled_successor", permuted["train"], permuted["dev"])

    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "controls_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
