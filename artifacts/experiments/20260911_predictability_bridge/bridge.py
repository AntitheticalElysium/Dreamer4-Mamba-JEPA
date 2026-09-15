"""Predictability bridge: can the frozen world transition a spatial stream?

The feature ladder showed that LeWM's ViT patch grid retains state its CLS/z
export discards.  That is a statement about *observation* only: the trained world
predicts projected ``z`` and cannot generate patch tokens, so patch retention and
``z``'s transition-preservation advantage are not yet combinable.

This asks whether they could be, without retraining anything.  For every root and
every candidate action the frozen world already produces ``h_next`` (from
consuming the completed pair) and a generated ``z_next``.  The question is
whether either predicts the *observed* successor's patch representation, and
whether a predicted patch vector still carries the semantics that made patch
tokens attractive.

Two stages, both fitted on TRAIN and evaluated on DEV:

1. Geometry -- fit a probe from the world state to the observed successor patch
   vector.  Reported against an action-only floor and a persistence floor (the
   root's own patch vector), and against a ceiling that is handed the real
   observed ``z``.  Total and action-effect R^2 are separated, because predicting
   a root's mean successor is easy and predicting how the seventeen actions
   differ is the part imagination needs.
2. Semantics -- read the *predicted* patch vector with the decoder fitted on
   *observed* patch vectors, exactly the M03 observed-to-generated contract.  A
   predicted vector that is close in MSE but semantically empty fails here.

Every condition is widened to the same 192+256 slot layout with zero fill, so all
of them enter a probe of identical width and parameter count -- the same device
``memory_view`` uses for its z/h ablations.

Decision map:

* predictable from ``h`` alone            -> a lightweight spatial readout may be viable
* predictable only with generated ``z``   -> the spatial stream rides on z, not on memory
* not predictable in either arm           -> Phase-1 must be retrained to carry the stream
* only Raw predictable                    -> stop TC, develop the Raw spatial export
* predictable but semantics fail          -> readout/probe transfer, not missing dynamics

Scope: the primary support-v2 panel, four-frame (``c4``) context, one seed.  This
is retention and one-step transition, never control, and it cannot authorize M4.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
LADDER = ROOT / "artifacts/experiments/20260910_feature_ladder/ladder.py"
SCHEMA = "d4mj_predictability_bridge"
CASE = "c4"
TARGETS = ("patch_pca192", "patch_mean")
Z_WIDTH, H_WIDTH = 192, 256


def _ladder():
    """Import the ladder's frozen encoding helpers rather than restating them."""
    spec = importlib.util.spec_from_file_location("ladder", LADDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slots(z: torch.Tensor | None, h: torch.Tensor | None, rows: int, actions: int) -> torch.Tensor:
    """One fixed [z|h] layout with zero fill, so every condition has equal width."""
    zeros = lambda width: torch.zeros(rows, actions, width)
    left = zeros(Z_WIDTH) if z is None else z
    right = zeros(H_WIDTH) if h is None else h
    if left.shape[-1] < Z_WIDTH:
        left = torch.cat((left, torch.zeros(rows, actions, Z_WIDTH - left.shape[-1])), -1)
    return torch.cat((left.float(), right.float()), -1).flatten(0, 1)


def _conditions(memory: dict, patch_root: torch.Tensor) -> dict[str, torch.Tensor]:
    """World-state inputs, plus the two floors and the observed-z ceiling."""
    rows, actions = memory[f"{CASE}_next_h"].shape[:2]
    h = memory[f"{CASE}_next_h"]
    return {
        "h_only": _slots(None, h, rows, actions),
        "joint_generated": _slots(memory[f"{CASE}_generated_z"], h, rows, actions),
        "joint_observed_ceiling": _slots(memory["observed_z"], h, rows, actions),
        "action_only_floor": _slots(torch.eye(actions)[None].expand(rows, -1, -1), None, rows, actions),
        "persistence_floor": _slots(patch_root[:, None].expand(rows, actions, -1), None, rows, actions),
    }


def _basis(fit: torch.Tensor, project):
    """Unit-variance target coordinates, with numerically null directions dropped.

    PCA components are wildly unequally scaled (per-coordinate TRAIN std spans
    0 to ~49 here).  Used raw, the near-null tail meets ``_fit_probe_many``'s
    ``clamp_min(1e-6)`` in stage 2 and amplifies pure noise by ~1e6.  Whitening
    also makes stage-1 R^2 an average over coordinates rather than a report on
    the leading component alone.  The ladder's own basis is left untouched.
    """
    values = project(fit)
    mean = values.mean(0, keepdim=True)
    std = values.std(0, unbiased=False)
    keep = std > std.max() * 1e-3
    scale = std[keep]
    return int(keep.sum()), lambda x: (project(x) - mean)[:, keep] / scale


def _r2(prediction: torch.Tensor, truth: torch.Tensor, roots: torch.Tensor, settings) -> dict:
    """Total and action-effect R^2 over all coordinates, bootstrapped over roots."""
    from d4mj.m03.gate import _root_bootstrap

    def score(values, target):
        def metric(rows):
            t, p = target[rows], values[rows]
            denominator = float((t - t.mean(0, keepdim=True)).square().sum())
            return 1 - float((p - t).square().sum()) / denominator if denominator > 1e-12 else None
        point, interval = _root_bootstrap(values, roots, metric,
                                          draws=settings.bootstrap_draws, seed=settings.seed + 4100)
        return {"value": point, "interval": interval,
                "status": "measured" if interval is not None else "insufficient_coverage"}

    effect = lambda x: (x.reshape(-1, 17, x.shape[-1]) - x.reshape(-1, 17, x.shape[-1]).mean(1, keepdim=True)
                        ).reshape(-1, x.shape[-1])
    return {"total": score(prediction, truth), "action_effect": score(effect(prediction), effect(truth))}


def run(source: Path, device: str, batch: int, limit: int = 0,
        frozen_eval_proof: Path | None = None) -> dict:
    from d4mj.data import _sha256
    from d4mj.m03.cache import resolve_payload
    from d4mj.m03.gate import (STATIC_BINARY, STATIC_CONTINUOUS, M03Settings, _binary_metrics,
                               _expanded_roots, _fit_probe_many, _regression_metrics, load_m03_bundle)

    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    ladder = _ladder()
    settings = M03Settings()
    manifest = json.loads((source / "sidecar/manifest.json").read_text())
    sidecar_path = source / "sidecar/sidecar.probe_only.pt"
    if _sha256(sidecar_path) != manifest["sidecar"]["sha256"]:
        raise ValueError("bridge: sidecar bytes differ from their sealed manifest")
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    splits = sidecar["splits"]
    if limit:
        splits = {s: {k: v[:limit] if isinstance(v, torch.Tensor) else v for k, v in rows.items()}
                  for s, rows in splits.items()}
    contract = json.loads((source / "run.json").read_text())["contract"]

    report = {"schema": SCHEMA, "panel": "primary_support_v2", "context": CASE, "device": device,
              "source_run": str(source.resolve()), "sidecar_sha256": manifest["sidecar"]["sha256"],
              "ladder_source": {"path": str(LADDER), "sha256": _sha256(LADDER)},
              "capacity": "every condition zero-filled to a 192+256 slot layout; equal probe width",
              "target_basis": "TRAIN-whitened to unit per-coordinate variance; null directions dropped",
              "stage2": "decoder fitted on observed patch vectors, applied unchanged to predicted ones",
              "arms": {}, "m4_authorized": False}

    for arm in ("raw", "tc"):
        bundle, payload, _ = load_m03_bundle(Path(contract[f"{arm}_checkpoint"]["path"]),
                                             device=device, dataset_sha256=sidecar["dataset_sha256"],
                                             frozen_eval_proof=frozen_eval_proof)
        del payload
        patch, memory = {}, {}
        for split, values in splits.items():
            count = len(values["context"])
            roots = ladder._encode_rungs(bundle, values["context"][:, -1], batch)
            forks = ladder._encode_rungs(bundle, values["successors"].flatten(0, 1), batch)
            patch[split] = {"root": roots, "successor": {k: v.reshape(count, 17, -1) for k, v in forks.items()}}
            cached = resolve_payload(torch.load(source / f"memory/features/{arm}.primary_{split}.pt",
                                                map_location="cpu", weights_only=False))["features"]
            memory[split] = {k: v[:limit] if limit else v for k, v in cached.items()}
            if len(memory[split][f"{CASE}_next_h"]) != count:
                raise ValueError("bridge: cached Mamba rows do not match the sidecar split")
        del bundle
        if device == "cuda":
            torch.cuda.empty_cache()

        project = ladder._pca(patch["train"]["successor"]["patch"].flatten(0, 1), Z_WIDTH)
        arm_report = {}
        for target in TARGETS:
            key = "patch" if target == "patch_pca192" else target
            base = project if target == "patch_pca192" else (lambda x: x.float())
            kept, transform = _basis(patch["train"]["successor"][key].flatten(0, 1), base)
            feature = {}
            for split in splits:
                count = len(patch[split]["root"][key])
                feature[split] = {
                    "root": transform(patch[split]["root"][key]),
                    "successor": transform(patch[split]["successor"][key].flatten(0, 1)).reshape(count, 17, -1)}
            train_y = feature["train"]["successor"].flatten(0, 1)
            dev_y = feature["dev"]["successor"].flatten(0, 1)
            dev_roots = _expanded_roots(splits["dev"]["episode"], splits["dev"]["next_binary"][..., 0])
            inputs = {s: _conditions(memory[s], feature[s]["root"]) for s in splits}

            # Stage 2's decoder is fitted once, on observed patch vectors only.
            semantic_train = feature["train"]["successor"].flatten(0, 1).to(device)
            rows = {}
            for hidden, family in ((False, "linear"), (True, "mlp")):
                predicted, geometry = {}, {}
                for name in inputs["train"]:
                    hat = _fit_probe_many(inputs["train"][name].to(device), train_y.to(device),
                                          {"dev": inputs["dev"][name].to(device)}, settings,
                                          hidden=hidden, binary=False)["dev"]
                    predicted[name] = hat
                    geometry[name] = _r2(hat, dev_y, dev_roots, settings)
                evaluation = {"observed_ceiling": dev_y.to(device)}
                evaluation.update({name: value.to(device) for name, value in predicted.items()})
                binary = _fit_probe_many(semantic_train, splits["train"]["next_binary"].flatten(0, 1).float().to(device),
                                         evaluation, settings, hidden=hidden, binary=True)
                continuous = _fit_probe_many(semantic_train, splits["train"]["next_continuous"].flatten(0, 1).float().to(device),
                                             evaluation, settings, hidden=hidden, binary=False)
                rows[family] = {
                    "geometry": geometry,
                    "semantics": {name: {
                        "binary": _binary_metrics(binary[name], splits["dev"]["next_binary"].flatten(0, 1).bool(),
                                                  dev_roots, STATIC_BINARY, settings),
                        "continuous": _regression_metrics(continuous[name], splits["dev"]["next_continuous"].flatten(0, 1).float(),
                                                          dev_roots, STATIC_CONTINUOUS, settings),
                    } for name in evaluation},
                }
            arm_report[target] = {"retained_coordinates": kept, **rows}
        report["arms"][arm] = arm_report
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2",
                        help="sealed M03 run supplying the sidecar and the cached Mamba state")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--frozen-eval-proof", type=Path, default=ROOT / "d4mj/m03/frozen_eval_compat.json",
                        help="measured source-delta proof; evaluation only, never training resume")
    parser.add_argument("--limit", type=int, default=0,
                        help="structural smoke over the first N roots per split; never a result")
    args = parser.parse_args(argv)

    import sys
    sys.path.insert(0, str(ROOT))
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / ("bridge.smoke.json" if args.limit else "bridge.json")
    if destination.exists():
        raise FileExistsError(f"bridge: refusing to replace {destination}")
    report = run(args.source, args.device, args.batch, args.limit, args.frozen_eval_proof)
    if args.limit:
        report["mode"] = "structural_smoke_not_a_result"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "report": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
