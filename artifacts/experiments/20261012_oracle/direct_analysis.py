"""CPU-only, frozen-criterion readings for the two direct-generator arms.

prepare-criteria consumes real outcomes only and must precede model optimization.
analyze never fits a reader or selects a checkpoint/threshold. Complete sampled
outcomes, including invalid or camera-misregistered outcomes, remain in scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

import direct_coherence as C

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/eda/oracle_v1"
OUT = BASE / "direct_generator"
REFERENCE = OUT / "eval_reference/coherence_reference.json"
METADATA = OUT / "eval_reference/metadata.pt"
PREREG = Path(__file__).with_name("DIRECT_PREREG.md")
BANK = BASE / "snap_bank_readers.pt"
REAL = BASE / "learned_innovation/eval_u3000_h01/mob_heads.pt"
BOOTSTRAPS, SEED = 2000, 20261014


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def clusters(roots):
    keys = []
    for i, root in enumerate(roots):
        if isinstance(root, dict):
            key = root.get("episode_id", root.get("episode_seed", root.get("seed")))
            if key is None:
                key = (root.get("shard"), root.get("slot")) if "shard" in root else i
        else:
            key = i
        keys.append(str(key))
    return keys


def interval(values, group_keys=None):
    """Paired calls share bootstrap draws for the same defined episode groups."""
    values = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
    keys = list(map(str, range(len(values)))) if group_keys is None else group_keys
    valid = np.isfinite(values)
    if not valid.any():
        return {"mean": None, "ci95": [None, None], "roots": 0, "episodes": 0}
    selected_keys = [key for key, ok in zip(keys, valid) if ok]
    unique = list(dict.fromkeys(selected_keys))
    sums = np.array([sum(values[i] for i in range(len(values)) if valid[i] and keys[i] == key)
                     for key in unique])
    counts = np.array([sum(valid[i] and keys[i] == key for i in range(len(values))) for key in unique])
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), BOOTSTRAPS)
    means = (weights @ sums) / (weights @ counts)
    return {"mean": float(values[valid].mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "roots": int(valid.sum()), "episodes": len(unique)}


def add(a, b):
    return [None if x is None or y is None else x + y for x, y in zip(a, b)]


def subtract(a, b):
    return [None if x is None or y is None else x - y for x, y in zip(a, b)]


def exploratory_bundle():
    metadata = load(METADATA)
    panel = load(BASE / "panel.pt")
    fan = load(BASE / "learned_innovation/real_futures/fan.pt")
    indices = metadata["selected_indices"]
    assert metadata["roots"] == [panel["roots"][i] for i in indices]
    assert panel["roots"] == fan["roots"]
    return {"roots": metadata["roots"], "root_tokens": panel["tokens"][indices, 15],
            "fan_tokens": fan["tokens"][indices], "root_layout": panel["layouts"][indices, 15],
            "fan_layout": fan["layouts"][indices], "actions": panel["actions"][indices, 15],
            "root_tiles": metadata["root_tiles"], "fan_tiles": metadata["real_tiles"],
            "shifts": metadata["shifts"]}


def source_alternatives(report, predictions, bundle, bank, real_head):
    """Only real-varying clean source units define semantic diversity coverage."""
    decoded = C.decode(predictions, bank, real_head)["mob"]
    shifts = torch.tensor(report["sampled_camera"]["shift_per_root"])
    n = len(bundle["roots"])
    output, varying, cows = {}, set(), set()
    for reader, flags in decoded.items():
        entries = [[] for _ in range(n)]
        for unit in report["source_units"]:
            if not unit["primary"] or not unit["reader_clean"]:
                continue
            i, p, kind = unit["root"], unit["points"], unit["kind"]
            truth = C._event(bundle["fan_layout"][i], p, kind)
            if len(truth.unique()) < 2:
                continue
            varying.add(i)
            if kind == 1:
                cows.add(i)
            event = C._event(flags[i], p, kind)
            valid = (event < 5) & (shifts[i] == 0).all(-1)
            entries[i].append(float(event[valid].unique().numel() >= 2))
        output[reader] = [float(np.mean(rows)) if rows else None for rows in entries]
    return output, {"variable_clean_roots": len(varying), "variable_clean_cow_roots": len(cows)}


def scored_reader_disagreement(predictions, bank, head):
    decoded = C.decode(predictions, bank, head)["mob"]
    return (decoded["real"][..., C.INTERIOR, :3] != decoded["bank"][..., C.INTERIOR, :3]).double().mean((1, 2, 3)).tolist()


def path_diagnostics(payload):
    """Missing execution evidence is inconclusive, never silently a valid negative."""
    diagnostics = payload.get("diagnostics", {})
    result = {"repeat_exact": None, "initial_live_paths": False}
    repeat = payload.get("noise_replay_maxabs", diagnostics.get("noise0_repeat_maxabs", diagnostics.get("repeat_noise_maxabs")))
    if repeat is not None:
        result["repeat_exact"] = float(repeat) == 0.
    validation = OUT / "validation.json"
    if validation.exists():
        proof = json.loads(validation.read_text())
        source_bound = all(sha(ROOT / name) == digest for name, digest in proof["sources"].items())
        result["validation_sha256"] = sha(validation)
        result["initial_live_paths"] = (source_bound and proof["matched_initial_output_maxabs"] == 0
                                        and proof["future_mutation_current_output_maxabs"] == 0
                                        and proof["independent_draw_output_maxabs"] > 0
                                        and all(x > 0 for x in proof["draw_gradient_norms"])
                                        and proof["backbone_gradient_norm"] > 0
                                        and proof["routing_only_noise_effect_maxabs"] > 0
                                        and proof["generation_only_noise_effect_maxabs"] > 0
                                        and proof["first_power_identity_symmetry_exact"]
                                        and all(proof["resume_streams_exact"].values())
                                        and max(proof["resume_endpoint_maxabs"].values()) < 1e-4
                                        and proof["real_batch16_peak_allocated_gib"] < 6)
    # Learned disappearance of noise is an experimental negative, not a broken
    # path. In particular the attraction-only arm is expected to collapse.
    pred = payload["predictions"]
    result["final_noise_output_maxabs"] = float((pred - pred[:, :1]).abs().max())
    result["passed"] = result["repeat_exact"] is True and result["initial_live_paths"] is True
    return result


def prepare_criteria(path=OUT / "criteria.json"):
    reference = json.loads(REFERENCE.read_text())
    contract = reference["reference_contract"]
    assert sha(Path(C.__file__)) == contract["helper_sha256"], "reference/helper drift"
    assert sha(BANK) == contract["bank_reader_sha256"]
    assert sha(REAL) == contract["real_reader_sha256"]
    assert sha(METADATA) == contract["metadata_sha256"]
    bundle = exploratory_bundle()
    groups = clusters(bundle["roots"])
    rows = reference["raw_per_root"]["real"]
    count = reference["coverage"]["primary_clean_variable_source_units"]
    assert count > 0
    real_draws = reference["real_draws"]
    finite_source = 3 / (count * real_draws)
    absence = rows["primary_clean_real_absent_fraction"]
    multiple = rows["primary_clean_real_multiple_destination_fraction"]
    real_source_duplicate = rows["primary_clean_source_and_other_destination_fraction"]
    camera = rows["primary_clean_real_camera_mismatch_fraction"]
    real_static = reference["static_content"]["scored_real_reader_corruption_per_root"]
    static_cells = sum(reference["static_content"]["scored_eligible_cells_per_root"])
    static_margin = interval(real_static, groups)["ci95"][1] + 3 / (static_cells * real_draws)
    half = real_draws // 2
    diversity_rows = [[] for _ in bundle["roots"]]
    for unit in reference["source_units"]:
        if not unit["primary"] or not unit["reader_clean"]:
            continue
        i = unit["root"]
        event = C._event(bundle["fan_layout"][i], unit["points"], unit["kind"])
        if len(event.unique()) < 2:
            continue
        diversity_rows[i].append(.5 * (float(event[:half].unique().numel() >= 2)
                                         + float(event[half:].unique().numel() >= 2)))
    diversity = [float(np.mean(x)) if x else None for x in diversity_rows]
    # Keep the reader guard on the same scored domain, not untrained edge cells.
    disagreement = scored_reader_disagreement(bundle["fan_tokens"], load(BANK), load(REAL)["heads"]["real"])
    criteria = {
        "format": "direct_frozen_criteria_v1", "prepared_from_real_only": True,
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), Path(C.__file__), PREREG, BANK, REAL, REFERENCE, METADATA)},
        "bootstrap": {"draws": BOOTSTRAPS, "seed": SEED, "unit": "episode clusters; paired root arrays"},
        "reference_roots": len(bundle["roots"]), "reference_coverage": reference["coverage"],
        "finite_source_resolution": finite_source,
        "margin_rule": "Real-reader upper95 reference rate plus rule-of-three finite-fan resolution; this is an operational frozen margin, not a population confidence bound",
        "margins": {"source_absent_or_multiple": interval(add(absence, multiple), groups)["ci95"][1] + finite_source,
                    "source_duplicate": interval(real_source_duplicate, groups)["ci95"][1] + finite_source,
                    "source_camera_mismatch": interval(camera, groups)["ci95"][1] + finite_source,
                    "static_paired_increase": static_margin,
                    "reader_disagreement_paired_increase": interval(disagreement, groups)["ci95"][1] + 3 / (len(bundle["roots"]) * real_draws * 35 * 3)},
        "minimum_actual_alternative_coverage": max(0., interval(diversity, groups)["ci95"][0] - finite_source),
        "alternative_reference": {"raw_per_root": diversity, "interval": interval(diversity, groups),
                                  "rule": "At least two valid source outcomes in independent real16 branches, averaged over both halves"},
        "static_reference": {"interval": interval(real_static, groups), "scored_cells": static_cells,
                             "finite_fan_resolution": 3 / (static_cells * real_draws)},
        "primary": "real-reader scored35 joint occupancy energy; upper95(energy-point)<0",
        "primary_checkpoint": 3000, "samples": 64,
        "confirmation_coverage": {"minimum_variable_clean_roots": 20, "minimum_variable_clean_cow_roots": 20},
        "reference_energy": reference["real_only_margins"],
        "limits": ["Current14 source roots permit exploratory_promising only", "Full-state fan is narrower than partial-history belief",
                   "Hidden-phase zombie/skeleton probability alignment is descriptive", "No actor, H16 or four-way separation claim"],
    }
    criteria["criteria_id"] = hashlib.sha256(json.dumps(criteria, sort_keys=True).encode()).hexdigest()
    if Path(path).exists():
        assert json.loads(Path(path).read_text()) == criteria, "frozen criteria drift"
    else:
        atomic(criteria, path)
    print(json.dumps({"stage": "criteria_frozen", "criteria_id": criteria["criteria_id"], "margins": criteria["margins"],
                      "minimum_actual_alternative_coverage": criteria["minimum_actual_alternative_coverage"], "path": str(path)}), flush=True)
    return criteria


def verify_criteria(path):
    criteria = json.loads(Path(path).read_text())
    for path, digest in criteria["sources"].items():
        assert sha(ROOT / path) == digest, f"frozen criteria source drift: {path}"
    data = {key: value for key, value in criteria.items() if key != "criteria_id"}
    assert hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() == criteria["criteria_id"]
    return criteria


def analyze(directory, fresh=None, criteria_path=OUT / "criteria.json"):
    criteria = verify_criteria(criteria_path)
    bundle = load(fresh) if fresh is not None else exploratory_bundle()
    groups = clusters(bundle["roots"])
    bank, head = load(BANK), load(REAL)["heads"]["real"]
    reports, prediction_contracts, paths, coverage, execution = {}, {}, {}, {}, {}
    for arm in ("point", "energy"):
        path = Path(directory) / f"pred_{arm}.pt"
        payload = load(path)
        assert payload["arm"] == arm and payload["roots"] == bundle["roots"], "arm/root order mismatch"
        predictions = payload["predictions"]
        assert predictions.shape == (len(bundle["roots"]), criteria["samples"], 81, 192)
        assert not predictions.is_cuda and bool(torch.isfinite(predictions).all())
        actions = bundle["actions"]
        if actions.ndim == 2:
            actions = actions[:, -1]
        report = C.analyze(predictions, bundle["root_tokens"], bundle["fan_tokens"], bundle["root_layout"], bundle["fan_layout"],
                           bank, head, shifts=bundle["shifts"], actions=actions,
                           root_tiles=bundle["root_tiles"], real_tiles=bundle["fan_tiles"])
        alternatives, coverage[arm] = source_alternatives(report, predictions, bundle, bank, head)
        report["actual_alternative_coverage_per_root"] = alternatives
        report["scored_reader_disagreement_per_root"] = scored_reader_disagreement(predictions, bank, head)
        reports[arm] = report
        execution[arm] = path_diagnostics(payload)
        prediction_contracts[arm] = {key: payload[key] for key in ("contract_id", "contract", "update", "diagnostics", "costs") if key in payload}
        paths[arm] = sha(path)
    raw_point, raw_energy = reports["point"]["raw_per_root"], reports["energy"]["raw_per_root"]
    checks, paired, reasons = {}, {}, []
    for reader in ("real", "bank"):
        name = "scored_joint_occupancy_energy_reader_reference"
        diff = subtract(raw_energy[reader][name], raw_point[reader][name])
        paired[reader + "_joint_energy_minus_point"] = {"raw_per_root": diff, "interval": interval(diff, groups)}
    checks["primary_distribution"] = paired["real_joint_energy_minus_point"]["interval"]["ci95"][1] < 0
    primary_clean_roots = {unit["root"] for unit in reports["energy"]["source_units"] if unit["primary"] and unit["reader_clean"]}
    clean_diff = [value if i in primary_clean_roots else None for i, value in enumerate(paired["real_joint_energy_minus_point"]["raw_per_root"])]
    paired["reader_valid_source_stratum_joint_energy_minus_point"] = {"raw_per_root": clean_diff, "interval": interval(clean_diff, groups)}
    physical = {}
    for name, metric in (("source_absent_or_multiple", None), ("source_duplicate", "source_and_other_destination_fraction"),
                         ("source_camera_mismatch", "camera_mismatch_fraction")):
        if metric is None:
            values = add(raw_energy["real"]["primary_clean_absent_fraction"], raw_energy["real"]["primary_clean_multiple_destination_fraction"])
        else:
            values = raw_energy["real"]["primary_clean_" + metric]
        reading = interval(values, groups)
        physical[name] = {"raw_per_root": values, "interval": reading, "margin": criteria["margins"][name]}
        checks[name] = reading["ci95"][1] is not None and reading["ci95"][1] <= criteria["margins"][name]
    static = subtract(reports["energy"]["static_content"]["scored_sampled_corruption_per_root"],
                      reports["point"]["static_content"]["scored_sampled_corruption_per_root"])
    paired["static_corruption_energy_minus_point"] = {"raw_per_root": static, "interval": interval(static, groups)}
    checks["static_fidelity"] = paired["static_corruption_energy_minus_point"]["interval"]["ci95"][1] <= criteria["margins"]["static_paired_increase"]
    alternatives = reports["energy"]["actual_alternative_coverage_per_root"]["real"]
    alt = interval(alternatives, groups)
    checks["actual_alternatives"] = alt["ci95"][0] is not None and alt["ci95"][0] >= criteria["minimum_actual_alternative_coverage"]
    disagreement = subtract(reports["energy"]["scored_reader_disagreement_per_root"], reports["point"]["scored_reader_disagreement_per_root"])
    paired["reader_disagreement_energy_minus_point"] = {"raw_per_root": disagreement, "interval": interval(disagreement, groups)}
    checks["reader_sensitivity"] = (paired["bank_joint_energy_minus_point"]["interval"]["mean"] < 0
                                    and paired["reader_valid_source_stratum_joint_energy_minus_point"]["interval"]["mean"] is not None
                                    and paired["reader_valid_source_stratum_joint_energy_minus_point"]["interval"]["mean"] < 0
                                    and interval(disagreement, groups)["ci95"][1] <= criteria["margins"]["reader_disagreement_paired_increase"])
    for name, passed in checks.items():
        if not passed:
            reasons.append(name + " failed")
    missing = alt["roots"] == 0
    valid_execution = all(row["passed"] for row in execution.values())
    if not valid_execution:
        reasons.append("missing or failed numerical/live-noise execution evidence")
    passed = all(checks.values())
    fresh_coverage = coverage["energy"]["variable_clean_roots"] >= 20 and coverage["energy"]["variable_clean_cow_roots"] >= 20
    update_values = {prediction_contracts[a].get("update") for a in ("point", "energy")}
    final = update_values == {criteria["primary_checkpoint"]}
    if not final:
        verdict = "milestone_diagnostic"
    elif not valid_execution:
        verdict = "inconclusive_execution"
    elif missing:
        verdict = "inconclusive_coverage"
    elif passed and fresh is None:
        verdict = "exploratory_promising"
    elif passed and fresh_coverage:
        verdict = "confirmed_success"
    elif fresh is not None and not fresh_coverage:
        verdict = "inconclusive_coverage"
    else:
        verdict = "coherent_completion_not_established"
    result = {"format": "direct_analysis_v1", "criteria_id": criteria["criteria_id"], "criteria_sha256": sha(criteria_path),
              "prediction_hashes": paths, "prediction_contracts": prediction_contracts,
              "fresh_bundle_sha256": sha(fresh) if fresh is not None else None,
              "scope": "One-step controlled continuous completion; no actor/H16/four-way separation claim",
              "verdict": verdict, "allows_confirmation": verdict == "exploratory_promising", "checks": checks,
              "failed_conditions": reasons, "coverage": coverage, "actual_alternatives": {"raw_per_root": alternatives, "interval": alt},
              "physical": physical, "paired": paired, "execution": execution, "arms": reports,
              "negative_validity_requires": ["Real-reader controls", "Live noise/gradient and numerical/resume checks", "Optimization curves reviewed before a budget-sufficiency claim"],
              "conditional_law_limit": "Cow source probabilities are the cleanest visible-conditioned sensitivity; hidden zombie/skeleton phase is not assumed observed"}
    destination = Path(directory) / "analysis.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == result, "analysis evidence drift"
    else:
        atomic(result, destination)
    print(json.dumps({"stage": "analysis_done", "verdict": verdict, "checks": checks, "coverage": coverage, "path": str(destination)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare-criteria")
    p.add_argument("--criteria", type=Path, default=OUT / "criteria.json")
    p = commands.add_parser("analyze")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--fresh", type=Path)
    p.add_argument("--criteria", type=Path, default=OUT / "criteria.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "prepare-criteria":
        prepare_criteria(args.criteria)
    else:
        analyze(args.directory, args.fresh, args.criteria)


if __name__ == "__main__":
    main()
