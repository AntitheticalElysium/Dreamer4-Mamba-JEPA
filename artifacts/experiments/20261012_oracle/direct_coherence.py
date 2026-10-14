"""CPU-only joint outcome readings for direct stochastic completion.

Training must never import this module or consume its simulator-derived masks.
Readers are supplied frozen and fitted on real states. Exact labels are used
only to validate those readers, define evaluation strata, and score outcomes.

Occupancy is not object identity. The primary source/destination reading uses
isolated, stationary-camera, near-player sources: the stay/one-step-neighbor
neighborhood cannot contain a second same-type source or a new spawn. Creation,
deletion, crowding, boundaries, arrows and camera changes remain separate.
Nearest-real local tokens are a sensitivity diagnostic, not a valid-state
projection: assigning a mean to a real branch does not make the mean physical.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

LAYOUT_TYPES = ("zombie", "cow", "skeleton", "arrow")
READER_TYPES = ("zombie", "skeleton", "cow", "arrow")
READER_TO_LAYOUT = (0, 2, 1, 3)
PRIMARY_TYPES = LAYOUT_TYPES[:3]
HEIGHT, WIDTH, PLAYER_CELL, CELLS = 7, 9, 31, 63
INTERIOR = tuple(r * WIDTH + c for r in range(1, 6) for c in range(1, 8))
SOURCE_STEPS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
EVENT_NAMES = ("stay", "up", "down", "left", "right", "absent", "multiple")
SCROLL_SOURCE = Path(__file__).resolve().parents[1] / "20260927_levers/scroll.py"


@torch.no_grad()
def _camera_shifts(root_tokens: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """Existing parameter-free scroll reader, equally applied to every branch."""
    spec = importlib.util.spec_from_file_location("direct_coherence_scroll", SCROLL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    lookup = torch.as_tensor(module.SHIFTS, dtype=torch.long)
    chunks = []
    # Limit temporary maps; never materialize the full repeated-root token bank.
    for start in range(0, len(root_tokens), 4):
        observed = root_tokens[start:start + 4, None]
        indices = module.estimate(observed, samples[start:start + 4])
        chunks.append(lookup[indices])
    return torch.cat(chunks)


def _cpu(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu":
        raise ValueError("direct_coherence receives CPU tensors only")
    return x.detach()


def _linear(head: dict, tokens: torch.Tensor, bank_convention: bool = False) -> torch.Tensor:
    shape = tokens.shape[:-1]
    flat = tokens.reshape(-1, tokens.shape[-1]).float()
    z = ((flat - head["mu"].cpu()) / head["sd"].cpu()).double()
    w = head["w"].cpu().double()
    if bank_convention:
        # Reproduce snap_eval.Readers's exact bias concatenation and float32
        # output. The newer real ridge reader uses separated FP64 bias.
        out = (torch.cat([z, torch.ones(len(z), 1, dtype=torch.float64)], 1) @ w).float()
    else:
        out = z @ w[:-1] + w[-1]
    return out.reshape(*shape, w.shape[-1])


@torch.no_grad()
def decode(tokens: torch.Tensor, bank_state: dict, real_head: dict | None = None) -> dict:
    """Same fixed functions and thresholds for real and generated states.

    The player patch is omitted by the established mob/tile probes. It is never
    counted as an empty physical cell; evaluation masks exclude it explicitly.
    """
    _cpu(tokens)
    if tokens.shape[-2:] != (81, 192):
        raise ValueError("Expected [...,81,192] normalized Raw patch tokens")
    shape = tokens.shape[:-2]
    selected = [c for c in range(CELLS) if c != PLAYER_CELL]
    local = tokens[..., selected, :]
    out: dict[str, Any] = {"mob": {}, "scores": {}}
    pairs = [("bank", bank_state["heads"]["mob"], bank_state["thresholds"])]
    if real_head is not None:
        pairs.append(("real", real_head, real_head["thresholds"]["dev_f1"]))
    for name, head, threshold in pairs:
        scores = _linear(head, local, bank_convention=name == "bank")
        flags = scores > torch.as_tensor(threshold, dtype=scores.dtype)
        full = torch.zeros(*shape, CELLS, 4, dtype=torch.bool)
        full_scores = torch.zeros(*shape, CELLS, 4, dtype=scores.dtype)
        full[..., selected, :] = flags[..., READER_TO_LAYOUT]
        full_scores[..., selected, :] = scores[..., READER_TO_LAYOUT]
        out["mob"][name], out["scores"][name] = full, full_scores
    tile = torch.full((*shape, CELLS), -1, dtype=torch.long)
    tile[..., selected] = _linear(bank_state["heads"]["tile"], local, bank_convention=True).argmax(-1)
    out["tile"] = tile
    return out


def _reader_validity(decoded: torch.Tensor, truth: torch.Tensor) -> dict:
    cells = [c for c in range(CELLS) if c != PLAYER_CELL]
    d, y = decoded[..., cells, :], truth[..., cells, :].bool()
    result = {}
    for j, name in enumerate(LAYOUT_TYPES):
        dd, yy = d[..., j], y[..., j]
        tp, fp, fn = int((dd & yy).sum()), int((dd & ~yy).sum()), int((~dd & yy).sum())
        true = int(yy.sum())
        result[name] = {"true": true, "tp": tp, "fp": fp, "fn": fn,
                        "recall": tp / true if true else None,
                        "precision": tp / (tp + fp) if tp + fp else None,
                        "count_exact_fraction": float((dd.sum(-1) == yy.sum(-1)).float().mean()),
                        "layout_exact_fraction": float((dd == yy).all(-1).float().mean())}
    return result


def _energy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """First-power Euclidean ES for joint categorical occupancy outcomes."""
    x, y = x.double().flatten(2), y.double().flatten(2)
    n = x.shape[1]
    cross = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").mean((1, 2))
    if n > 1:
        spread = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist").sum((1, 2)) / (n * (n - 1))
    else:
        spread = torch.zeros(len(x), dtype=torch.float64)
    return (cross - .5 * spread) / np.sqrt(x.shape[-1])


def _categorical_brier(pred: torch.Tensor, real: torch.Tensor, classes: int) -> torch.Tensor:
    """Mean Brier against each observed branch, not distance to mean alone."""
    probability = torch.nn.functional.one_hot(pred.long(), classes).double().mean(1)
    actual = torch.nn.functional.one_hot(real.long(), classes).double()
    return (probability[:, None] - actual).square().sum(-1).mean(1)


def _count_brier(pred: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return _categorical_brier(pred.long().sum(-1), real.long().sum(-1), CELLS + 1)


def _event(layout: torch.Tensor, points: list[int], kind: int) -> torch.Tensor:
    local = layout[..., points, kind]
    count = local.long().sum(-1)
    # Geometric ordering is stay, up, down, left, right.
    code = local.long().argmax(-1)
    return torch.where(count == 0, 5, torch.where(count > 1, 6, code))


def _neighbor_cells(source: int) -> list[int]:
    row, col = divmod(source, WIDTH)
    if not (0 < row < HEIGHT - 1 and 0 < col < WIDTH - 1):
        raise ValueError("Source neighborhood crosses the view boundary")
    return [(row + dr) * WIDTH + col + dc for dr, dc in SOURCE_STEPS]


def _source_units(root: torch.Tensor, truth: torch.Tensor, shifts: torch.Tensor | None,
                  actions: torch.Tensor | None) -> tuple[list[dict], dict]:
    units, coverage = [], {"root_count": len(root), "no_scroll_roots": 0,
                          "boundary_sources": 0, "crowded_sources": 0,
                          "DO_excluded_sources": 0, "observed_creation_deletion_sources": 0,
                          "outside_trained_neighborhood_sources": 0,
                          "primary_sources": 0, "far_empirical_sources": 0,
                          "missing_camera_provenance": shifts is None,
                          "missing_action_provenance": actions is None}
    if shifts is None or actions is None:
        return units, coverage
    if shifts.shape != (len(root), truth.shape[1], 2):
        raise ValueError("shifts must be exact real-future camera offsets [N,R,2]")
    for i in range(len(root)):
        if not bool((shifts[i] == 0).all()):
            continue
        coverage["no_scroll_roots"] += 1
        for kind in range(3):
            sources = torch.where(root[i, :, kind])[0].tolist()
            for source in sources:
                r, c = divmod(source, WIDTH)
                if not (0 < r < HEIGHT - 1 and 0 < c < WIDTH - 1):
                    coverage["boundary_sources"] += 1
                    continue
                if any(s != source and abs(s // WIDTH - r) + abs(s % WIDTH - c) <= 2 for s in sources):
                    coverage["crowded_sources"] += 1
                    continue
                # Action.DO is 5 in the verified Craftax Classic enum. This
                # exclusion avoids treating killed mobs as failed conservation.
                if int(actions[i]) == 5:
                    coverage["DO_excluded_sources"] += 1
                    continue
                points = _neighbor_cells(source)
                events = _event(truth[i], points, kind)
                if bool((events >= 5).any()):
                    coverage["observed_creation_deletion_sources"] += 1
                    continue
                near = abs(r - 3) + abs(c - 4) <= 2
                supported = all(p in INTERIOR for p in points)
                if not supported:
                    coverage["outside_trained_neighborhood_sources"] += 1
                primary = near and supported
                coverage["primary_sources" if primary else "far_empirical_sources"] += 1
                units.append({"root": i, "source": source, "kind": kind,
                              "points": points, "primary": primary, "within_scored_domain": supported,
                              "real_events": events})
    return units, coverage


def _collisions(x: torch.Tensor) -> torch.Tensor:
    """Different terrestrial mob types read at one tile; arrows excluded."""
    return (x[..., :3].long().sum(-1) > 1).float().sum(-1).mean(1)


def _root_mean(rows: list[tuple[int, float]], n: int) -> list[float | None]:
    result: list[list[float]] = [[] for _ in range(n)]
    for root, value in rows:
        result[root].append(float(value))
    return [float(np.mean(v)) if v else None for v in result]


def _summary(values: list[float | None]) -> dict:
    defined = [v for v in values if v is not None]
    return {"mean": float(np.mean(defined)) if defined else None, "roots": len(defined)}


def _nearest_local(pred: torch.Tensor, real: torch.Tensor, events: torch.Tensor) -> dict:
    """Same-root, same-source local real features; no whole-frame retrieval."""
    p, r = pred.double().flatten(1), real.double().flatten(1)
    distance = torch.cdist(p, r, compute_mode="donot_use_mm_for_euclid_dist") / np.sqrt(p.shape[-1])
    best, index = distance.min(-1)
    halfway = len(r) // 2
    rr = torch.cdist(r[:halfway], r[halfway:], compute_mode="donot_use_mm_for_euclid_dist") / np.sqrt(p.shape[-1])
    reference = rr.min(-1).values
    cutoff = torch.quantile(reference, .99)
    return {"events": events[index], "distance": best, "real_reference_mean": float(reference.mean()),
            "real_reference_p99": float(cutoff), "beyond_real_p99_fraction": float((best > cutoff).double().mean())}


@torch.no_grad()
def analyze(predictions: torch.Tensor, root_tokens: torch.Tensor, real_tokens: torch.Tensor,
            root_layout: torch.Tensor, real_layout: torch.Tensor, bank_state: dict,
            real_head: dict | None = None, shifts: torch.Tensor | None = None,
            actions: torch.Tensor | None = None, root_tiles: torch.Tensor | None = None,
            real_tiles: torch.Tensor | None = None, sampled_shifts: torch.Tensor | None = None) -> dict:
    """Analyze one arm; returned raw root rows support paired cluster intervals.

    Layout channels are zombie,cow,skeleton,arrow. Real and sampled tensors must
    retain independent branches, [N,K,81,192] / [N,R,81,192]. No head is fitted,
    threshold selected, model loaded, or GPU operation performed here.
    """
    arrays = [predictions, root_tokens, real_tokens, root_layout, real_layout]
    for x in arrays:
        _cpu(x)
    n, k = predictions.shape[:2]
    draws = real_tokens.shape[1]
    if predictions.shape[2:] != (81, 192) or root_tokens.shape != (n, 81, 192):
        raise ValueError("Prediction/root token shapes differ")
    if real_tokens.shape != (n, draws, 81, 192) or real_layout.shape != (n, draws, CELLS, 4):
        raise ValueError("Real fan does not match roots")
    if root_layout.shape != (n, CELLS, 4) or draws < 4 or k < 2:
        raise ValueError("Need [N,63,4] roots and >=4 real / >=2 sampled branches")
    for name, value in (("predictions", predictions), ("real_tokens", real_tokens)):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite {name}")
    generated = decode(predictions, bank_state, real_head)
    real = decode(real_tokens, bank_state, real_head)
    roots = decode(root_tokens, bank_state, real_head)
    estimated_sample_shifts = _camera_shifts(root_tokens, predictions)
    estimated_real_shifts = _camera_shifts(root_tokens, real_tokens)
    if sampled_shifts is not None and not torch.equal(sampled_shifts, estimated_sample_shifts):
        raise ValueError("Supplied sampled shifts disagree with the parameter-free scroll reader")
    sampled_shifts = estimated_sample_shifts
    units, coverage = _source_units(root_layout.bool(), real_layout.bool(), shifts, actions)
    for unit in units:
        i, points, kind = unit["root"], unit["points"], unit["kind"]
        unit["reader_validity"] = {}
        for reader_name, rd in real["mob"].items():
            event = _event(rd[i], points, kind)
            correct = float((event == unit["real_events"]).double().mean())
            root_detected = bool(roots["mob"][reader_name][i, unit["source"], kind])
            unit["reader_validity"][reader_name] = {"event_agreement": correct, "root_source_detected": root_detected}
        chosen = "real" if real_head is not None else "bank"
        row = unit["reader_validity"][chosen]
        unit["reader_clean"] = row["event_agreement"] >= .95 and row["root_source_detected"]
        unit["dual_reader_clean"] = all(row["event_agreement"] >= .95 and row["root_source_detected"]
                                         for row in unit["reader_validity"].values())
    coverage["primary_reader_clean_sources"] = sum(unit["primary"] and unit["reader_clean"] for unit in units)
    coverage["far_reader_clean_sources"] = sum(not unit["primary"] and unit["reader_clean"] for unit in units)
    coverage["primary_dual_reader_clean_sources"] = sum(unit["primary"] and unit["dual_reader_clean"] for unit in units)
    coverage["reader_clean_rule"] = "Primary real ridge reader detects the real root source and matches true joint events on at least 95% of real branches; both-reader clean is a separate sensitivity stratum"
    split = draws // 2
    output = {"format": "direct_joint_coherence_v1", "root_count": n, "samples": k,
              "real_draws": draws, "coverage": coverage, "readers": {}, "raw_per_root": {},
              "event_names": EVENT_NAMES, "layout_types": LAYOUT_TYPES,
              "registered_event_names": EVENT_NAMES + ("camera_mismatch",),
              "scored_map_cells": INTERIOR,
              "primary_joint_distribution_score": "scored_joint_occupancy_energy_reader_reference",
              "whole_map_scores_are_diagnostic_outside_the_scored_domain": True,
              "scope": "One-step frozen-reader and exact-fan evaluation; no actor or four-way uncertainty claim",
              "physical_claim_requirements": ["Independent real-fan reader fidelity must pass", "Both frozen readers must agree",
                  "Source cohorts retain camera-misregistered samples and camera mismatch must pass",
                  "Nearest-real mapping is sensitivity only", "Full-state fan does not bound hidden-history HUD uncertainty"],
              "source_units": [{key: value for key, value in unit.items() if key != "real_events"} for unit in units]}
    exact_truth = real_layout.bool().clone()
    exact_truth[..., PLAYER_CELL, :] = False
    for reader_name, pred in generated["mob"].items():
        rd = real["mob"][reader_name]
        rows: dict[str, list] = {}
        rows["joint_occupancy_energy_reader_reference"] = _energy(pred[..., :3], rd[..., :3]).tolist()
        rows["scored_joint_occupancy_energy_reader_reference"] = _energy(pred[..., INTERIOR, :3], rd[..., INTERIOR, :3]).tolist()
        rows["scored_joint_occupancy_energy_truth_reference"] = _energy(pred[..., INTERIOR, :3], exact_truth[..., INTERIOR, :3]).tolist()
        rows["real_split_scored_joint_occupancy_energy"] = _energy(rd[:, :split, INTERIOR, :3], rd[:, split:, INTERIOR, :3]).tolist()
        rows["joint_occupancy_energy_truth_reference"] = _energy(pred[..., :3], exact_truth[..., :3]).tolist()
        rows["real_split_joint_occupancy_energy"] = _energy(rd[:, :split, :, :3], rd[:, split:, :, :3]).tolist()
        rows["real_split_joint_occupancy_energy_truth"] = _energy(exact_truth[:, :split, :, :3], exact_truth[:, split:, :, :3]).tolist()
        rows["cross_type_collisions"] = _collisions(pred).tolist()
        rows["real_cross_type_collisions"] = _collisions(rd).tolist()
        rows["true_cross_type_collisions"] = _collisions(exact_truth).tolist()
        if shifts is not None:
            no_scroll = (shifts == 0).all((1, 2))
            sampled_wrong = (estimated_sample_shifts != 0).any(-1).double().mean(1)
            real_wrong = (estimated_real_shifts != shifts).any(-1).double().mean(1)
            rows["no_scroll_camera_mismatch_fraction"] = [float(sampled_wrong[i]) if no_scroll[i] else None for i in range(n)]
            rows["real_camera_mismatch_fraction"] = real_wrong.tolist()
        for kind, name in enumerate(LAYOUT_TYPES):
            rows[name + "_count_brier"] = _count_brier(pred[..., kind], rd[..., kind]).tolist()
            rows[name + "_real_split_count_brier"] = _count_brier(rd[:, :split, :, kind], rd[:, split:, :, kind]).tolist()
            rows[name + "_mean_count"] = pred[..., kind].long().sum(-1).double().mean(1).tolist()
            rows[name + "_true_mean_count"] = exact_truth[..., kind].long().sum(-1).double().mean(1).tolist()
            low, high = exact_truth[..., kind].long().sum(-1).amin(1), exact_truth[..., kind].long().sum(-1).amax(1)
            count = pred[..., kind].long().sum(-1)
            rows[name + "_outside_observed_count_range"] = ((count < low[:, None]) | (count > high[:, None])).double().mean(1).tolist()
        all_event_rows: dict[str, list[tuple[int, float]]] = {}
        for unit in units:
            i, points, kind = unit["root"], unit["points"], unit["kind"]
            pred_event = _event(pred[i], points, kind)
            real_event = _event(rd[i], points, kind)
            truth_event = unit["real_events"]
            sample_registered = (estimated_sample_shifts[i] == 0).all(-1)
            real_registered = (estimated_real_shifts[i] == shifts[i]).all(-1)
            sampled_joint_event = torch.where(sample_registered, pred_event, 7)
            real_joint_event = torch.where(real_registered, real_event, 7)
            prefix = "primary_" if unit["primary"] else "far_empirical_"
            metrics = {
                "event_brier_truth": float(_categorical_brier(pred_event[None], truth_event[None], 7)[0]),
                "event_brier_reader": float(_categorical_brier(pred_event[None], real_event[None], 7)[0]),
                "real_split_event_brier": float(_categorical_brier(real_event[None, :split], real_event[None, split:], 7)[0]),
                "registered_event_brier_truth": float(_categorical_brier(sampled_joint_event[None], truth_event[None], 8)[0]),
                "registered_event_brier_reader": float(_categorical_brier(sampled_joint_event[None], real_joint_event[None], 8)[0]),
                "real_split_registered_event_brier": float(_categorical_brier(real_joint_event[None, :split], real_joint_event[None, split:], 8)[0]),
                "exactly_one_destination_fraction": float((pred_event < 5).double().mean()),
                "multiple_destination_fraction": float((pred_event == 6).double().mean()),
                "absent_fraction": float((pred_event == 5).double().mean()),
                "source_and_other_destination_fraction": float((pred[i, :, points[0], kind] & pred[i, :, points[1:], kind].any(-1)).double().mean()),
                "real_multiple_destination_fraction": float((real_event == 6).double().mean()),
                "real_absent_fraction": float((real_event == 5).double().mean()),
                "camera_mismatch_fraction": float((~sample_registered).double().mean()),
                "real_camera_mismatch_fraction": float((~real_registered).double().mean()),
                "valid_joint_and_registered_fraction": float(((pred_event < 5) & sample_registered).double().mean()),
                "distinct_valid_outcomes": float(len(torch.unique(pred_event[(pred_event < 5) & sample_registered]))),
                "true_distinct_valid_outcomes": float(len(torch.unique(truth_event))),
            }
            valid_probability = torch.stack([((pred_event == outcome) & sample_registered).double().mean() for outcome in range(5)])
            valid_mass = valid_probability.sum()
            conditional = valid_probability / valid_mass if valid_mass > 0 else valid_probability
            metrics["valid_outcome_effective_diversity"] = float(torch.exp(-(conditional[conditional > 0] * conditional[conditional > 0].log()).sum())) if valid_mass > 0 else 0.
            metrics["valid_outcome_mass"] = float(valid_probability.sum())
            nearest = _nearest_local(predictions[i, :, points], real_tokens[i, :, points], truth_event)
            metrics.update({"local_nearest_event_disagreement": float((pred_event != nearest["events"]).double().mean()),
                            "local_nearest_distance": float(nearest["distance"].mean()),
                            "local_real_split_nearest_distance": nearest["real_reference_mean"],
                            "local_nearest_beyond_real_p99_fraction": nearest["beyond_real_p99_fraction"]})
            for metric, value in metrics.items():
                all_event_rows.setdefault(prefix + metric, []).append((i, value))
                if unit["reader_clean"]:
                    clean_prefix = "primary_clean_" if unit["primary"] else "far_empirical_clean_"
                    all_event_rows.setdefault(clean_prefix + metric, []).append((i, value))
                if unit["primary"] and unit["dual_reader_clean"]:
                    all_event_rows.setdefault("primary_dual_clean_" + metric, []).append((i, value))
                if unit["primary"] and unit["reader_clean"] and kind == 1:
                    all_event_rows.setdefault("primary_cow_clean_" + metric, []).append((i, value))
            all_event_rows.setdefault(prefix + "real_reader_event_agreement", []).append((i, unit["reader_validity"][reader_name]["event_agreement"]))
        rows.update({name: _root_mean(values, n) for name, values in all_event_rows.items()})
        output["readers"][reader_name] = {"real_validity": _reader_validity(rd, exact_truth),
                                         "root_validity": _reader_validity(roots["mob"][reader_name], root_layout.bool()),
                                         "summary": {name: _summary(values) for name, values in rows.items()}}
        output["raw_per_root"][reader_name] = rows
    # Fixed reader comparison keeps threshold artifacts visible.
    if "real" in generated["mob"]:
        disagreement = (generated["mob"]["real"][..., :3] != generated["mob"]["bank"][..., :3]).double().mean((1, 2, 3))
        baseline = (real["mob"]["real"][..., :3] != real["mob"]["bank"][..., :3]).double().mean((1, 2, 3))
        output["reader_disagreement"] = {"sampled_per_root": disagreement.tolist(), "real_per_root": baseline.tolist()}
    # Semantic stability masks use exact labels when supplied, otherwise this is
    # explicitly a decoder-level reading. Context-sensitive tokens may vary
    # even when these facts agree; variation is not automatically nuisance.
    rt = real["tile"] if real_tiles is None else real_tiles.long()
    old = roots["tile"] if root_tiles is None else root_tiles.long()
    stable_terrain = (rt == old[:, None]).all(1)
    stable_terrain[:, PLAYER_CELL] = False
    stable_mobs = (real_layout.bool() == root_layout[:, None].bool()).all((1, 3))
    stable_facts = stable_terrain & stable_mobs
    scored_stable_facts = stable_facts.clone()
    scored_stable_facts[:, [c for c in range(CELLS) if c not in INTERIOR]] = False
    uncertain_mobs = (real_layout.bool() != real_layout[:, :1].bool()).any((1, 3))
    pred_var = predictions.double().var(1, unbiased=False).mean(-1)
    true_var = real_tokens.double().var(1, unbiased=False).mean(-1)
    masks = {"all": torch.ones(n, 81, dtype=torch.bool),
             "mob_uncertain_map": torch.nn.functional.pad(uncertain_mobs, (0, 18)),
             "semantic_stable_map": torch.nn.functional.pad(stable_facts, (0, 18)),
             "HUD": torch.cat([torch.zeros(n, 63, dtype=torch.bool), torch.ones(n, 18, dtype=torch.bool)], 1)}
    scored_mask = torch.zeros(n, 81, dtype=torch.bool)
    scored_mask[:, INTERIOR] = True
    masks.update({"scored_domain": scored_mask, "outside_scored_domain": ~scored_mask,
                  "scored_mob_uncertain": masks["mob_uncertain_map"] & scored_mask,
                  "scored_semantic_stable": masks["semantic_stable_map"] & scored_mask})
    corrupt = (generated["tile"] != old[:, None]) & stable_facts[:, None]
    real_corrupt = (real["tile"] != old[:, None]) & stable_facts[:, None]
    count = stable_facts.sum(-1)
    raw_corruption = [float(corrupt[i].double().sum() / (k * count[i])) if count[i] else None for i in range(n)]
    raw_real_corruption = [float(real_corrupt[i].double().sum() / (draws * count[i])) if count[i] else None for i in range(n)]
    output["static_content"] = {"definition": "Exact unchanged terrain and mob facts" if real_tiles is not None and root_tiles is not None else "Unchanged decoded terrain plus exact unchanged mob facts",
                                "sampled_corruption_per_root": raw_corruption,
                                "real_reader_corruption_per_root": raw_real_corruption,
                                "eligible_cells_per_root": count.tolist()}
    scored_count = scored_stable_facts.sum(-1)
    scored_corrupt = (generated["tile"] != old[:, None]) & scored_stable_facts[:, None]
    scored_real_corrupt = (real["tile"] != old[:, None]) & scored_stable_facts[:, None]
    output["static_content"]["scored_sampled_corruption_per_root"] = [float(scored_corrupt[i].double().sum() / (k * scored_count[i])) if scored_count[i] else None for i in range(n)]
    output["static_content"]["scored_real_reader_corruption_per_root"] = [float(scored_real_corrupt[i].double().sum() / (draws * scored_count[i])) if scored_count[i] else None for i in range(n)]
    output["static_content"]["scored_eligible_cells_per_root"] = scored_count.tolist()
    output["sampled_variation"] = {"predicted_total_per_root": pred_var.sum(-1).tolist(),
                                   "real_total_per_root": true_var.sum(-1).tolist(),
                                   "masks": {name: mask.tolist() for name, mask in masks.items()},
                                   "scope": "Variance locations, not a learned nuisance/consequence decomposition. Context-dependent real tokens can vary in cells whose local facts remain stable; compare real-fan variation before calling it corruption."}
    for name, mask in masks.items():
        amount = (pred_var * mask).sum(-1)
        reference = (true_var * mask).sum(-1)
        total = pred_var.sum(-1)
        output["sampled_variation"][name] = {"variance_per_root": amount.tolist(), "real_variance_per_root": reference.tolist(),
                                             "share_per_root": [float(amount[i] / total[i]) if total[i] > 0 else None for i in range(n)]}
    output["sampled_camera"] = {"shift_per_root": estimated_sample_shifts.tolist(),
                               "nonzero_fraction_per_root": (estimated_sample_shifts != 0).any(-1).double().mean(1).tolist(),
                               "real_estimated_shift_per_root": estimated_real_shifts.tolist(),
                               "reader": "Existing parameter-free scroll.estimate; same function on real and generated branches",
                               "reader_source_sha256": _sha(SCROLL_SOURCE),
                               "caution": "Camera-misregistered samples remain in all primary source cohorts. Joint conservation claims require the binding camera condition."}
    return output


def selftest() -> dict:
    """Separating invariants, including an equal-marginal incoherent sampler."""
    source = 22
    points = _neighbor_cells(source)
    good = torch.zeros(1, 4, CELLS, 4, dtype=torch.bool)
    good[0, :2, source, 0] = True
    good[0, 2:, source + 1, 0] = True
    bad = torch.zeros_like(good)
    bad[0, :2, source, 0] = True
    bad[0, :2, source + 1, 0] = True
    assert torch.equal(good.float().mean(1), bad.float().mean(1))
    event_good, event_bad = _event(good, points, 0), _event(bad, points, 0)
    assert (event_good < 5).all() and int((event_bad == 6).sum()) == 2 and int((event_bad == 5).sum()) == 2
    bs_good = _categorical_brier(event_good, event_good, 7)
    bs_bad = _categorical_brier(event_bad, event_good, 7)
    assert float(bs_bad[0]) > float(bs_good[0])
    assert float(_energy(bad[..., :3], good[..., :3])[0]) > float(_energy(good[..., :3], good[..., :3])[0])
    root = good[:, 0].clone()
    units, coverage = _source_units(root, good, torch.zeros(1, 4, 2), torch.zeros(1, dtype=torch.long))
    assert len(units) == 1 and coverage["primary_sources"] == 1
    _, missing = _source_units(root, good, None, None)
    assert missing["missing_camera_provenance"] and missing["primary_sources"] == 0
    _, moving = _source_units(root, good, torch.ones(1, 4, 2), torch.zeros(1, dtype=torch.long))
    assert moving["no_scroll_roots"] == 0
    _, attack = _source_units(root, good, torch.zeros(1, 4, 2), torch.full((1,), 5))
    assert attack["DO_excluded_sources"] == 1 and attack["primary_sources"] == 0
    boundary_root = root.clone()
    boundary_root.zero_()
    boundary_root[0, 13, 0] = True  # row1: its row0 destination is untrained.
    boundary_future = boundary_root[:, None].expand(-1, 4, -1, -1)
    _, boundary = _source_units(boundary_root, boundary_future, torch.zeros(1, 4, 2), torch.zeros(1, dtype=torch.long))
    assert boundary["primary_sources"] == 0 and boundary["outside_trained_neighborhood_sources"] == 1
    generator = torch.Generator().manual_seed(20261014)
    observed = torch.randn(1, 81, 192, generator=generator)
    moved = torch.randn(1, 2, 81, 192, generator=generator)
    moved[:, 0] = observed
    # A rightward camera move maps new column c to previous column c+1.
    moved[:, 1, :63].view(1, 7, 9, 192)[:, :, :8] = observed[:, :63].view(1, 7, 9, 192)[:, :, 1:]
    camera = _camera_shifts(observed, moved)
    assert torch.equal(camera, torch.tensor([[[0, 0], [0, 1]]]))
    bad_camera_events = torch.full_like(event_good, 7)
    assert float(_categorical_brier(bad_camera_events, event_good, 8)[0]) > float(bs_good[0])
    return {"status": "passed", "equal_marginal_sampler_rejected": True,
            "valid_event_brier": float(bs_good[0]), "invalid_event_brier": float(bs_bad[0]),
            "camera_and_action_provenance_enforced": True, "scored_domain_enforced": True,
            "parameter_free_camera_reader_checked": True, "misregistered_samples_penalized_not_dropped": True,
            "CPU_only": True}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(value, path: Path, as_json: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w" if as_json else "wb") as stream:
        json.dump(value, stream, indent=2) if as_json else torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_reference(directory: Path | None = None) -> dict:
    """Replay-only metadata and frozen-reader margins, resumable by root.

    Called before either world is trained. Simulator fields stay in a separate
    eval_reference directory, which the observation-only trainer never reads.
    """
    import sys
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "artifacts/eda"))
    import replay as RP
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Reference replay must run on CPU only")
    torch.set_num_threads(4)
    base = root / "artifacts/eda/oracle_v1"
    directory = directory or base / "direct_generator/eval_reference"
    selection_path = base / "direct_generator/data/existing_eval_selection.json"
    selection = json.loads(selection_path.read_text())
    indices = selection["selected_indices"]
    panel = torch.load(base / "panel.pt", map_location="cpu", mmap=True, weights_only=False)
    fan = torch.load(base / "learned_innovation/real_futures/fan.pt", map_location="cpu", mmap=True, weights_only=False)
    assert panel["roots"] == fan["roots"]
    assert selection["panel_sha256"] == _sha(base / "panel.pt")
    metadata_contract = {"format": "direct_eval_replay_metadata_v1", "CPU_only": True,
                         "selection_sha256": _sha(selection_path), "fan_contract_id": fan["contract_id"],
                         "replay_source_sha256": _sha(Path(RP.__file__)), "selected_indices": indices}
    contract_id = hashlib.sha256(json.dumps(metadata_contract, sort_keys=True).encode()).hexdigest()
    env, params, _, _, render = RP.env_and_render()

    def tiles(state):
        top = state.player_position - jnp.array([3, 4]) + 9
        return jax.lax.dynamic_slice(jnp.pad(state.map, 9, constant_values=1), top, (7, 9)).reshape(63)

    @jax.jit
    def fan_tiles(state, action, keys):
        _, futures, _, _, _ = jax.vmap(lambda key: env.step(key, state, action, params))(keys)
        return jax.vmap(tiles)(futures), futures.player_position

    rows = []
    for completed, index in enumerate(indices):
        path = directory / "replay_rows" / f"root_{index:03d}.pt"
        if path.exists():
            row = torch.load(path, map_location="cpu", weights_only=False)
            assert row["contract_id"] == contract_id and row["index"] == index
        else:
            item = panel["roots"][index]
            fields = RP.episode_fields(item["shard"], item["slot"])
            assert fields["split"] == "dev" and fields["episode_id"] == item["episode_id"]
            state = RP.advance_to(item["shard"], item["slot"], item["t"])
            observed = np.asarray(render(state)).astype(np.int32)
            stored = fields["observations"][item["t"]].numpy().astype(np.int32)
            pixel_difference = int(np.abs(observed - stored).max())
            assert pixel_difference == 0
            single = torch.load(base / f"learned_innovation/real_futures/root_{index:03d}.pt", map_location="cpu", weights_only=False)
            assert single["root"] == item and int(single["action"]) == int(panel["actions"][index, 15])
            future_tiles, future_position = fan_tiles(state, jnp.asarray(single["action"]), jnp.asarray(single["keys"].numpy()))
            future_position = torch.from_numpy(np.asarray(future_position).copy())
            assert torch.equal(future_position, single["player_position"])
            position = torch.from_numpy(np.asarray(state.player_position).copy())
            row = {"contract_id": contract_id, "index": index, "root": item,
                   "current_position": position, "future_position": future_position,
                   "shifts": future_position - position[None],
                   "root_tiles": torch.from_numpy(np.asarray(tiles(state)).copy()),
                   "future_tiles": torch.from_numpy(np.asarray(future_tiles).copy()),
                   "recorded_root_replay_pixel_maxdiff": pixel_difference}
            _atomic(row, path)
        rows.append(row)
        if completed % 10 == 0:
            print(json.dumps({"stage": "coherence_reference_replay", "completed": completed + 1, "total": len(indices)}), flush=True)
    metadata = {**metadata_contract, "contract_id": contract_id,
                "roots": [panel["roots"][i] for i in indices],
                "current_positions": torch.stack([r["current_position"] for r in rows]),
                "shifts": torch.stack([r["shifts"] for r in rows]),
                "root_tiles": torch.stack([r["root_tiles"] for r in rows]),
                "real_tiles": torch.stack([r["future_tiles"] for r in rows]),
                "pixel_maxdiff": max(r["recorded_root_replay_pixel_maxdiff"] for r in rows)}
    _atomic(metadata, directory / "metadata.pt")
    bank = torch.load(base / "snap_bank_readers.pt", map_location="cpu", weights_only=False)
    real_head = torch.load(base / "learned_innovation/eval_u3000_h01/mob_heads.pt", map_location="cpu", weights_only=False)["heads"]["real"]
    result = analyze(fan["tokens"][indices, :16], panel["tokens"][indices, 15], fan["tokens"][indices],
                     panel["layouts"][indices, 15], fan["layouts"][indices], bank, real_head,
                     shifts=metadata["shifts"], actions=panel["actions"][indices, 15],
                     root_tiles=metadata["root_tiles"], real_tiles=metadata["real_tiles"])
    real_decoded = decode(fan["tokens"][indices], bank, real_head)
    units = result["source_units"]
    primary_clean = [u for u in units if u["primary"] and u["reader_clean"]]
    varying = [u for u in primary_clean if len(torch.unique(_event(fan["layouts"][indices[u["root"]]], u["points"], u["kind"]))) > 1]
    result["coverage"].update({"primary_clean_independent_roots": len(set(u["root"] for u in primary_clean)),
                               "primary_clean_variable_independent_roots": len(set(u["root"] for u in varying)),
                               "primary_clean_variable_source_units": len(varying),
                               "min20_variable_roots_available": len(set(u["root"] for u in varying)) >= 20,
                               "primary_clean_variable_cow_roots": len(set(u["root"] for u in varying if u["kind"] == 1)),
                               "min20_variable_cow_roots_available": len(set(u["root"] for u in varying if u["kind"] == 1)) >= 20})
    # Fixed, real-only margins: percentile of per-root independent split score
    # disagreement. They measure finite-fan uncertainty, not world improvement.
    rng = np.random.default_rng(20261014)
    floors = {name: [] for name in real_decoded["mob"]}
    for _ in range(64):
        permutation = rng.permutation(32)
        a, b = permutation[:16], permutation[16:]
        for name, flags in real_decoded["mob"].items():
            floors[name].append(_energy(flags[:, a][:, :, INTERIOR, :3], flags[:, b][:, :, INTERIOR, :3]).numpy())
    result["real_only_margins"] = {}
    for name, scores in floors.items():
        array = np.stack(scores)
        result["real_only_margins"][name] = {"real_split_score_mean": float(array.mean()),
                                            "per_root_split_score_p95": np.quantile(array, .95, axis=0).tolist(),
                                            "split_score_mean_ci95": np.quantile(array.mean(1), [.025, .975]).tolist(),
                                            "rule": "64 fixed random 16/16 independent-real partitions; interval across partitions is Monte Carlo sensitivity, not root-cluster uncertainty"}
    result["reference_contract"] = {**metadata_contract, "helper_sha256": _sha(Path(__file__)),
                                    "bank_reader_sha256": _sha(base / "snap_bank_readers.pt"),
                                    "real_reader_sha256": _sha(base / "learned_innovation/eval_u3000_h01/mob_heads.pt"),
                                    "metadata_sha256": _sha(directory / "metadata.pt")}
    result["reference_scope"] = "Real-only pretraining reference; sampled input is first16 real branches, so its full32-reference score is not an independent model score. Independent16/16 scores define reference margins."
    _atomic(result, directory / "coherence_reference.json", True)
    print(json.dumps({"stage": "coherence_reference_complete", "coverage": result["coverage"], "output": str(directory / "coherence_reference.json")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--reference", action="store_true", help="Replay exact evaluation-only metadata and establish real-fan margins")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reference:
        build_reference(args.output)
    elif args.selftest:
        print(json.dumps(selftest(), sort_keys=True))
    else:
        parser.error("Import analyze from the CPU evaluator or use --selftest / --reference")
