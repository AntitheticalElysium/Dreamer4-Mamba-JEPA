"""CPU analysis of the oracle projection diagnostic's raw count evidence.

Usage:
  python snap_analysis.py stats_TRUE.json stats_oracle_deltairis.json \
      stats_control_deltairis.json --out artifacts/eda/oracle_v1/snap_analysis.json

Consumes oracle_snap_stats_v1, not rounded aggregate snap_<world>.json. Metrics
are pooled count ratios. Intervals resample complete episodes and use identical
draws across modes. The primary population includes every root-depth. A shared
scroll-aligned intersection is reported separately as a selected sensitivity
population; it is not a replacement for the primary population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


FORMAT = "oracle_snap_stats_v1"
COUNTS = (
    "true", "predicted", "true_positive", "false_positive", "arrived",
    "arrive_true_positive", "left", "residue",
)
METRICS = {
    "fp_per_true": ("false_positive", "true"),
    "recall": ("true_positive", "true"),
    "census": ("predicted", "true"),
    "arrive_recall": ("arrive_true_positive", "arrived"),
    "residue": ("residue", "left"),
}
GROUPS = {"1": (1,), "2-4": (2, 3, 4), "5-8": (5, 6, 7, 8), "9-16": tuple(range(9, 17))}
CONTRASTS = (
    ("readout", "raw"),
    ("feedback_pre", "raw"),
    ("feedback_projected", "readout"),
    ("feedback_projected", "feedback_pre"),
)


def episode_id(root):
    if "shard" in root and "slot" in root:
        return f"{int(root['shard'])}:{int(root['slot'])}"
    if root.get("episode_id") is not None:
        return str(root["episode_id"])
    raise ValueError("a root needs (shard, slot) or an episode_id")


def load_evidence(path):
    payload = json.loads(path.read_text())
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path}: expected {FORMAT}; aggregate ratios cannot supply paired intervals")
    roots = payload["roots"]
    if not roots:
        raise ValueError(f"{path}: empty roots")
    signature = tuple((episode_id(row), int(row["t"])) for row in roots)
    if len(set(signature)) != len(signature):
        raise ValueError(f"{path}: duplicate episode/root addresses")
    types = tuple(payload["types"])
    depths = tuple(int(x) for x in payload.get("depths", range(1, 17)))
    if len(set(types)) != len(types) or "cow" not in types:
        raise ValueError(f"{path}: mob types must be unique and include cow")
    if depths != tuple(range(1, 17)):
        raise ValueError(f"{path}: expected depth labels 1 through 16")
    shape = (len(roots), len(depths), len(types))
    modes = {}
    for name, mode in payload["modes"].items():
        aligned = np.asarray(mode["aligned"])
        if aligned.shape != shape[:2] or aligned.dtype != np.dtype(bool):
            raise ValueError(f"{path}:{name}: alignment must be a boolean [root, depth] array")
        counts = {}
        for key in COUNTS:
            arr = np.asarray(mode["counts"][key])
            if (arr.shape != shape or arr.dtype.kind not in "iu" or (arr < 0).any()):
                raise ValueError(f"{path}:{name}:{key}: expected nonnegative integer counts {shape}")
            counts[key] = arr.astype(np.int64, copy=False)
        if not np.array_equal(counts["predicted"], counts["true_positive"] + counts["false_positive"]):
            raise ValueError(f"{path}:{name}: predicted != TP + FP")
        for left, right in (("true_positive", "true"), ("arrived", "true"),
                            ("arrive_true_positive", "arrived"), ("arrive_true_positive", "true_positive"),
                            ("residue", "left"), ("residue", "false_positive")):
            if (counts[left] > counts[right]).any():
                raise ValueError(f"{path}:{name}: {left} exceeds {right}")
        modes[name] = {"aligned": aligned, "counts": counts}
    if "raw" not in modes or "readout" not in modes:
        raise ValueError(f"{path}: raw and readout modes are required")
    if ("feedback_pre" in modes) != ("feedback_projected" in modes):
        raise ValueError(f"{path}: both feedback modes must be present together")
    for name, mode in modes.items():
        for key in ("true", "arrived", "left"):
            if not np.array_equal(mode["counts"][key], modes["raw"]["counts"][key]):
                raise ValueError(f"{path}:{name}: truth denominators differ across paired modes")
    if not np.array_equal(modes["raw"]["aligned"], modes["readout"]["aligned"]):
        raise ValueError(f"{path}: readout-only mode must inherit raw trajectory alignment")
    if "feedback_pre" in modes and not np.array_equal(
            modes["feedback_pre"]["aligned"], modes["feedback_projected"]["aligned"]):
        raise ValueError(f"{path}: feedback pre/post modes must share trajectory alignment")
    return {"world": payload["world"], "roots": roots, "signature": signature,
            "types": types, "depths": depths, "modes": modes,
            "source": {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "contract": payload.get("contract"), "hashes": payload.get("hashes")}}


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def interval(samples, clusters):
    valid = np.isfinite(samples)
    # Undefined resamples are never changed into zero or denominator=1. An
    # interval with any such draws would condition on an altered population.
    ci = (np.quantile(samples, [.025, .975]).tolist()
          if clusters >= 2 and valid.all() else None)
    return {"interval95": ci, "defined_draws": int(valid.sum()),
            "undefined_draws": int((~valid).sum()),
            "interval_unavailable_reason": ("fewer than two selected episodes" if clusters < 2 else
                                            "zero denominator in a bootstrap draw" if not valid.all() else None)}


def analyse_world(evidence, draws=2000, seed=20261012):
    ids = np.asarray([episode_id(root) for root in evidence["roots"]])
    clusters, inverse = np.unique(ids, return_inverse=True)
    n_clusters = len(clusters)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=draws)
    modes = evidence["modes"]
    common = np.logical_and.reduce([mode["aligned"] for mode in modes.values()])
    populations = {"all": np.ones_like(common), "common_aligned": common}
    out = {"world": evidence["world"], "source": evidence["source"],
           "roots": len(ids), "episodes": n_clusters, "types": list(evidence["types"]),
           "alignment_by_mode": {name: mode["aligned"].mean(0).tolist() for name, mode in modes.items()},
           "populations": {}}
    scopes = {**GROUPS, **{f"depth_{depth}": (depth,) for depth in evidence["depths"]}}
    for population, mask in populations.items():
        scopes_out = {}
        for scope, depths in scopes.items():
            select = mask[:, np.asarray(depths) - 1]
            selected = select.any(1)
            selected_episodes = len(np.unique(ids[selected]))
            row = {"depths": list(depths), "selected_root_steps": int(select.sum()),
                   "selected_roots": int(selected.sum()), "selected_episodes": selected_episodes,
                   "root_step_share": float(select.mean()), "modes": {}, "contrasts": {}}
            ratio_draws = {}
            for name, mode in modes.items():
                count = {key: (arr[:, np.asarray(depths) - 1] * select[..., None]).sum(1)
                         for key, arr in mode["counts"].items()}
                cluster_count = {}
                for key, arr in count.items():
                    grouped = np.zeros((n_clusters, len(evidence["types"])), dtype=np.int64)
                    np.add.at(grouped, inverse, arr)
                    cluster_count[key] = grouped
                type_out = {}
                for j, mob_type in enumerate(evidence["types"]):
                    totals = {key: int(arr[:, j].sum()) for key, arr in count.items()}
                    metrics = {}
                    for metric, (num, den) in METRICS.items():
                        bn, bd = weights @ cluster_count[num][:, j], weights @ cluster_count[den][:, j]
                        samples = np.full(draws, np.nan)
                        np.divide(bn, bd, out=samples, where=bd > 0)
                        ratio_draws[name, mob_type, metric] = samples
                        metrics[metric] = {"estimate": ratio(totals[num], totals[den]),
                                           "numerator": totals[num], "denominator": totals[den],
                                           **interval(samples, selected_episodes)}
                    type_out[mob_type] = {"counts": totals, "metrics": metrics}
                row["modes"][name] = type_out
            for intervention, baseline in CONTRASTS:
                if intervention not in modes or baseline not in modes:
                    continue
                contrast = {}
                for mob_type in evidence["types"]:
                    metrics = {}
                    for metric in METRICS:
                        left = row["modes"][intervention][mob_type]["metrics"][metric]["estimate"]
                        right = row["modes"][baseline][mob_type]["metrics"][metric]["estimate"]
                        samples = ratio_draws[intervention, mob_type, metric] - ratio_draws[baseline, mob_type, metric]
                        stats = interval(samples, selected_episodes)
                        ci = stats["interval95"]
                        metrics[metric] = {"difference": left - right if left is not None and right is not None else None,
                                           **stats, "excludes_zero": None if ci is None else ci[0] > 0 or ci[1] < 0}
                    contrast[mob_type] = metrics
                row["contrasts"][f"{intervention}_minus_{baseline}"] = contrast
            scopes_out[scope] = row
        out["populations"][population] = scopes_out
    return out


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args(argv)
    if args.draws < 1:
        parser.error("--draws must be positive")
    evidence = [load_evidence(path) for path in args.evidence]
    first = evidence[0]
    for item in evidence[1:]:
        if (item["signature"] != first["signature"] or item["types"] != first["types"] or
                item["depths"] != first["depths"]):
            raise ValueError("all input worlds must use the same ordered roots, depths and types")
        for key in ("true", "arrived", "left"):
            if not np.array_equal(item["modes"]["raw"]["counts"][key], first["modes"]["raw"]["counts"][key]):
                raise ValueError(f"truth counts disagree across worlds: {key}")
    if len({item["world"] for item in evidence}) != len(evidence):
        raise ValueError("world names must be unique")
    payload = {"format": "oracle_snap_analysis_v1",
               "scope": "fixed trained worlds and reader on reused DEV episodes; post hoc mechanism diagnostic, not actor evidence",
               "primary": "all roots and depths 9-16; cow fp_per_true, recall, census",
               "bootstrap": {"unit": "episode", "draws": args.draws, "seed": args.seed,
                             "method": "paired percentile bootstrap of pooled count ratios",
                             "undefined_denominators": "reported, never clamped; interval omitted if any draw is undefined"},
               "common_aligned": "intersection of all mode trajectory-alignment masks within a world; selected sensitivity population",
               "worlds": {item["world"]: analyse_world(item, args.draws, args.seed) for item in evidence}}
    atomic_json(args.out, payload)
    brief = {name: data["populations"]["all"]["9-16"]["contrasts"] for name, data in payload["worlds"].items()}
    print(json.dumps({"output": str(args.out), "late_all_contrasts": brief}, allow_nan=False))


if __name__ == "__main__":
    main()
