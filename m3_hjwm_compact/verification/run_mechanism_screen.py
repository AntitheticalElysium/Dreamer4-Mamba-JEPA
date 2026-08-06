"""Runner for the 2026-07-17 mechanism screen (factor isolation).

Protocol: reviews/2026-07-17-mechanism-screen-protocol.md. Uses ONLY spent
resources: training seeds 505/606, monitor bundle 131-134. The registered
exploratory readout (cb27d20) is never recomputed; this screen answers the
companion's three ordering questions before any fresh-seed confirmation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import ModelConfig  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from fork_oracle_v2 import ENCODER_CKPT, sha256_file  # noqa: E402
from consolidation import ARTIFACTS, seed_level_summary, symmetric_eval  # noqa: E402
from step4_runner import (  # noqa: E402
    anchor_strata, attach_strata, git_head, software_versions, source_digest,
    tracked_dirty)
from run_exploratory_topology import (  # noqa: E402
    BUNDLE_PATH, MANIFEST_PATH, STEPS, train_arm)
from mechanism_screen import build_mechanism_world  # noqa: E402

REPORT_PATH = ARTIFACTS / "mechanism_screen.json"
SEEDS = (505, 606)
ARM_LIST = tuple((f"{arm}_s{seed}", arm, seed, False)
                 for arm in ("MS-PC", "MS-FB", "MS-FF") for seed in SEEDS)

# Reference values from the committed exploratory report (registered arms,
# seeds 505/606 only, for comparability). Read at runtime, never hand-typed.
EXPLORATORY_REPORT = ARTIFACTS / "exploratory_topology_screen.json"


def tie_retrieval(rows) -> float:
    values = []
    for row in rows:
        matrix = np.array(row["d_all"])
        for s in range(4):
            winners = np.flatnonzero(np.isclose(
                matrix[s], matrix[s].min(), atol=1e-12, rtol=0.0))
            values.append(1.0 / len(winners) if s in winners else 0.0)
    return float(np.mean(values))


def strata_block(rows) -> dict:
    out = {}
    for label, pred in (("day", lambda r: not r["night"]),
                        ("night", lambda r: r["night"]),
                        ("pixel_effective", lambda r: r["pixel_effective"]),
                        ("task_effective", lambda r: r["task_effective"])):
        sub = [r for r in rows if pred(r)]
        out[label] = {
            "n": len(sub),
            "retrieval_all": float(np.mean([r["retrieval_all"] for r in sub]))
            if sub else None,
            "separation_all": float(np.mean([r["separation_all"] for r in sub]))
            if sub else None}
    return out


def ordering_call(arm_mean: float, low_anchor: float, high_anchor: float,
                  high_label: str, low_label: str) -> dict:
    """Preregistered ordering rule: an arm 'reaches' the high anchor if it
    covers >=75% of the gap; it 'stays at' the low anchor if it covers <=25%;
    otherwise unresolved at screen scale."""
    gap = high_anchor - low_anchor
    fraction = (arm_mean - low_anchor) / gap if gap != 0 else float("nan")
    call = (high_label if fraction >= 0.75
            else low_label if fraction <= 0.25 else "unresolved")
    return {"mean": arm_mean, "low_anchor": low_anchor,
            "high_anchor": high_anchor, "gap_fraction": round(fraction, 4),
            "call": call}


def main():
    device = torch.device("cuda")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError("commit before the outcome-bearing run:\n" + "\n".join(dirty))
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert sha256_file(BUNDLE_PATH) == manifest["sha256"], "bundle hash drift"
    anchors = torch.load(BUNDLE_PATH, weights_only=False)
    strata = anchor_strata(anchors)
    train, _ = load_scaled_data()
    pretrainer = IJEPAPretrainer(
        ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0))
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    encoder = pretrainer.target_encoder.to(device).eval()
    from consolidation import build_world

    report = (json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists()
              else {"protocol": "reviews/2026-07-17-mechanism-screen-protocol.md",
                    "arms": {}, "evaluation": {}, "strata": {},
                    "tie_aware_retrieval": {}})
    report["head"] = git_head()
    report["source_digest"] = source_digest()
    report["versions"] = software_versions()
    report["hashes"] = {"encoder": sha256_file(ENCODER_CKPT),
                        "replay_file": sha256_file(TRAIN_40K_CACHE),
                        "bundle": sha256_file(BUNDLE_PATH)}

    references = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        ref = build_world("global_gru", 64, device)
        references[seed] = {n: t.detach().cpu().clone()
                            for n, t in ref.state_dict().items()}
        del ref
        torch.cuda.empty_cache()

    for name, arm, seed, shuffled in ARM_LIST:
        ckpt_path = ARTIFACTS / f"xtopo_{name}_{STEPS}.pt"
        if name in report["arms"] and ckpt_path.exists():
            prov = torch.load(ckpt_path, weights_only=False)["provenance"]
            if (prov["source_digest"] == source_digest()
                    and prov["arm"]["kind"] == arm and prov["arm"]["seed"] == seed
                    and report["arms"][name]["checkpoint_sha256"]
                    == sha256_file(ckpt_path)):
                print(f"[{name}] resume-valid, skipping", flush=True)
                continue
        report["arms"].pop(name, None)
        torch.cuda.reset_peak_memory_stats()
        world, info = train_arm(name, arm, seed, shuffled, train, encoder,
                                references[seed], device,
                                builder=build_mechanism_world)
        info["peak_vram_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / 2**20, 1)
        report["arms"][name] = info
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        del world
        torch.cuda.empty_cache()

    for seed in SEEDS:
        digests = {report["arms"][n]["replay_stream_digest"]
                   for n in report["arms"] if n.endswith(f"_s{seed}")}
        assert len(digests) == 1, f"seed {seed}: replay streams diverge"

    for name, arm, seed, shuffled in ARM_LIST:
        ckpt = torch.load(ARTIFACTS / f"xtopo_{name}_{STEPS}.pt", weights_only=False)
        assert ckpt["provenance"]["source_digest"] == source_digest(), name
        world = build_mechanism_world(arm, seed, device)
        world.load_state_dict(ckpt["state_dict"], strict=True)
        world.eval()
        rows = attach_strata(symmetric_eval(world, encoder, anchors, device), strata)
        (ARTIFACTS / f"xtopo_rows_{name}.json").write_text(json.dumps(rows))
        summary = {k: seed_level_summary(rows, k)
                   for k in ("retrieval_all", "retrieval_changed", "separation_all")}
        report["evaluation"][name] = {
            k: {"mean": v["mean"], "ci95": v["ci95"]} for k, v in summary.items()}
        report["strata"][name] = strata_block(rows)
        report["tie_aware_retrieval"][name] = tie_retrieval(rows)
        del world
        torch.cuda.empty_cache()

    # ---------------- preregistered ordering readout ----------------
    ex = json.loads(EXPLORATORY_REPORT.read_text())
    def sep(name):
        return ex["evaluation"][name]["separation_all"]["mean"]
    fl_mean = float(np.mean([sep(f"X-FL{b}_s{s}") for b in ("G", "M")
                             for s in SEEDS]))
    pooled_mean = float(np.mean([sep(f"base_M1_gru64_s{s}")
                                 for s in (101, 202, 303)]))
    def arm_mean(arm):
        return float(np.mean(
            [report["evaluation"][f"{arm}_s{s}"]["separation_all"]["mean"]
             for s in SEEDS]))
    report["readout"] = {
        "anchors": {"fullgrid_mean_separation": fl_mean,
                    "pooled64_mean_separation": pooled_mean},
        "R1_capacity": ordering_call(
            arm_mean("MS-PC"), pooled_mean, fl_mean,
            high_label="capacity_explains_the_effect",
            low_label="capacity_alone_refuted"),
        "R2_bypass": ordering_call(
            arm_mean("MS-FB"), pooled_mean, fl_mean,
            high_label="bypass_removal_not_required",
            low_label="bypass_removal_load_bearing"),
        "R3_recurrence": ordering_call(
            arm_mean("MS-FF"), pooled_mean, fl_mean,
            high_label="recurrence_not_required_projections_suffice",
            low_label="temporal_state_load_bearing"),
        "note": ("EXPLORATORY factor isolation on spent seeds; primary "
                 "metric = separation_all (the H-T effect); tie-aware "
                 "retrieval + strata recorded per arm; GRU-only (backend at "
                 "parity in both topologies). Confirmation stays HELD until "
                 "the surviving factor is identified.")}
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["readout"], indent=2))


if __name__ == "__main__":
    main()
