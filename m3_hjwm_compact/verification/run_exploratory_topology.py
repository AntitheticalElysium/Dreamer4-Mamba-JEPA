"""Runner for the 2026-07-17 exploratory topology/conditioning screen.

Protocol: reviews/2026-07-17-exploratory-topology-protocol.md (EXPLORATORY —
screening only; no defaults change; licenses at most a registered
confirmation on fresh seeds).

Arms (T=16, B=4, 16k updates, frozen encoder, frozen_dynamics_recipe — the
exact validated step-4 training contract):
  X-FLG  flattened no-bypass GRU (width matched to X-FLM pre-outcome) x{505,606}
  X-FLM  flattened no-bypass Mamba-2 (256/depth2/state64)             x{505,606}
  X-ADA  global-GRU-64 + AdaLN-zero conditioned predictor             x{505,606}
  X-FLG-shuf, X-ADA-shuf: same-topology shuffled-action controls, seed 505.

Baselines: the twelve committed step-4 16k checkpoints (M1/M2/M3) are
RE-EVALUATED on this screen's fresh monitor bundle — same training contract,
zero extra GPU training cost. Their checkpoints predate this round's source
digest; only their recorded arm config and state shapes are asserted.

Monitor bundle: seeds 131-134 (24 anchors, 4 day/2 night per seed), canonical
collector with verify_repeat, hash-pinned before training. Seeds 115-130
remain reserved/untouched for a potential 4b confirmation.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from data import Episode, EpisodeReplay  # noqa: E402
from model import frozen_dynamics_recipe, ModelConfig  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from fork_oracle_v2 import ENCODER_CKPT, sha256_file  # noqa: E402
from consolidation import ARTIFACTS, build_world, seed_level_summary, symmetric_eval  # noqa: E402
from step4_runner import (  # noqa: E402
    anchor_strata, attach_strata, git_head, hash_batch, shared_state_digest,
    software_versions, source_digest, tracked_dirty)
from exploratory_topology import build_exploratory_world  # noqa: E402

BUNDLE_PATH = REPO_ROOT / "data" / "exploratory_monitor_131_134.pt"
MANIFEST_PATH = ARTIFACTS / "exploratory_monitor_131_134.manifest.json"
REPORT_PATH = ARTIFACTS / "exploratory_topology_screen.json"
SEEDS = (505, 606)
STEPS = 16_000

ARM_LIST = (
    ("X-FLG_s505", "X-FLG", 505, False), ("X-FLG_s606", "X-FLG", 606, False),
    ("X-FLM_s505", "X-FLM", 505, False), ("X-FLM_s606", "X-FLM", 606, False),
    ("X-ADA_s505", "X-ADA", 505, False), ("X-ADA_s606", "X-ADA", 606, False),
    ("X-FLG-shuf_s505", "X-FLG", 505, True),
    ("X-ADA-shuf_s505", "X-ADA", 505, True),
)
# LABELLED POST-REGISTRATION EXTENSION (2026-07-17, after the registered
# readout was computed and committed at cb27d20): one more training seed for
# each flattened arm plus the missing Mamba-side shuffled control. Reported
# in a separate "extension" block; the registered readout is NOT recomputed
# over these arms.
EXTENSION_ARMS = (
    ("X-FLG_s707", "X-FLG", 707, False),
    ("X-FLM_s707", "X-FLM", 707, False),
    ("X-FLM-shuf_s505", "X-FLM", 505, True),
)
STEP4_BASELINES = tuple(
    (f"{family}_s{seed}", backend, 64)
    for family, backend in (("M1_gru64", "global_gru"), ("M2_mamba2", "global_mamba2"),
                            ("M3_gru64_shuf", "global_gru"),
                            ("M3_mamba2_shuf", "global_mamba2"))
    for seed in (101, 202, 303)
)


def collect_bundle():
    from collect_final_79_94 import collect
    anchors = collect(seeds=(131, 132, 133, 134), day_quota=4, night_quota=2,
                      verify_repeat=True)
    torch.save(anchors, BUNDLE_PATH)
    manifest = {
        "bundle": str(BUNDLE_PATH), "sha256": sha256_file(BUNDLE_PATH),
        "anchors": len(anchors), "night": int(sum(a["night"] for a in anchors)),
        "seeds": [131, 132, 133, 134], "branches": 3,
        "canonical_collector": True, "live_env_canonical": True,
        "verify_repeat": True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return anchors


def exclude_prefixes(arm: str) -> tuple[str, ...]:
    """Which state prefixes an arm may legitimately differ in."""
    return ("temporal.", "future.") if arm == "X-ADA" else ("temporal.",)


def train_arm(name, arm, seed, shuffled, train_data, encoder, monitor_ref,
              device):
    replay = EpisodeReplay(capacity_steps=500_000)
    for ep in train_data:
        replay.add(Episode(**ep))
    world = build_exploratory_world(arm, seed, device)
    # pair shared state against the per-seed global-gru reference
    state = world.state_dict()
    skips = exclude_prefixes(arm)
    for key, tensor in monitor_ref.items():
        if any(key.startswith(p) for p in skips):
            continue
        assert key in state and state[key].shape == tensor.shape, key
        state[key] = tensor.to(device=device, dtype=state[key].dtype)
    world.load_state_dict(state, strict=True)
    shared_digest = shared_state_digest(world, exclude_prefix=skips[0]) \
        if len(skips) == 1 else "n/a_predictor_arm"
    weights = frozen_dynamics_recipe()
    trainable = [p for p in world.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    temporal_params = sum(p.numel() for n_, p in world.named_parameters()
                          if n_.startswith("temporal."))
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    rng = np.random.default_rng(seed)
    replay_hash = hashlib.sha256()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        batch = replay.sample(batch=4, observations=16, device=device, rng=rng)
        hash_batch(replay_hash, batch)
        if shuffled:
            batch["actions"] = batch["actions"].roll(1, 0)
            batch["previous_actions"] = batch["previous_actions"].roll(1, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = world(batch, weights)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        world.mark_parameters_updated()
        losses.append(float(out.metrics["jepa"]))
        if step % 4000 == 0:
            print(f"[{name}] {step}: jepa {np.mean(losses[-500:]):.4f}", flush=True)
    minutes = round((time.perf_counter() - started) / 60, 1)
    ckpt_path = ARTIFACTS / f"xtopo_{name}_{STEPS}.pt"
    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in world.state_dict().items()},
         "provenance": {
             "head": git_head(), "source_digest": source_digest(),
             "versions": software_versions(),
             "encoder_sha256": sha256_file(ENCODER_CKPT),
             "replay_file_sha256": sha256_file(TRAIN_40K_CACHE),
             "loss_config": dataclasses.asdict(weights),
             "arm": {"name": name, "kind": arm, "seed": seed,
                     "shuffled": shuffled, "steps": STEPS,
                     "temporal_class": type(world.temporal.impl).__name__,
                     "predictor_class": type(world.future).__name__,
                     "trainable_params": n_params,
                     "temporal_params": temporal_params},
             "shared_state_digest": shared_digest,
             "replay_stream_digest": replay_hash.hexdigest()},
         "loss_history": losses},
        ckpt_path)
    info = {"trainable_params": n_params, "temporal_params": temporal_params,
            "train_minutes": minutes,
            "temporal_class": type(world.temporal.impl).__name__,
            "predictor_class": type(world.future).__name__,
            "shared_state_digest": shared_digest,
            "replay_stream_digest": replay_hash.hexdigest(),
            "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "loss_last500": float(np.mean(losses[-500:]))}
    del replay
    return world, info


def load_step4_baseline(name, backend, hidden, device):
    ckpt = torch.load(ARTIFACTS / f"step4_{name}_16000.pt", weights_only=False)
    arm = ckpt["provenance"]["arm"]
    assert arm["backend"] == backend and arm["global_hidden"] == hidden, name
    world = build_world(backend, hidden, device)
    world.load_state_dict(ckpt["state_dict"], strict=True)
    return world.eval()


def load_xtopo_for_eval(name, arm, seed, device):
    ckpt = torch.load(ARTIFACTS / f"xtopo_{name}_{STEPS}.pt", weights_only=False)
    prov = ckpt["provenance"]
    assert prov["source_digest"] == source_digest(), f"{name}: stale source"
    world = build_exploratory_world(arm, seed, device)
    world.load_state_dict(ckpt["state_dict"], strict=True)
    return world.eval()


def main():
    device = torch.device("cuda")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError("commit before the outcome-bearing run:\n" + "\n".join(dirty))
    if BUNDLE_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        assert sha256_file(BUNDLE_PATH) == manifest["sha256"], "bundle hash drift"
        anchors = torch.load(BUNDLE_PATH, weights_only=False)
    else:
        anchors = collect_bundle()
    strata = anchor_strata(anchors)

    train, _ = load_scaled_data()
    pretrainer = IJEPAPretrainer(
        ModelConfig(temporal_backend="gru", predictor="deterministic", mask_ratio=0.0))
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    encoder = pretrainer.target_encoder.to(device).eval()

    report = (json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists()
              else {"protocol": "reviews/2026-07-17-exploratory-topology-protocol.md",
                    "arms": {}, "baselines": {}})
    report["head"] = git_head()
    report["source_digest"] = source_digest()
    report["versions"] = software_versions()
    report["hashes"] = {"encoder": sha256_file(ENCODER_CKPT),
                        "replay_file": sha256_file(TRAIN_40K_CACHE),
                        "bundle": sha256_file(BUNDLE_PATH)}
    report["strata_counts"] = {
        "night": int(sum(s["night"] for s in strata)),
        "pixel_effective": int(sum(s["pixel_effective"] for s in strata)),
        "task_effective": int(sum(s["task_effective"] for s in strata)),
        "total": len(strata)}

    # per-seed shared reference (global-gru base, matches step-4 pairing)
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
                    and prov["arm"]["shuffled"] == shuffled
                    and report["arms"][name]["checkpoint_sha256"]
                    == sha256_file(ckpt_path)):
                print(f"[{name}] resume-valid, skipping", flush=True)
                continue
        report["arms"].pop(name, None)
        world, info = train_arm(name, arm, seed, shuffled, train, encoder,
                                references[seed], device)
        report["arms"][name] = info
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        del world
        torch.cuda.empty_cache()

    # replay pairing check within each seed (across arms sharing that seed)
    for seed in SEEDS:
        digests = {report["arms"][n]["replay_stream_digest"]
                   for n in report["arms"] if n.endswith(f"_s{seed}")}
        assert len(digests) == 1, f"seed {seed}: replay streams diverge"

    # ---------------- evaluation ----------------
    rows_by_arm = {}
    for name, arm, seed, shuffled in ARM_LIST:
        world = load_xtopo_for_eval(name, arm, seed, device)
        rows = attach_strata(symmetric_eval(world, encoder, anchors, device), strata)
        rows_by_arm[name] = rows
        (ARTIFACTS / f"xtopo_rows_{name}.json").write_text(json.dumps(rows))
        del world
        torch.cuda.empty_cache()
    for name, backend, hidden in STEP4_BASELINES:
        world = load_step4_baseline(name, backend, hidden, device)
        rows = attach_strata(symmetric_eval(world, encoder, anchors, device), strata)
        rows_by_arm[f"base_{name}"] = rows
        (ARTIFACTS / f"xtopo_rows_base_{name}.json").write_text(json.dumps(rows))
        del world
        torch.cuda.empty_cache()

    def summary(rows):
        out = {k: seed_level_summary(rows, k)
               for k in ("retrieval_all", "retrieval_changed", "separation_all")}
        return {k: {"mean": v["mean"], "ci95": v["ci95"]} for k, v in out.items()}

    report["evaluation"] = {n: summary(r) for n, r in rows_by_arm.items()}

    # ---------------- preregistered exploratory readout ----------------
    def mean_ret(names):
        return float(np.mean([report["evaluation"][n]["retrieval_all"]["mean"]
                              for n in names]))
    base_real = mean_ret([f"base_M1_gru64_s{s}" for s in (101, 202, 303)])
    base_shuf = mean_ret([f"base_M3_gru64_shuf_s{s}" for s in (101, 202, 303)]
                         + [f"base_M3_mamba2_shuf_s{s}" for s in (101, 202, 303)])
    flg = [report["evaluation"][f"X-FLG_s{s}"] for s in SEEDS]
    flm = [report["evaluation"][f"X-FLM_s{s}"] for s in SEEDS]
    ada = [report["evaluation"][f"X-ADA_s{s}"] for s in SEEDS]
    fl_control = report["evaluation"]["X-FLG-shuf_s505"]["retrieval_all"]["mean"]
    ada_control = report["evaluation"]["X-ADA-shuf_s505"]["retrieval_all"]["mean"]
    readout = {
        "baseline_global64_retrieval": base_real,
        "baseline_shuffled_retrieval": base_shuf,
        "H_T_flattened": {
            "flg_retrieval": [e["retrieval_all"]["mean"] for e in flg],
            "flm_retrieval": [e["retrieval_all"]["mean"] for e in flm],
            "fl_separations_positive": bool(
                all(e["separation_all"]["mean"] > 0 for e in flg + flm)),
            "fl_shuffled_control": fl_control,
            "interesting": bool(
                float(np.mean([e["retrieval_all"]["mean"] for e in flg + flm]))
                >= base_real
                and all(e["separation_all"]["mean"] > 0 for e in flg + flm)
                and float(np.mean([e["retrieval_all"]["mean"] for e in flg + flm]))
                >= fl_control + 0.015),
        },
        "H_C_adaln": {
            "ada_retrieval": [e["retrieval_all"]["mean"] for e in ada],
            "ada_shuffled_control": ada_control,
            "interesting": bool(
                float(np.mean([e["retrieval_all"]["mean"] for e in ada]))
                >= base_real + 0.01
                and all(e["retrieval_all"]["mean"] > base_real for e in ada)
                and float(np.mean([e["retrieval_all"]["mean"] for e in ada]))
                >= ada_control + 0.015),
        },
        "note": ("EXPLORATORY: screening thresholds only; any 'interesting' "
                 "outcome licenses a registered confirmation on fresh seeds, "
                 "never a default change. Seeds 131-134 are spent for "
                 "selection after this screen."),
    }
    report["readout"] = readout
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(readout, indent=2))


def run_extension():
    """Train/evaluate EXTENSION_ARMS only; report under a separate
    'extension' block. The committed registered readout is left untouched."""
    device = torch.device("cuda")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError("commit before the extension run:\n" + "\n".join(dirty))
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
    report = json.loads(REPORT_PATH.read_text())
    extension = report.setdefault("extension", {"arms": {}, "evaluation": {}})
    references = {}
    for seed in sorted({seed for _, _, seed, _ in EXTENSION_ARMS}):
        torch.manual_seed(seed)
        ref = build_world("global_gru", 64, device)
        references[seed] = {n: t.detach().cpu().clone()
                            for n, t in ref.state_dict().items()}
        del ref
        torch.cuda.empty_cache()
    for name, arm, seed, shuffled in EXTENSION_ARMS:
        if name in extension["arms"] \
                and (ARTIFACTS / f"xtopo_{name}_{STEPS}.pt").exists():
            continue
        world, info = train_arm(name, arm, seed, shuffled, train, encoder,
                                references[seed], device)
        extension["arms"][name] = info
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
        del world
        torch.cuda.empty_cache()
    for name, arm, seed, shuffled in EXTENSION_ARMS:
        world = load_xtopo_for_eval(name, arm, seed, device)
        rows = attach_strata(symmetric_eval(world, encoder, anchors, device), strata)
        (ARTIFACTS / f"xtopo_rows_{name}.json").write_text(json.dumps(rows))
        out = {k: seed_level_summary(rows, k)
               for k in ("retrieval_all", "retrieval_changed", "separation_all")}
        extension["evaluation"][name] = {
            k: {"mean": v["mean"], "ci95": v["ci95"]} for k, v in out.items()}
        del world
        torch.cuda.empty_cache()
    extension["note"] = (
        "post-registration extension (committed AFTER the registered readout "
        "at cb27d20): consistency data only, never merged into the "
        "registered H-T/H-C criteria")
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(extension["evaluation"], indent=2))


if __name__ == "__main__":
    run_extension() if "--extension" in sys.argv else main()
