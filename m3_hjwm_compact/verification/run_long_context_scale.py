"""Outcome-bearing Step-4b long-context x temporal-scale screen.

The protocol was fixed in ``reviews/2026-07-16-long-context-scale-protocol.md``
before this runner or its monitor bundle was executed.  The default invocation
trains all four registered arms through 2,000 updates.  ``--continue-4k`` is
accepted only when the registered resource gate in the 2k report passed.
"""
from __future__ import annotations

import argparse
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

from consolidation import ARTIFACTS, build_world  # noqa: E402
from data import Episode, EpisodeReplay  # noqa: E402
from fork_oracle_v2 import ENCODER_CKPT, sha256_file  # noqa: E402
from long_context_scale import (  # noqa: E402
    LARGE_MAMBA_DEPTH,
    LARGE_MAMBA_DSTATE,
    LARGE_MAMBA_HEADDIM,
    LARGE_MAMBA_WIDTH,
    LONG_CONTEXT,
    build_long_world,
    evaluate_long_bundle,
    summarize_long_rows,
    temporal_parameter_count,
)
from model import ModelConfig, frozen_dynamics_recipe  # noqa: E402
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from step3_temporal import TRAIN_40K_CACHE, load_scaled_data  # noqa: E402
from step4_runner import (  # noqa: E402
    RESUME_VERSION_KEYS,
    git_head,
    hash_batch,
    shared_state_digest,
    software_versions,
    source_digest,
    tracked_dirty,
)


PROTOCOL = "reviews/2026-07-16-long-context-scale-protocol.md"
MONITOR_BUNDLE = REPO_ROOT / "data" / "long_context_monitor_111_114.pt"
MONITOR_MANIFEST = ARTIFACTS / "long_context_monitor_111_114.manifest.json"
REPORT = ARTIFACTS / "long_context_scale_screen.json"
ARMS = ("LS-G64", "LS-M64", "LL-G", "LL-M")
RUNG_2K = (500, 1_000, 2_000)
TRAIN_SEED = 404


def checkpoint_path(arm: str, step: int) -> Path:
    return ARTIFACTS / f"long_scale_{arm}_{step}.pt"


def rows_path(arm: str, step: int) -> Path:
    return ARTIFACTS / f"long_scale_rows_{arm}_{step}.json"


def _arm_spec(arm: str) -> dict:
    common = {
        "id": arm,
        "training_seed": TRAIN_SEED,
        "observations": LONG_CONTEXT,
        "batch": 1,
        "pooled_global_topology": True,
    }
    if arm == "LS-G64":
        return {**common, "family": "GRU", "scale": "small",
                "width": 64, "depth": 1}
    if arm == "LS-M64":
        return {**common, "family": "Mamba-2", "scale": "small",
                "width": 64, "depth": 1, "d_state": 32,
                "headdim": 16}
    if arm == "LL-G":
        return {**common, "family": "GRU", "scale": "large",
                "width": 524, "depth": 2,
                "parameter_matching": "mechanical closest integer width"}
    if arm == "LL-M":
        return {**common, "family": "Mamba-2", "scale": "large",
                "width": LARGE_MAMBA_WIDTH, "depth": LARGE_MAMBA_DEPTH,
                "d_state": LARGE_MAMBA_DSTATE,
                "headdim": LARGE_MAMBA_HEADDIM}
    raise ValueError(arm)


def _manifest_and_hashes() -> tuple[dict, dict]:
    if not MONITOR_BUNDLE.exists() or not MONITOR_MANIFEST.exists():
        raise RuntimeError(
            "collect and pin the long-context monitor bundle before training")
    manifest = json.loads(MONITOR_MANIFEST.read_text())
    actual = sha256_file(MONITOR_BUNDLE)
    if manifest.get("sha256") != actual:
        raise RuntimeError("long-context monitor hash differs from manifest")
    if (manifest.get("prefix"), manifest.get("suffix"),
            manifest.get("anchors"), manifest.get("seeds"),
            manifest.get("branches"), manifest.get("canonical_collector"),
            manifest.get("live_env_canonical"), manifest.get("verify_repeat")) != (
                LONG_CONTEXT, 8, 24, [111, 112, 113, 114], 3,
                True, True, True):
        raise RuntimeError(f"long-context manifest contract mismatch: {manifest}")
    return manifest, {
        "encoder": sha256_file(ENCODER_CKPT),
        "replay_file": sha256_file(TRAIN_40K_CACHE),
        "monitor_bundle": actual,
        "monitor_manifest": sha256_file(MONITOR_MANIFEST),
    }


def _make_replay(train_data) -> EpisodeReplay:
    replay = EpisodeReplay(capacity_steps=500_000)
    for episode in train_data:
        replay.add(Episode(**episode))
    return replay


def _encoder(device):
    pretrainer = IJEPAPretrainer(ModelConfig(
        temporal_backend="gru", predictor="deterministic", mask_ratio=0.0))
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"], strict=True)
    return pretrainer.target_encoder.to(device).eval()


def _optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _checkpoint_provenance(
    arm, step, world, optimizer, rng, replay_digest, initial_shared_digest,
    hashes, source, versions, histories, monitor, peak_allocated, peak_reserved,
):
    return {
        "head": git_head(),
        "source_digest": source,
        "versions": versions,
        "protocol": PROTOCOL,
        "arm": {**_arm_spec(arm), "steps": step},
        "model_config": dataclasses.asdict(world.cfg),
        "temporal_impl_class": type(world.temporal.impl).__name__,
        "temporal_adapter_name": world.temporal.name,
        "loss_config": dataclasses.asdict(frozen_dynamics_recipe()),
        "hashes": hashes,
        "initial_shared_state_digest": initial_shared_digest,
        "replay_stream_digest": replay_digest,
        "numpy_rng": rng.bit_generator.state,
        "torch_rng_cpu": torch.get_rng_state(),
        "torch_rng_cuda": torch.cuda.get_rng_state(),
        "trainable_parameters": sum(
            parameter.numel() for parameter in world.parameters()
            if parameter.requires_grad),
        "temporal_parameters": temporal_parameter_count(world.temporal.impl),
        "peak_vram_allocated_mib": peak_allocated,
        "peak_vram_reserved_mib": peak_reserved,
        "histories": histories,
        "monitor": monitor,
    }


def _save_checkpoint(path, world, optimizer, provenance):
    torch.save({
        "state_dict": {
            name: value.detach().cpu()
            for name, value in world.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "provenance": provenance,
    }, path)


def _validate_checkpoint(ckpt: dict, arm: str, step: int, hashes: dict,
                         source: str):
    provenance = ckpt.get("provenance", {})
    errors = []
    if provenance.get("head") != git_head():
        errors.append("HEAD mismatch")
    if provenance.get("source_digest") != source:
        errors.append("source digest mismatch")
    if provenance.get("arm") != {**_arm_spec(arm), "steps": step}:
        errors.append("arm specification mismatch")
    if provenance.get("hashes") != hashes:
        errors.append("input hash mismatch")
    if provenance.get("loss_config") != dataclasses.asdict(
            frozen_dynamics_recipe()):
        errors.append("loss configuration mismatch")
    current_versions = software_versions()
    saved_versions = provenance.get("versions", {})
    for key in RESUME_VERSION_KEYS:
        if saved_versions.get(key) != current_versions.get(key):
            errors.append(f"environment drift: {key}")
    if errors:
        raise RuntimeError(f"invalid {arm}@{step} checkpoint: {errors}")


def _evaluate(world, encoder, anchors, arm, step):
    world.eval()
    rows = evaluate_long_bundle(world, encoder, anchors, world.action_input.weight.device)
    summary = summarize_long_rows(rows)
    rows_path(arm, step).write_text(json.dumps(rows, indent=2))
    world.train()
    return summary


def _train_new_arm(
    arm, train_data, reference_shared, anchors, encoder, hashes, source,
    versions, device, report,
):
    replay = _make_replay(train_data)
    rng = np.random.default_rng(TRAIN_SEED)
    replay_hash = hashlib.sha256()
    torch.manual_seed(TRAIN_SEED)
    world = build_long_world(arm, TRAIN_SEED, reference_shared, device).train()
    initial_shared_digest = shared_state_digest(world)
    trainable = [parameter for parameter in world.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    histories: dict[str, list[float]] = {}
    monitor: list[dict] = []
    checkpoints = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for step in range(1, RUNG_2K[-1] + 1):
        batch = replay.sample(
            batch=1, observations=LONG_CONTEXT, device=device, rng=rng)
        hash_batch(replay_hash, batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = world(batch, frozen_dynamics_recipe())
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        world.mark_parameters_updated()
        histories.setdefault("total", []).append(float(output.loss.detach()))
        histories.setdefault("grad_norm", []).append(float(grad_norm))
        for key, value in output.metrics.items():
            histories.setdefault(key, []).append(float(value))
        if not bool(torch.isfinite(output.loss)):
            raise RuntimeError(f"{arm} became non-finite at step {step}")

        if step in RUNG_2K:
            summary = _evaluate(world, encoder, anchors, arm, step)
            monitor.append({"step": step, **summary})
            allocated = torch.cuda.max_memory_allocated() / 2**20
            reserved = torch.cuda.max_memory_reserved() / 2**20
            provenance = _checkpoint_provenance(
                arm, step, world, optimizer, rng, replay_hash.hexdigest(),
                initial_shared_digest, hashes, source, versions, histories,
                monitor, allocated, reserved)
            path = checkpoint_path(arm, step)
            _save_checkpoint(path, world, optimizer, provenance)
            checkpoints[str(step)] = sha256_file(path)
            report["arms"][arm] = {
                "spec": _arm_spec(arm),
                "initial_shared_state_digest": initial_shared_digest,
                "replay_stream_digest": replay_hash.hexdigest(),
                "trainable_parameters": provenance["trainable_parameters"],
                "temporal_parameters": provenance["temporal_parameters"],
                "peak_vram_allocated_mib": allocated,
                "peak_vram_reserved_mib": reserved,
                "elapsed_minutes": (time.perf_counter() - started) / 60,
                "monitor": monitor,
                "checkpoint_sha256": checkpoints,
            }
            REPORT.write_text(json.dumps(report, indent=2))
            print(
                f"[{arm}] {step}: sep={summary['separation_all']:.6f}, "
                f"retrieval={summary['retrieval_tie']:.4f}, "
                f"reserved={reserved:.1f} MiB", flush=True)

    del optimizer, world, replay
    torch.cuda.empty_cache()


def _continue_arm(
    arm, train_data, anchors, encoder, hashes, source, versions, device, report,
):
    start, end = 2_000, 4_000
    path_2k = checkpoint_path(arm, start)
    recorded_hash = report["arms"][arm]["checkpoint_sha256"].get(str(start))
    if recorded_hash != sha256_file(path_2k):
        raise RuntimeError(f"{arm}@2k checkpoint hash differs from report")
    ckpt = torch.load(path_2k, weights_only=False)
    _validate_checkpoint(ckpt, arm, start, hashes, source)
    provenance = ckpt["provenance"]
    world = build_long_world(arm, TRAIN_SEED, {}, device)
    world.load_state_dict(ckpt["state_dict"], strict=True)
    world.train()
    trainable = [parameter for parameter in world.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    optimizer.load_state_dict(ckpt["optimizer"])
    _optimizer_to(optimizer, device)
    rng = np.random.default_rng()
    rng.bit_generator.state = provenance["numpy_rng"]
    torch.set_rng_state(provenance["torch_rng_cpu"])
    torch.cuda.set_rng_state(provenance["torch_rng_cuda"])
    histories = provenance["histories"]
    monitor = provenance["monitor"]
    initial_shared_digest = provenance["initial_shared_state_digest"]
    replay = _make_replay(train_data)
    phase_hash = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for step in range(start + 1, end + 1):
        batch = replay.sample(
            batch=1, observations=LONG_CONTEXT, device=device, rng=rng)
        hash_batch(phase_hash, batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = world(batch, frozen_dynamics_recipe())
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        optimizer.step()
        world.mark_parameters_updated()
        histories.setdefault("total", []).append(float(output.loss.detach()))
        histories.setdefault("grad_norm", []).append(float(grad_norm))
        for key, value in output.metrics.items():
            histories.setdefault(key, []).append(float(value))
        if not bool(torch.isfinite(output.loss)):
            raise RuntimeError(f"{arm} became non-finite at step {step}")

    summary = _evaluate(world, encoder, anchors, arm, end)
    monitor.append({"step": end, **summary})
    allocated = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    continuation_digest = phase_hash.hexdigest()
    continuation_hashes = dict(hashes)
    continuation_hashes["replay_prefix_digest"] = \
        provenance["replay_stream_digest"]
    continuation_hashes["replay_continuation_digest"] = continuation_digest
    final_provenance = _checkpoint_provenance(
        arm, end, world, optimizer, rng, continuation_digest,
        initial_shared_digest, continuation_hashes, source, versions,
        histories, monitor, allocated, reserved)
    path = checkpoint_path(arm, end)
    _save_checkpoint(path, world, optimizer, final_provenance)
    info = report["arms"][arm]
    info["continuation_replay_stream_digest"] = continuation_digest
    info["peak_vram_allocated_mib_4k"] = allocated
    info["peak_vram_reserved_mib_4k"] = reserved
    info["elapsed_minutes_2k_to_4k"] = (time.perf_counter() - started) / 60
    info["monitor"] = monitor
    info["checkpoint_sha256"][str(end)] = sha256_file(path)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"[{arm}] 4000: sep={summary['separation_all']:.6f}, "
          f"retrieval={summary['retrieval_tie']:.4f}", flush=True)
    del optimizer, world, replay
    torch.cuda.empty_cache()


def _at_step(info: dict, step: int) -> dict:
    matches = [entry for entry in info["monitor"] if entry["step"] == step]
    if len(matches) != 1:
        raise RuntimeError(f"expected one monitor entry at {step}: {matches}")
    return matches[0]


def _registered_readout(report: dict, step: int) -> dict:
    result = {arm: _at_step(report["arms"][arm], step) for arm in ARMS}
    delta_small = (
        result["LS-M64"]["separation_all"]
        - result["LS-G64"]["separation_all"])
    delta_large = (
        result["LL-M"]["separation_all"]
        - result["LL-G"]["separation_all"])
    interaction = delta_large - delta_small
    env_deltas = {
        seed: (result["LL-M"]["per_env_seed"][seed]["separation_all"]
               - result["LL-G"]["per_env_seed"][seed]["separation_all"])
        for seed in result["LL-M"]["per_env_seed"]
    }
    retrieval_delta = (
        result["LL-M"]["retrieval_tie"] - result["LL-G"]["retrieval_tie"])
    patch_delta = (
        result["LL-M"]["separation_patch"]
        - result["LL-G"]["separation_patch"])
    threshold = min(
        0.1 * abs(result["LL-G"]["separation_all"]), 0.0005)
    conditions = {
        "finite_and_under_5000_mib": bool(
            report["arms"]["LL-M"].get(
                "peak_vram_reserved_mib_4k",
                report["arms"]["LL-M"]["peak_vram_reserved_mib"])
            < 5000),
        "ll_mamba_positive": result["LL-M"]["separation_all"] > 0,
        "delta_large_positive_at_least_3_of_4_env_seeds": sum(
            value > 0 for value in env_deltas.values()) >= 3,
        "interaction_positive": interaction > 0,
        "minimum_effect": delta_large >= threshold,
        "not_both_secondary_metrics_contradict": not (
            retrieval_delta < 0 and patch_delta < 0),
    }
    return {
        "step": step,
        "primary_separation": {
            arm: result[arm]["separation_all"] for arm in ARMS},
        "delta_small": delta_small,
        "delta_large": delta_large,
        "interaction": interaction,
        "delta_large_by_environment_seed": env_deltas,
        "large_retrieval_tie_delta": retrieval_delta,
        "large_patch_separation_delta": patch_delta,
        "minimum_effect_threshold": threshold,
        "conditions": conditions,
        "licenses_confirmatory_replication": all(conditions.values()),
    }


def _resource_continue_gate(report: dict) -> dict:
    finite = all(
        np.isfinite(report["arms"][arm]["monitor"][-1]["separation_all"])
        for arm in ARMS)
    changes = {
        arm: (_at_step(report["arms"][arm], 2_000)["separation_all"]
              - _at_step(report["arms"][arm], 1_000)["separation_all"])
        for arm in ("LS-M64", "LL-M")
    }
    return {
        "all_arms_finite": finite,
        "mamba_separation_change_1k_to_2k": changes,
        "continue_to_4k": finite and any(value > 0 for value in changes.values()),
    }


def main(continue_4k: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("Step 4b must run on the target CUDA GPU")
    dirty = tracked_dirty()
    if dirty:
        raise RuntimeError("tracked source tree must be clean:\n" + "\n".join(dirty))
    manifest, hashes = _manifest_and_hashes()
    source = source_digest()
    versions = software_versions()
    device = torch.device("cuda")
    train_data, _ = load_scaled_data()
    anchors = torch.load(MONITOR_BUNDLE, weights_only=False)
    encoder = _encoder(device)

    if continue_4k:
        if not REPORT.exists():
            raise RuntimeError("the 2k screen report does not exist")
        report = json.loads(REPORT.read_text())
        if not report.get("resource_continue_gate", {}).get("continue_to_4k"):
            raise RuntimeError("registered 2k -> 4k resource gate did not pass")
        for arm in ARMS:
            _continue_arm(
                arm, train_data, anchors, encoder, hashes, source, versions,
                device, report)
        continuation = {
            report["arms"][arm]["continuation_replay_stream_digest"]
            for arm in ARMS
        }
        if len(continuation) != 1:
            raise RuntimeError("2k -> 4k replay streams diverged")
        report["final_rung"] = 4_000
        report["registered_readout"] = _registered_readout(report, 4_000)
        REPORT.write_text(json.dumps(report, indent=2))
        return

    if REPORT.exists():
        raise RuntimeError(
            f"refusing to overwrite existing outcome report {REPORT}")
    torch.manual_seed(TRAIN_SEED)
    reference = build_world("global_gru", 64, device)
    reference_shared = {
        name: value.detach().cpu().clone()
        for name, value in reference.state_dict().items()
        if not name.startswith("temporal.")
    }
    del reference
    torch.cuda.empty_cache()

    report = {
        "protocol": PROTOCOL,
        "head": git_head(),
        "source_digest": source,
        "versions": versions,
        "tracked_dirty": dirty,
        "manifest": manifest,
        "hashes": hashes,
        "arms": {},
    }
    for arm in ARMS:
        _train_new_arm(
            arm, train_data, reference_shared, anchors, encoder, hashes,
            source, versions, device, report)

    initial_shared = {
        report["arms"][arm]["initial_shared_state_digest"] for arm in ARMS}
    replay_streams = {
        report["arms"][arm]["replay_stream_digest"] for arm in ARMS}
    if len(initial_shared) != 1:
        raise RuntimeError("non-temporal initial states diverged")
    if len(replay_streams) != 1:
        raise RuntimeError("training replay streams diverged")
    report["pairing_checks"] = {
        "shared_initial_state": True,
        "identical_replay_stream": True,
    }
    report["resource_continue_gate"] = _resource_continue_gate(report)
    report["final_rung"] = 2_000
    report["registered_readout"] = _registered_readout(report, 2_000)
    REPORT.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue-4k", action="store_true")
    args = parser.parse_args()
    main(continue_4k=args.continue_4k)
