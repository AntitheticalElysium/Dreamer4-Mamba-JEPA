"""Production checkpoint save/load for the assembled world model.

2026-07-18 sprint Stage A. Every checkpoint carries the complete ModelConfig
and LossConfig, the frozen-encoder identity, full model state, optional
optimizer/RNG state for resumption, source fingerprints, and component loss
histories. Loading is strict: the stored config rebuilds the model, state
loads with strict=True, and an optional expected-config assertion rejects
silent architecture drift.
"""
from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from pathlib import Path

import torch

from model import LossConfig, M3HJWM, ModelConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=_REPO_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def _model_source_sha256() -> str:
    return hashlib.sha256(
        (Path(__file__).parent / "model.py").read_bytes()).hexdigest()


def _normalized_model_config(values: dict) -> dict:
    """Add explicit defaults to checkpoints written before config axes."""
    output = dict(values)
    output.setdefault("reward_operator", "local_symlog")
    return output


def derived_encoder_digest(world: M3HJWM) -> str:
    """Digest DERIVED from the actual online+target encoder state (2026-07-18
    companion HIGH 3: caller-supplied strings are not provenance)."""
    from model import _encoder_state_digest
    return _encoder_state_digest(world)


def save_world_checkpoint(path, world: M3HJWM, loss_config: LossConfig,
                          optimizer=None,
                          loss_histories: dict | None = None,
                          numpy_rng=None, extra: dict | None = None) -> str:
    payload = {
        "format": "m3_world_checkpoint_v1",
        "model_config": dataclasses.asdict(world.cfg),
        "loss_config": dataclasses.asdict(loss_config),
        "state_dict": {k: v.detach().cpu() for k, v in world.state_dict().items()},
        "temporal_class": type(world.temporal.impl).__name__,
        "predictor_class": type(world.future).__name__,
        "provenance": {
            "head": _git_head(),
            "model_source_sha256": _model_source_sha256(),
            "torch": torch.__version__,
            "encoder_state_sha256": derived_encoder_digest(world),
        },
        "loss_histories": loss_histories or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
        payload["rng"] = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state()
                           if torch.cuda.is_available() else None),
            "numpy": (numpy_rng.bit_generator.state
                      if numpy_rng is not None else None),
        }
    torch.save(payload, path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_world_checkpoint(path, device,
                          expect_config: ModelConfig | None = None,
                          expect_sha256: str | None = None):
    """Rebuild the world from the stored config and load state strict=True.
    Returns (world, payload)."""
    if expect_sha256 is not None:
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expect_sha256:
            raise RuntimeError(
                f"checkpoint hash mismatch: {actual[:16]} != {expect_sha256[:16]}")
    payload = torch.load(path, weights_only=False)
    if payload.get("format") != "m3_world_checkpoint_v1":
        raise RuntimeError("not a m3_world_checkpoint_v1 file")
    stored_config = _normalized_model_config(payload["model_config"])
    cfg = ModelConfig(**stored_config)
    if expect_config is not None and dataclasses.asdict(expect_config) != \
            stored_config:
        expected = dataclasses.asdict(expect_config)
        keys = set(expected) | set(stored_config)
        diffs = {
            key: (expected.get(key), stored_config.get(key))
            for key in keys
            if expected.get(key) != stored_config.get(key)
        }
        raise RuntimeError(f"checkpoint config drift: {diffs}")
    world = M3HJWM(cfg).to(device)
    if type(world.temporal.impl).__name__ != payload["temporal_class"]:
        raise RuntimeError(
            f"temporal class drift: built {type(world.temporal.impl).__name__}, "
            f"checkpoint has {payload['temporal_class']}")
    world.load_state_dict(payload["state_dict"], strict=True)
    for key, tensor in world.state_dict().items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"non-finite checkpoint tensor {key}")
    stored = payload["provenance"].get("encoder_state_sha256")
    if stored is not None and derived_encoder_digest(world) != stored:
        raise RuntimeError(
            "loaded encoder state does not match the checkpoint's derived "
            "encoder digest")
    if payload["provenance"].get("model_source_sha256") != _model_source_sha256():
        payload.setdefault("warnings", []).append(
            "model.py differs from the checkpoint's source; state loaded "
            "strict, but behavior may have changed — verify before reuse")
    return world, payload


def restore_optimizer_and_rng(payload: dict, optimizer, numpy_rng=None) -> None:
    """Explicit resumption (2026-07-18 companion HIGH 3): restore optimizer
    and every RNG state saved by
    save_world_checkpoint(optimizer=..., numpy_rng=...).

    NumPy's Generator state belongs to a particular Generator instance; it
    cannot be restored through the legacy module-global NumPy RNG. Refuse a
    silently partial resume when a checkpoint carries that state but the
    caller omits the corresponding Generator.
    """
    if "optimizer" not in payload:
        raise RuntimeError("checkpoint carries no optimizer state")
    rng = payload.get("rng", {})
    if rng.get("numpy") is not None and numpy_rng is None:
        raise RuntimeError(
            "checkpoint carries NumPy Generator state; pass numpy_rng "
            "to restore_optimizer_and_rng for an exact resume")
    if rng.get("torch_cuda") is not None and not torch.cuda.is_available():
        raise RuntimeError(
            "checkpoint carries CUDA RNG state, but CUDA is unavailable; "
            "exact resumption is impossible")

    optimizer.load_state_dict(payload["optimizer"])
    if rng.get("torch_cpu") is not None:
        torch.set_rng_state(rng["torch_cpu"])
    if rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(rng["torch_cuda"])
    if rng.get("numpy") is not None:
        numpy_rng.bit_generator.state = rng["numpy"]


def sprint_candidate_config(backend: str = "mamba2") -> ModelConfig:
    """The frozen 2026-07-18 sprint contract (see ARCHITECTURE_SPEC banner):
    full-grid, no dense bypass, deterministic predictor, frozen encoder
    usage, standard rollout contract. backend='gru' gives the parameter-
    matched control."""
    return ModelConfig(temporal_topology="full_grid", temporal_backend=backend,
                       predictor="deterministic", mask_ratio=0.0,
                       rollout_steps=2, dense_bypass=False)
