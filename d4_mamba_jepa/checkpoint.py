"""Strict, atomic checkpoints for the source-pinned D4-lite track."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import tempfile

import torch

from .config import D4LiteConfig
from .model import D4LiteWorld, build_tokenizer
from .source import source_report
from .training import WorldLossNormalizer


FORMAT = "d4_mamba_jepa_world_v1"
TOKENIZER_FORMAT = "d4_mamba_jepa_tokenizer_v1"
PACKAGE_ROOT = Path(__file__).resolve().parent
IMPLEMENTATION_FILES = (
    "__init__.py",
    "config.py",
    "source.py",
    "data.py",
    "model.py",
    "temporal.py",
    "objectives.py",
    "training.py",
    "rollout.py",
    "checkpoint.py",
    "crafter_preflight.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for name in IMPLEMENTATION_FILES:
        path = PACKAGE_ROOT / name
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _cpu_state_dict(module) -> dict:
    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
    }


def _atomic_save(payload: dict, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        torch.save(payload, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return file_sha256(target)


def save_checkpoint(
    path: str | Path,
    *,
    world: D4LiteWorld,
    normalizer: WorldLossNormalizer,
    optimizer=None,
    numpy_rng=None,
    step: int,
    extra: dict | None = None,
) -> str:
    """Write to a sibling temporary file and atomically replace ``path``."""
    target = Path(path)
    payload = {
        "format": FORMAT,
        "config": asdict(world.cfg),
        "world": _cpu_state_dict(world),
        "normalizer": _cpu_state_dict(normalizer),
        "step": int(step),
        "provenance": {
            "sources": source_report(),
            "implementation_sha256": implementation_sha256(),
            "torch": torch.__version__,
        },
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
        payload["rng"] = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_generator": (
                copy.deepcopy(numpy_rng.bit_generator.state)
                if numpy_rng is not None
                else None
            ),
        }

    return _atomic_save(payload, target)


def save_tokenizer_checkpoint(
    path: str | Path,
    *,
    tokenizer,
    config: D4LiteConfig,
    step: int,
    extra: dict | None = None,
) -> str:
    payload = {
        "format": TOKENIZER_FORMAT,
        "config": asdict(config),
        "tokenizer": _cpu_state_dict(tokenizer),
        "step": int(step),
        "provenance": {
            "sources": source_report(),
            "implementation_sha256": implementation_sha256(),
            "torch": torch.__version__,
        },
        "extra": extra or {},
    }
    return _atomic_save(payload, Path(path))


def load_tokenizer_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    expected_config: D4LiteConfig | None = None,
    expected_sha256: str | None = None,
    training_mask: bool = False,
) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = Path(path)
    if expected_sha256 is not None and file_sha256(checkpoint_path) != expected_sha256:
        raise RuntimeError("tokenizer checkpoint digest drift")
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if payload.get("format") != TOKENIZER_FORMAT:
        raise RuntimeError("unsupported tokenizer checkpoint format")
    config = D4LiteConfig(**payload["config"])
    if expected_config is not None and asdict(expected_config) != payload["config"]:
        raise RuntimeError("tokenizer checkpoint config drift")
    if payload["provenance"]["implementation_sha256"] != implementation_sha256():
        raise RuntimeError("tokenizer checkpoint implementation drift")
    if payload["provenance"]["sources"] != source_report():
        raise RuntimeError("tokenizer checkpoint primary-source provenance drift")
    tokenizer = build_tokenizer(config, training_mask=training_mask).to(device)
    tokenizer.load_state_dict(payload["tokenizer"], strict=True)
    return tokenizer, payload


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    expected_config: D4LiteConfig | None = None,
    expected_sha256: str | None = None,
    strict_implementation: bool = True,
) -> tuple[D4LiteWorld, WorldLossNormalizer, dict]:
    checkpoint_path = Path(path)
    if expected_sha256 is not None:
        actual = file_sha256(checkpoint_path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"checkpoint digest drift: {actual} != {expected_sha256}"
            )
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if payload.get("format") != FORMAT:
        raise RuntimeError(f"unsupported checkpoint format {payload.get('format')!r}")
    config = D4LiteConfig(**payload["config"])
    if expected_config is not None and asdict(expected_config) != payload["config"]:
        expected = asdict(expected_config)
        actual = payload["config"]
        differences = {
            key: (expected.get(key), actual.get(key))
            for key in set(expected) | set(actual)
            if expected.get(key) != actual.get(key)
        }
        raise RuntimeError(f"checkpoint config drift: {differences}")
    stored_implementation = payload["provenance"]["implementation_sha256"]
    current_implementation = implementation_sha256()
    if strict_implementation and stored_implementation != current_implementation:
        raise RuntimeError(
            "checkpoint implementation drift: "
            f"{stored_implementation} != {current_implementation}"
        )
    # Re-running source_report hard-fails primary-source drift before weights
    # enter a newly constructed model.
    current_sources = source_report()
    if payload["provenance"]["sources"] != current_sources:
        raise RuntimeError("checkpoint primary-source provenance drift")

    world = D4LiteWorld(config).to(device)
    normalizer = WorldLossNormalizer().to(device)
    world.load_state_dict(payload["world"], strict=True)
    normalizer.load_state_dict(payload["normalizer"], strict=True)
    for name, tensor in world.state_dict().items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"non-finite checkpoint tensor {name}")
    return world, normalizer, payload


def restore_optimizer_and_rng(
    payload: dict,
    *,
    optimizer,
    numpy_rng=None,
) -> None:
    """Restore all-or-refuse preconditions before mutating live state."""
    if "optimizer" not in payload:
        raise RuntimeError("checkpoint carries no optimizer state")
    rng = payload.get("rng", {})
    if rng.get("numpy_generator") is not None and numpy_rng is None:
        raise RuntimeError(
            "checkpoint carries NumPy Generator state; numpy_rng is required"
        )
    if rng.get("torch_cuda_all") is not None and not torch.cuda.is_available():
        raise RuntimeError(
            "checkpoint carries CUDA RNG states but CUDA is unavailable"
        )

    optimizer.load_state_dict(payload["optimizer"])
    if rng.get("torch_cpu") is not None:
        torch.set_rng_state(rng["torch_cpu"])
    if rng.get("torch_cuda_all") is not None:
        torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
    if rng.get("numpy_generator") is not None:
        numpy_rng.bit_generator.state = copy.deepcopy(rng["numpy_generator"])
