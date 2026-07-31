from dataclasses import asdict
from pathlib import Path

import torch

from .config import Config
from .sources import source_digests, verify_sources

FORMAT = "d4mj_checkpoint_v1"


def save(path: Path, config: Config, **objects) -> None:
    """Atomic, and carrying enough to prove what produced it and to resume it: the
    config, the digests of every pinned source a decision rests on, both RNG
    streams, and any plain-dict state such as the running-RMS normalisers."""
    payload = {
        "format": FORMAT,
        "config": asdict(config),
        "sources": source_digests(config),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "modules": {
            name: value.state_dict() if hasattr(value, "state_dict") else value
            for name, value in objects.items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.rename(path)


def load(path: Path, config: Config, **modules) -> dict:
    payload = torch.load(path, weights_only=False)
    if payload["format"] != FORMAT:
        raise ValueError(f"expected {FORMAT}, found {payload['format']}")
    if payload["config"] != asdict(config):
        raise ValueError("checkpoint config differs from the one requested")
    verify_sources(payload["sources"], config)
    torch.set_rng_state(payload["rng"])
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    for name, module in modules.items():
        module.load_state_dict(payload["modules"][name])
    return payload
