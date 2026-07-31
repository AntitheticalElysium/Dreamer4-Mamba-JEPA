from dataclasses import asdict
from pathlib import Path

import torch

from .config import Config
from .sources import source_digests, verify_sources

FORMAT = "d4mj_checkpoint_v1"


def save(path: Path, config: Config, **modules) -> None:
    """Atomic, and carrying enough to prove what produced it: the config, the
    digests of every pinned source a decision rests on, and the RNG state."""
    payload = {
        "format": FORMAT,
        "config": asdict(config),
        "sources": source_digests(config),
        "rng": torch.get_rng_state(),
        "modules": {name: module.state_dict() for name, module in modules.items()},
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
    for name, module in modules.items():
        module.load_state_dict(payload["modules"][name])
    return payload
