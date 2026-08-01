from dataclasses import asdict
from pathlib import Path

import torch

from .config import Config
from .sources import source_digests, verify_sources

FORMAT = "d4mj_checkpoint_v1"


def save(path: Path, config: Config, **objects) -> None:
    """Atomic, and carrying enough to prove what produced it and to resume it: the
    config, the digests of every pinned source a decision rests on, and any
    plain-dict state -- the running-RMS normalisers and, via
    `train.generator_state`, the sampler and model generators that actually drive
    training. The global stream is stored too, but nothing here draws from it."""
    payload = {
        "format": FORMAT,
        "config": asdict(config),
        "sources": source_digests(config),
        "rng": torch.get_rng_state(),
        "modules": {
            name: value.state_dict() if hasattr(value, "state_dict") else value
            for name, value in objects.items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.rename(path)


def load(path: Path, config: Config, **objects) -> dict:
    """Restores modules through `load_state_dict` and plain state -- the running-RMS
    dicts among them -- by replacing their contents in place."""
    payload = torch.load(path, weights_only=False)
    if payload["format"] != FORMAT:
        raise ValueError(f"expected {FORMAT}, found {payload['format']}")
    if payload["config"] != asdict(config):
        raise ValueError("checkpoint config differs from the one requested")
    verify_sources(payload["sources"], config)
    torch.set_rng_state(payload["rng"])
    for name, target in objects.items():
        stored = payload["modules"][name]
        if hasattr(target, "load_state_dict"):
            target.load_state_dict(stored)
        else:
            target.clear()
            target.update(stored)
    return payload
