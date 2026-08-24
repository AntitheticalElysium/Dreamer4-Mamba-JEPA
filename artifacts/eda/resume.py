"""Crash-safe resume for the diagnostic trainers.

A milestone checkpoint stores modules only, which is enough to evaluate but not to
continue: resuming from one restarts the optimizer moments and the sampler stream.
This stores everything a step depends on -- module states, optimizer state, the step
counter, the model RNG and the numpy draw stream -- and writes it atomically, so an
interrupt at any moment costs at most `every` steps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save_state(path: Path, step: int, modules: dict, optimiser, rng, draw, extra=None):
    payload = {
        "step": step,
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "optimiser": optimiser.state_dict(),
        "model_rng": rng.get_state(),
        "draw": draw.bit_generator.state if draw is not None else None,
        "torch_rng": torch.get_rng_state(),
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_state(path: Path, modules: dict, optimiser, rng, draw) -> tuple[int, dict]:
    """Returns the step to continue from, and whatever `extra` was stored."""
    if not path.exists():
        return 0, {}
    payload = torch.load(path, weights_only=False, map_location="cpu")
    for name, module in modules.items():
        module.load_state_dict(payload["modules"][name])
    optimiser.load_state_dict(payload["optimiser"])
    rng.set_state(payload["model_rng"].to(rng.get_state().device)
                  if hasattr(payload["model_rng"], "to") else payload["model_rng"])
    if draw is not None and payload["draw"] is not None:
        draw.bit_generator.state = payload["draw"]
    torch.set_rng_state(payload["torch_rng"].cpu())
    return int(payload["step"]), payload.get("extra", {})
