"""Shared contracts for the Phase-1B diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from artifacts.run_stage_a import ARCHIVE, SUPPORT, corpus
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.representation import Encoder
from d4mj.train import cache_latents, cache_latents_to_store


ROOT = Path(__file__).resolve().parent.parent


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def stored_state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def implementation_digests(*paths: Path) -> dict[str, str]:
    common = (
        ROOT / "artifacts" / "run_stage_a.py",
        ROOT / "d4mj" / "agent.py",
        ROOT / "d4mj" / "checkpoint.py",
        ROOT / "d4mj" / "config.py",
        ROOT / "d4mj" / "data.py",
        ROOT / "d4mj" / "expert.py",
        ROOT / "d4mj" / "representation.py",
        ROOT / "d4mj" / "train.py",
        ROOT / "d4mj" / "transition.py",
        Path(__file__).resolve(),
    )
    return {
        str(path.resolve().relative_to(ROOT)): file_digest(path.resolve())
        for path in common + paths
    }


def data_digests(support: Path = SUPPORT) -> dict[str, str | None]:
    support_manifest = (
        support / "manifest.json"
        if support.is_dir()
        else Path(f"{support}.manifest.json")
    )
    return {
        "expert_manifest": file_digest(Path(f"{ARCHIVE}.manifest.json")),
        "support_manifest": (
            file_digest(support_manifest) if support_manifest.exists() else None
        ),
    }


def cached_train(
    phase1a: Path,
    base: Config,
    expert: int,
    log=print,
    *,
    support: Path = SUPPORT,
    cache: Path | None = None,
) -> tuple[Encoder, list]:
    train, _ = corpus(base, expert, log, support=support)
    encoder = Encoder(base).to(base.device)
    load(phase1a, base, part0=encoder)
    encoder.eval()
    if cache is None:
        cached = cache_latents(encoder, train, base)
    else:
        cached = cache_latents_to_store(
            encoder,
            train,
            base,
            cache,
            source_contract={
                "data": data_digests(support),
                "expert": expert,
                "split": "production whole-episode TRAIN",
            },
        )
    return encoder, cached


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
