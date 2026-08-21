"""The production Phase-1B TRAIN corpus, rebuilt in the exact order the cache holds.

`run_stage_a.corpus` appends archive TRAIN first -- in `episode_splits` *permutation*
order, not sorted index order -- then support TRAIN in on-disk order. Verified
element-by-element against the 261-shard latent cache.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
CACHE = ROOT / "artifacts/terminal_diversity_v2/train_latent_cache"
SUPPORT = ROOT / "artifacts/craftax_support_v2"
ARCHIVE = ROOT / "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt"


@functools.cache
def train_rows() -> list[dict]:
    import sys

    sys.path.insert(0, str(ROOT))
    from d4mj.config import Config
    from d4mj.data import episode_splits

    config = Config()
    archive = torch.load(ARCHIVE, weights_only=False, mmap=True)
    train_idx, _, _ = episode_splits(len(archive), config.seed + 0)
    rows = []
    for slot in train_idx.tolist():
        record = archive[slot]
        rows.append(dict(source="expert", shard=-1, slot=slot,
                         steps=len(record["actions"]), epsilon=None, bc_eligible=True))
    manifest = json.loads((SUPPORT / "manifest.json").read_text())
    for shard_index, record in enumerate(manifest["shards"]):
        payload = torch.load(SUPPORT / record["file"], weights_only=False, mmap=True)
        for slot, fields in enumerate(payload["episodes"]):
            if fields["split"] != "train":
                continue
            rows.append(dict(source="support", shard=shard_index, slot=slot,
                             steps=len(fields["actions_taken"]),
                             epsilon=float(fields["epsilon"]), bc_eligible=False))
        del payload
    return rows


def offsets(rows: list[dict]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum([r["steps"] for r in rows])])


def frame_offsets(rows: list[dict]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum([r["steps"] + 1 for r in rows])])


def frames_of(row: dict) -> np.ndarray:
    if row["source"] == "support":
        manifest = json.loads((SUPPORT / "manifest.json").read_text())
        payload = torch.load(SUPPORT / manifest["shards"][row["shard"]]["file"],
                             weights_only=False, mmap=True)
        return payload["episodes"][row["slot"]]["observations"].numpy()
    archive = torch.load(ARCHIVE, weights_only=False, mmap=True)
    return archive[row["slot"]]["obs"][:, :, :63, :63].permute(0, 2, 3, 1).numpy()


def verify_order(rows: list[dict]) -> None:
    manifest = json.loads((CACHE / "manifest.json").read_text())
    lengths = []
    for record in manifest["shards"]:
        payload = torch.load(CACHE / record["file"], weights_only=False, mmap=True)
        lengths += [f["latents"].shape[0] - 1 for f in payload["episodes"]]
        del payload
    expected = [r["steps"] for r in rows]
    if lengths != expected:
        raise AssertionError("corpus order does not match the production latent cache")


def iter_cached_latents():
    """(episode index, latents) in production order, memory mapped."""
    manifest = json.loads((CACHE / "manifest.json").read_text())
    index = 0
    for record in manifest["shards"]:
        payload = torch.load(CACHE / record["file"], weights_only=False, mmap=True)
        for fields in payload["episodes"]:
            yield index, fields["latents"]
            index += 1
        del payload
