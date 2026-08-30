"""Replace the locally-encoded history with the production cached latents.

Encoding only the last 32 raw frames leaves the final latent exact but gives the earlier
positions in that window no preceding encoder context of their own, so they are not the
latents the world model is trained against. The production cache already holds the exact
ones, computed with memory carried across the whole episode.

The cache *does* preserve `episode_id` -- an earlier note here said otherwise, from
checking index 0, which is an archive episode carrying none. The mapping is recovered
positionally from the corpus and then confirmed two ways: the final history latent agrees
with the locally encoded one, the position that had full context either way, and an
id-keyed lookup returns bit-identical windows.

Only `z_history` is replaced. The 17 branch targets keep their fresh encoding, since each
was produced by appending a successor to the same 32-frame window and reading the final
block -- the position that is exact.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts"))
sys.path.insert(0, str(HERE))

from d4mj.config import Config
from d4mj.data import load_episodes

HISTORY = 32
CACHE = HERE / "latent_cache_64"
LATENTS = HERE / "actionable_latents"


def main() -> None:
    base = replace(Config(), n_latents=64, d_bottleneck=16)
    os.chdir(ROOT)
    from run_stage_a import corpus

    train, _ = corpus(base, 320, lambda m: None,
                      support=ROOT / "artifacts/craftax_support_v2")
    index = {}
    for position, episode in enumerate(train):
        identity = getattr(episode, "episode_id", None)
        if identity and str(identity).startswith("support-v2"):
            _, _, shard, slot = str(identity).split(":")
            index[(int(shard), int(slot))] = position
    print(f"{len(index):,} support train episodes mapped by (shard, slot)", flush=True)

    digest = json.loads((CACHE / "manifest.json").read_text())["cache_digest"]
    cached = load_episodes(CACHE, digest, verify=False)
    assert len(cached) == len(train), f"cache holds {len(cached)}, corpus has {len(train)}"

    worst, replaced = 0.0, 0
    for path in sorted(glob.glob(str(LATENTS / "shard-*.pt"))):
        rows = torch.load(path, weights_only=False)
        for row in rows:
            key = (int(row["shard"]), int(row["slot"]))
            position = index[key]
            latents = cached[position].latents
            steps = int(row["steps"])
            assert len(latents) >= steps, f"{key}: {len(latents)} latents, steps {steps}"
            window = latents[steps - HISTORY : steps].flatten(1).float()
            assert window.shape == row["z_history"].shape, (
                f"{key}: {tuple(window.shape)} against {tuple(row['z_history'].shape)}")
            # the final position had full context in both encodings, so it must agree
            worst = max(worst, float((window[-1] - row["z_history"][-1]).abs().max()))
            row["z_history_local"] = row["z_history"]
            row["z_history"] = window
            replaced += 1
        torch.save(rows, path)
    print(f"replaced {replaced:,} histories with production latents", flush=True)
    print(f"worst disagreement at the final position: {worst:.3e} "
          f"(that position had full context in both encodings)", flush=True)

    manifest = json.loads((LATENTS / "manifest.json").read_text())
    manifest["z_history_source"] = "latent_cache_64, aligned positionally by episode_id"
    manifest["final_position_max_abs_difference"] = worst
    (LATENTS / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
