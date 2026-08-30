"""Latents for the actionable terminal roots, under the repaired 64x16 tokenizer.

Same causal contract as `encode_fork_dataset`: the history is encoded once, and each of
the 17 successors is appended to that same history and read at the final block. Roots
without a full 32-frame history are dropped, matching the fork corpus convention -- two
of 3,200.

Output per root: z_history (32, 1024), z_branch (17, 1024), terminated (17,),
lethal_action, safe_actions. The factual arm reads z_branch at `lethal_action`; the
counterfactual arm cycles across all 17.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.representation import Encoder, pack

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"
HISTORY = 32
OUT = HERE / "actionable_latents"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-size", type=int, default=400)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = replace(Config(), n_latents=64, d_bottleneck=16)
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(config).to(DEVICE)
    load(ENCODER, replace(config, batch=stored["batch"], seed=stored["seed"]), part0=encoder)
    encoder.eval()

    rows, shard, started, seen, dropped = [], 0, time.time(), 0, 0
    with torch.no_grad():
        for path in sorted(glob.glob(str(HERE / "actionable_roots" / "shard-*.pt"))):
            for root in torch.load(path, weights_only=False):
                seen += 1
                if root["history"].shape[0] != HISTORY:
                    dropped += 1
                    continue
                history = root["history"]
                z, _, _ = encoder(patchify(history[None], config.patch).to(DEVICE))
                z_history = pack(z, config)[0].flatten(1).cpu()
                branch = []
                for lo in range(0, 17, 4):
                    stacked = torch.stack([torch.cat([history, root["successors"][a][None]])
                                           for a in range(lo, min(lo + 4, 17))])
                    z, _, _ = encoder(patchify(stacked, config.patch).to(DEVICE))
                    branch.append(pack(z, config)[:, -1].flatten(1).cpu())
                rows.append({
                    "shard": root["shard"], "slot": root["slot"], "steps": root["steps"],
                    "z_history": z_history, "z_branch": torch.cat(branch),
                    "terminated": root["terminated"],
                    "lethal_action": root["lethal_action"],
                    "safe_actions": root["safe_actions"], "epsilon": root["epsilon"],
                })
                if len(rows) >= args.shard_size:
                    torch.save(rows, OUT / f"shard-{shard:03d}.pt")
                    shard, rows = shard + 1, []
                    rate = seen / (time.time() - started)
                    print(f"  {seen} roots [{time.time()-started:.0f}s, "
                          f"{(3200-seen)/rate:.0f}s left]", flush=True)
    if rows:
        torch.save(rows, OUT / f"shard-{shard:03d}.pt")
        shard += 1
    (OUT / "manifest.json").write_text(json.dumps(
        {"roots": seen - dropped, "dropped_short_history": dropped, "shards": shard,
         "encoder": str(ENCODER)}, indent=2))
    print(f"wrote {shard} shards, {seen - dropped} roots, dropped {dropped}", flush=True)


if __name__ == "__main__":
    main()
