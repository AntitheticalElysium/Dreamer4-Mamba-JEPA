"""Per-arm latents for the Phase-1B fork diagnostic.

For every damage root: the 32-frame causal history encoded once, and each of the 17
successor frames encoded *appended to that same history*, which is the causal
contract Z* is defined by. Successors are never encoded in isolation.

Both geometries see identical physical states; only the tokenizer differs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.representation import Decoder, Encoder, pack

DEVICE = "cuda"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-latents", type=int, required=True)
    parser.add_argument("--d-bottleneck", type=int, default=16)
    parser.add_argument("--suffix", type=str, default="s1")
    parser.add_argument("--milestone", type=int, default=6000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    folder = HERE / "capacity6k" / f"n{args.n_latents}d{args.d_bottleneck}_{args.suffix}"
    report = json.loads((folder / "training_report.json").read_text())
    config = replace(Config(), n_latents=args.n_latents, d_bottleneck=args.d_bottleneck,
                     batch=report["batch"], seed=report["seed"])
    encoder = Encoder(config).to(DEVICE)
    load(folder / f"encoder_{args.milestone:06d}.pt", config, part0=encoder,
         part1=Decoder(config))
    encoder.eval()
    print(f"arm {args.n_latents}x{args.d_bottleneck} {args.suffix} @ {args.milestone}: "
          f"z dim {config.n_spatial * config.d_spatial}", flush=True)

    histories = {}
    for path in sorted((HERE / "root_frames").glob("shard-*.pt")):
        for row in torch.load(path, weights_only=False):
            histories[(row["seed"], row["step"])] = row
    successors = {}
    for path in sorted((HERE / "fork_successors").glob("shard-*.pt")):
        for row in torch.load(path, weights_only=False):
            successors[(row["seed"], row["step"])] = row
    keys = sorted(set(histories) & set(successors))
    print(f"{len(keys)} roots with both history and successors", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    # resumable: whole shards already on disk are complete, so skip the roots they
    # cover and continue from the next shard index.
    done = sorted(args.out.glob("shard-*.pt"))
    shard = len(done)
    skip = shard * 400
    if skip:
        print(f"resuming: {shard} shards on disk, skipping {skip} roots", flush=True)
    keys = keys[skip:]
    rows, started = [], time.time()
    with torch.no_grad():
        for n, key in enumerate(keys):
            history = histories[key]["frames"]
            branch = successors[key]["successors"]
            # the history itself, teacher-forced targets for the ordinary loss
            z_history, _, _ = encoder(patchify(history[None], config.patch).to(DEVICE))
            z_history = pack(z_history, config)[0].flatten(1).cpu()
            # each successor appended to that same history; take the final latent
            stacked = torch.stack([torch.cat([history, branch[a][None]]) for a in range(17)])
            z_branch, _, _ = encoder(patchify(stacked, config.patch).to(DEVICE))
            z_branch = pack(z_branch, config)[:, -1].flatten(1).cpu()
            rows.append({
                "seed": key[0], "step": key[1],
                "z_history": z_history, "z_branch": z_branch,
                "label": successors[key]["label"],
                "terminated": successors[key]["terminated"],
            })
            if len(rows) >= 400:
                torch.save(rows, args.out / f"shard-{shard:03d}.pt")
                shard, rows = shard + 1, []
            if (n + 1) % 500 == 0:
                rate = (n + 1) / (time.time() - started)
                print(f"  {n+1}/{len(keys)} [{time.time()-started:.0f}s, "
                      f"{(len(keys)-n-1)/rate:.0f}s left]", flush=True)
    if rows:
        torch.save(rows, args.out / f"shard-{shard:03d}.pt")
        shard += 1
    (args.out / "manifest.json").write_text(json.dumps({
        "shards": shard, "roots": len(keys), "n_latents": args.n_latents,
        "d_bottleneck": args.d_bottleneck, "suffix": args.suffix,
        "milestone": args.milestone, "z_dim": config.n_spatial * config.d_spatial}, indent=2))
    print(f"wrote {shard} shards", flush=True)


if __name__ == "__main__":
    main()
