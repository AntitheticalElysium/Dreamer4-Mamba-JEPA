"""Latents for the v2 broad corpus, under the frozen 64x16 tokenizer.

Same causal contract as `encode_fork_dataset`: the 32-frame history is encoded once, and
every successor is encoded *appended to that same history*, never in isolation. The second
successor is appended to history + its own first successor, so each latent is read at the
position that actually had the preceding context.

Only branches marked `second_valid` are encoded at the second step. Invalid slots keep a
zero latent and the mask travels with the data, so a consumer cannot accidentally train on
a fabricated post-terminal target.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.representation import Decoder, Encoder, pack

DEVICE = "cuda"
N_ACTIONS = 17
PER_SHARD = 400


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=HERE / "broad_forks_v2")
    parser.add_argument("--out", type=Path, default=HERE / "broad_latents_v2")
    parser.add_argument("--n-latents", type=int, default=64)
    parser.add_argument("--d-bottleneck", type=int, default=16)
    parser.add_argument("--suffix", default="s1")
    parser.add_argument("--milestone", type=int, default=6000)
    parser.add_argument("--chunk", type=int, default=4, help="successors encoded at once")
    args = parser.parse_args()

    folder = HERE / "capacity6k" / f"n{args.n_latents}d{args.d_bottleneck}_{args.suffix}"
    report = json.loads((folder / "training_report.json").read_text())
    config = replace(Config(), n_latents=args.n_latents, d_bottleneck=args.d_bottleneck,
                     batch=report["batch"], seed=report["seed"])
    encoder = Encoder(config).to(DEVICE)
    load(folder / f"encoder_{args.milestone:06d}.pt", config, part0=encoder,
         part1=Decoder(config))
    encoder.eval()

    files = sorted(args.source.glob("seed-*-r*.pt"))
    assert files, f"no collected seeds in {args.source}"
    args.out.mkdir(parents=True, exist_ok=True)
    shard = len(sorted(args.out.glob("shard-*.pt")))
    skip = shard * PER_SHARD
    print(f"{args.n_latents}x{args.d_bottleneck} {args.suffix} @ {args.milestone}, "
          f"z dim {config.n_spatial * config.d_spatial}; {len(files)} seed files"
          + (f"; resuming past {skip:,} roots" if skip else ""), flush=True)

    def encode(frames):
        z, _, _ = encoder(patchify(frames[None], config.patch).to(DEVICE))
        return pack(z, config)[0]

    def encode_last(stack):
        out = []
        for lo in range(0, len(stack), args.chunk):
            z, _, _ = encoder(patchify(torch.stack(stack[lo:lo + args.chunk]),
                                       config.patch).to(DEVICE))
            out.append(pack(z, config)[:, -1].flatten(1).cpu())
        return torch.cat(out)

    rows, seen, started = [], 0, time.time()
    with torch.no_grad():
        for path in files:
            for row in torch.load(path, weights_only=False):
                seen += 1
                if seen <= skip:
                    continue
                history = row["frames"]
                z_history = encode(history).flatten(1).cpu()
                z_branch = encode_last([torch.cat([history, row["successors"][a][None]])
                                        for a in range(N_ACTIONS)])
                valid = row["second_valid"]
                index = torch.nonzero(valid).flatten().tolist()
                z_second = torch.zeros_like(z_branch)
                if index:
                    z_second[index] = encode_last(
                        [torch.cat([history, row["successors"][a][None],
                                    row["second"][a][None]]) for a in index])
                rows.append({
                    "seed": row["seed"], "step": row["step"],
                    "z_history": z_history, "z_branch": z_branch, "z_second": z_second,
                    "led_to_action": row["led_to_action"], "bc_action": row["bc_action"],
                    "second_valid": valid, "terminated": row["terminated"],
                    "truncated": row["truncated"], "reward": row["reward"],
                    "health_delta": row["health_delta"],
                    "achievement_delta": row["achievement_delta"],
                    "second_terminated": row["second_terminated"],
                })
                if len(rows) >= PER_SHARD:
                    target = args.out / f"shard-{shard:04d}.pt"
                    temporary = target.with_suffix(".tmp")
                    torch.save(rows, temporary)
                    temporary.replace(target)
                    rows, shard = [], shard + 1
                    rate = (seen - skip) / (time.time() - started)
                    print(f"  {seen:,} roots, {shard} shards "
                          f"[{time.time()-started:.0f}s, {rate:.1f} roots/s]", flush=True)
    if rows:
        target = args.out / f"shard-{shard:04d}.pt"
        torch.save(rows, target.with_suffix(".tmp"))
        target.with_suffix(".tmp").replace(target)
        shard += 1
    print(f"encoded {seen - skip:,} roots into {shard} shards in "
          f"{time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
