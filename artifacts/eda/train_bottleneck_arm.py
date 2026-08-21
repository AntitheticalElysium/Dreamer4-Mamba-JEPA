"""Phase-1A only, at a chosen bottleneck geometry. No dynamics, no latent cache.

Mirrors `train.train_representation` exactly -- same MAE objective, masking, LPIPS
weight, running-RMS balancing, optimizer, warmup, clipping and sampler stream -- and
differs only in that it stops at the encoder, records milestones, and takes
`n_latents`, `d_bottleneck`, `batch` and the representation seed as arguments.
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
import corpus

from d4mj.checkpoint import save
from d4mj.config import Config
from d4mj.data import Episode, EpisodeCorpus, sample_batch
from d4mj.representation import Decoder, Encoder, reconstruction_loss
from d4mj.train import _balance, _generators, _to, _update, optimizer
from resume import load_state, save_state


def build_corpus() -> EpisodeCorpus:
    rows = corpus.train_rows()
    manifest = json.loads((corpus.SUPPORT / "manifest.json").read_text())
    archive = torch.load(corpus.ARCHIVE, weights_only=False, mmap=True)
    by_shard: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        by_shard.setdefault(row["shard"], []).append(i)
    episodes: list[Episode | None] = [None] * len(rows)
    for shard, indices in sorted(by_shard.items()):
        payload = (torch.load(corpus.SUPPORT / manifest["shards"][shard]["file"],
                              weights_only=False, mmap=True) if shard >= 0 else None)
        for i in indices:
            row = rows[i]
            if payload is not None:
                fields = payload["episodes"][row["slot"]]
                frames, actions = fields["observations"], fields["actions_taken"]
            else:
                record = archive[row["slot"]]
                frames = record["obs"][:, :, :63, :63].permute(0, 2, 3, 1)
                actions = record["actions"]
            n = len(actions)
            episodes[i] = Episode(
                observations=frames, actions_taken=actions, rewards=torch.zeros(n),
                terminated=torch.zeros(n, dtype=torch.bool),
                truncated=torch.zeros(n, dtype=torch.bool),
                events=torch.zeros(n, dtype=torch.bool))
    return EpisodeCorpus(episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-latents", type=int, required=True)
    parser.add_argument("--d-bottleneck", type=int, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--milestones", type=int, nargs="+", default=(500, 1500, 3000))
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-every", type=int, default=500)
    args = parser.parse_args()

    base = Config()
    config = replace(base, n_latents=args.n_latents, d_bottleneck=args.d_bottleneck,
                     batch=args.batch, seed=base.seed + args.seed_offset)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"arm {args.n_latents}x{args.d_bottleneck} batch {args.batch} "
          f"seed {config.seed}: {args.n_latents*args.d_bottleneck} scalars/frame, "
          f"n_spatial {config.n_spatial}", flush=True)

    import lpips

    episodes = build_corpus()
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(config.device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)

    torch.manual_seed(config.seed)
    encoder, decoder = Encoder(config).to(config.device), Decoder(config).to(config.device)
    opt = optimizer([encoder, decoder], config)
    sampler, rng = _generators(config, 0)
    balance: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats()

    resume_path = args.out / "resume.pt"
    begin, extra = load_state(resume_path, {"encoder": encoder, "decoder": decoder},
                              opt, rng, None)
    curve, started = list(extra.get("curve", [])), time.time()
    if begin:
        balance.update(extra.get("balance", {}))
        sampler.set_state(torch.tensor(extra["sampler"], dtype=torch.uint8))
        print(f"resumed from step {begin}", flush=True)
    for step in range(begin, args.steps):
        batch = _to(sample_batch(episodes, sampler, config, step, args.steps), config.device)
        z, _, masked = encoder(batch.patches, p_mask=config.mae_p_max, rng=rng)
        predicted, _ = decoder(z)
        losses = reconstruction_loss(predicted, batch.patches, masked, batch.scored,
                                     perceptual, config)
        loss = _balance(losses, balance, config, {"lpips": config.lpips_weight})
        _update(opt, loss, [encoder, decoder], config, step)
        curve.append({"step": step + 1, "mse": float(losses["mse"].detach()),
                      "lpips": float(losses["lpips"].detach())})
        if (step + 1) % 250 == 0 or step + 1 == args.steps:
            window = curve[-250:]
            rate = (step + 1) / (time.time() - started)
            print(f"  {step+1}/{args.steps} mse={sum(c['mse'] for c in window)/len(window):.5f} "
                  f"lpips={sum(c['lpips'] for c in window)/len(window):.5f} "
                  f"[{time.time()-started:.0f}s, {(args.steps-step-1)/rate:.0f}s left]",
                  flush=True)
        if (step + 1) in tuple(args.milestones):
            save(args.out / f"encoder_{step+1:06d}.pt", config, part0=encoder, part1=decoder)
        if (step + 1) % args.resume_every == 0 or step + 1 == args.steps:
            save_state(resume_path, step + 1, {"encoder": encoder, "decoder": decoder},
                       opt, rng, None,
                       extra={"curve": curve, "balance": dict(balance),
                              "sampler": sampler.get_state().tolist()})

    peak = torch.cuda.max_memory_allocated() / 2**30
    (args.out / "training_report.json").write_text(json.dumps({
        "n_latents": args.n_latents, "d_bottleneck": args.d_bottleneck,
        "scalars_per_frame": args.n_latents * args.d_bottleneck,
        "batch": args.batch, "seed": config.seed, "steps": args.steps,
        "peak_vram_gib": peak, "seconds": time.time() - started, "curve": curve,
    }, indent=2))
    print(f"done in {time.time()-started:.0f}s, peak {peak:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
