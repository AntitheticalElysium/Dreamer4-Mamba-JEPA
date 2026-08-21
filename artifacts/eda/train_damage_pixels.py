"""Step 3: the same balanced damage objective, from causal pixel history.

Identical splits, sampler, backbone, head, optimizer and step budget to
`train_damage_classifier`; the only change is the input path. Instead of reading the
frozen Phase-1A latents, a trainable encoder of the same architecture consumes the
raw frames, so the comparison isolates what the frozen representation kept.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import corpus
from train_damage_classifier import DamageHead, Sampler, predict_logits

from d4mj.checkpoint import save
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.representation import Encoder, pack
from d4mj.train import _share_initialisation, optimizer
from d4mj.transition import World, commit_inputs
from resume import load_state, save_state

FRAMES: dict[int, np.ndarray] = {}
ACTIONS: dict[int, np.ndarray] = {}


def gather(chosen, length, off, damage, rows, config: Config):
    frames, actions, labels, valid = [], [], [], []
    for episode, start in chosen:
        window = FRAMES[episode][start : start + length]
        frames.append(torch.from_numpy(np.ascontiguousarray(window)))
        act = np.zeros(length, dtype=np.int64)
        lab = np.zeros(length, dtype=np.float32)
        ok = np.zeros(length, dtype=bool)
        n = rows[episode]["steps"]
        acts = ACTIONS[episode]
        for j in range(length - 1):
            t = start + j
            if t < n:
                act[j + 1] = int(acts[t])
                lab[j] = float(damage[off[episode] + t])
                ok[j] = True
        act[0] = config.n_actions if start == 0 else int(acts[start - 1])
        actions.append(torch.from_numpy(act))
        labels.append(torch.from_numpy(lab))
        valid.append(torch.from_numpy(ok))
    return (torch.stack(frames), torch.stack(actions), torch.stack(labels),
            torch.stack(valid))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=HERE / "damage_pixels")
    parser.add_argument("--milestones", type=int, nargs="+", default=(5_000, 10_000, 20_000))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume-every", type=int, default=500)
    parser.add_argument("--init-phase1a", action="store_true",
                        help="mini-H2: start from the Phase-1A encoder and unfreeze it")
    args = parser.parse_args()

    config = Config(transition="direct", time_mixer="attention")
    rows = corpus.train_rows()
    blob = np.load(HERE / "damage_labels.npz")
    off, damage = blob["offsets"], blob["damage"]
    print(f"corpus {len(rows)} episodes, {int(damage.sum()):,} damaging "
          f"({damage.mean():.3%})", flush=True)

    manifest = json.loads((corpus.SUPPORT / "manifest.json").read_text())
    archive = torch.load(corpus.ARCHIVE, weights_only=False, mmap=True)
    by_shard: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        by_shard.setdefault(row["shard"], []).append(i)
    for shard, indices in sorted(by_shard.items()):
        payload = (torch.load(corpus.SUPPORT / manifest["shards"][shard]["file"],
                              weights_only=False, mmap=True) if shard >= 0 else None)
        for i in indices:
            row = rows[i]
            if payload is not None:
                fields = payload["episodes"][row["slot"]]
                FRAMES[i] = fields["observations"].numpy()
                ACTIONS[i] = fields["actions_taken"].numpy()
            else:
                record = archive[row["slot"]]
                FRAMES[i] = record["obs"][:, :, :63, :63].permute(0, 2, 3, 1).numpy()
                ACTIONS[i] = record["actions"].numpy()
    print(f"frame views mapped for {len(FRAMES)} episodes", flush=True)

    torch.manual_seed(config.seed)
    encoder = Encoder(config).to(config.device)
    if args.init_phase1a:
        from d4mj.checkpoint import load as load_checkpoint

        # Phase 1A was saved under the base config; `checkpoint.load` compares the
        # whole config, so the encoder must be restored under the config it was
        # written with, not the direct-arm one this run trains.
        load_checkpoint(ROOT / "artifacts/stage_a_terminalfix/phase1a.pt", Config(),
                        part0=encoder)
        print("encoder initialized from Phase-1A and left trainable", flush=True)
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(config.device)
    head = DamageHead(config).to(config.device)
    opt = optimizer([encoder, world, head], config)
    rng = torch.Generator(device=config.device).manual_seed(config.seed + 1001)
    sampler = Sampler(rows, off, damage, config, seed=config.seed + 91)

    args.out.mkdir(parents=True, exist_ok=True)
    steps = 100 if args.preflight else args.steps
    started, curve, positives_seen = time.time(), [], 0
    resume_path = args.out / "resume.pt"
    begin, extra = (0, {}) if args.preflight else load_state(
        resume_path, {"encoder": encoder, "world": world, "head": head},
        opt, rng, sampler.rng)
    if begin:
        curve = extra.get("curve", [])
        positives_seen = int(extra.get("positives", 0))
        print(f"resumed from step {begin}", flush=True)
    for step in range(begin, steps):
        finetune = step >= args.steps * (1 - config.long_only_fraction)
        long = finetune or (step + 1) % config.long_batch_every == 0
        length = config.sequence_long if long else config.sequence
        chosen = sampler.draw(step, args.steps, config.batch, length)
        frames, actions, labels, valid = gather(chosen, length, off, damage, rows, config)
        patches = patchify(frames, config.patch).to(config.device)
        actions = actions.to(config.device)
        labels = labels.to(config.device)
        valid = valid.to(config.device)
        z, _, _ = encoder(patches)
        committed, conditioning = commit_inputs(pack(z, config), rng, config)
        features, _, _ = world(None, actions, committed, conditioning)
        logits = predict_logits(world, head, features[:, :-1], actions[:, 1:])
        target = labels[:, : length - 1]
        mask = valid[:, : length - 1].float()
        positive = float((target * mask).sum())
        weight = torch.tensor(
            max(float(mask.sum()) - positive, 1.0) / max(positive, 1.0), device=config.device
        )
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target, weight=mask, pos_weight=weight, reduction="sum"
        ) / mask.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        for group in opt.param_groups:
            group["lr"] = config.learning_rate * min(1.0, (step + 1) / config.warmup)
        torch.nn.utils.clip_grad_norm_(
            [p for m in (encoder, world, head) for p in m.parameters()], config.grad_clip
        )
        opt.step()
        curve.append(float(loss.detach()))
        positives_seen += int(positive)
        if (step + 1) % 250 == 0 or step + 1 == steps:
            window = curve[-250:]
            rate = (step + 1) / (time.time() - started)
            print(f"  {step + 1}/{steps} loss={sum(window)/len(window):.4f} "
                  f"positives={positives_seen:,} "
                  f"[{time.time()-started:.0f}s, {(steps-step-1)/rate:.0f}s left]", flush=True)
        if not args.preflight and (step + 1) in tuple(args.milestones):
            save(args.out / f"model_{step + 1:06d}.pt", config,
                 part0=encoder, part1=world, part2=head)
        if not args.preflight and ((step + 1) % args.resume_every == 0 or step + 1 == steps):
            save_state(resume_path, step + 1,
                       {"encoder": encoder, "world": world, "head": head}, opt, rng,
                       sampler.rng,
                       extra={"curve": curve[-500:], "positives": positives_seen})

    (args.out / ("preflight.json" if args.preflight else "training_report.json")).write_text(
        json.dumps({"steps": steps, "seconds": time.time() - started,
                    "positive_targets_seen": positives_seen,
                    "curve_tail": curve[-250:],
                    "input": ("raw frames, Phase-1A encoder unfrozen" if args.init_phase1a
                              else "raw frames, trainable encoder from scratch")},
                   indent=2))
    print(f"done in {time.time() - started:.0f}s; positives seen {positives_seen:,}")


if __name__ == "__main__":
    main()
