"""Test 2: the same classifier, trained on explicit all-action roots.

Everything is held to `train_damage_classifier`: frozen Z*, the production
Direct-Attention backbone from its registered initialization, the same head,
optimizer, batch, short/long schedule and 20,000-step budget. The only change is
what a training row is. Instead of one factual transition with its observed action,
a row is one collected root with all 17 of its simulator-executed outcomes, so the
objective sees the state x action interaction directly.

Diagnostic only: no production data or sampler is touched.
"""

from __future__ import annotations

import argparse
import hashlib
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
from train_damage_classifier import DamageHead, predict_logits

from d4mj.checkpoint import save
from d4mj.config import Config
from d4mj.train import _share_initialisation, optimizer
from d4mj.transition import World, commit_inputs


def split_of(record) -> str:
    key = f"consequence-probe:{record['shard']}:{record['slot']}:{record['t']}"
    draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % 10
    return "fit" if draw < 6 else ("tune" if draw < 8 else "test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=HERE / "damage_allaction")
    parser.add_argument("--milestones", type=int, nargs="+", default=(5_000, 10_000, 20_000))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    config = Config(transition="direct", time_mixer="attention")
    rows = corpus.train_rows()
    lookup = {(r["shard"], r["slot"]): i for i, r in enumerate(rows) if r["source"] == "support"}
    cached = {i: v for i, v in corpus.iter_cached_latents()}
    manifest = json.loads((corpus.SUPPORT / "manifest.json").read_text())

    records = []
    for path in sorted((HERE / "latent_forks").glob("shard-*.pt")):
        records += torch.load(path, weights_only=False)
    action_cache: dict[int, dict] = {}
    fit = []
    for record in records:
        if split_of(record) != "fit":
            continue
        shard = record["shard"]
        if shard not in action_cache:
            payload = torch.load(corpus.SUPPORT / manifest["shards"][shard]["file"],
                                 weights_only=False, mmap=True)
            action_cache[shard] = {slot: f["actions_taken"].numpy()
                                   for slot, f in enumerate(payload["episodes"])}
            del payload
        acts = action_cache[shard][record["slot"]]
        health, dead = record["health"].numpy(), record["dead"].numpy()
        fit.append({
            "episode": lookup[(shard, record["slot"])], "t": record["t"], "acts": acts,
            "label": ((health <= -1) | dead).astype(np.float32),
        })
    labels = np.stack([row["label"] for row in fit])
    print(f"{len(fit)} fit roots, {int(labels.sum()):,} damaging of {labels.size:,} "
          f"(root-action) pairs ({labels.mean():.2%})", flush=True)

    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(config.device)
    head = DamageHead(config).to(config.device)
    opt = optimizer([world, head], config)
    rng = torch.Generator(device=config.device).manual_seed(config.seed + 1001)
    draw = np.random.default_rng(config.seed + 91)

    args.out.mkdir(parents=True, exist_ok=True)
    steps = 100 if args.preflight else args.steps
    started, curve, positives = time.time(), [], 0
    candidates = torch.arange(17, device=config.device)
    for step in range(steps):
        finetune = step >= args.steps * (1 - config.long_only_fraction)
        long = finetune or (step + 1) % config.long_batch_every == 0
        length = config.sequence_long if long else config.sequence
        chosen = draw.integers(0, len(fit), config.batch)
        histories, leds, targets = [], [], []
        for index in chosen:
            row = fit[index]
            t = row["t"]
            start = max(0, t - length + 1)
            histories.append(cached[row["episode"]][start : t + 1].clone())
            acts = row["acts"]
            led = np.concatenate([[config.n_actions] if start == 0 else [acts[start - 1]],
                                  acts[start : t]]).astype(np.int64)
            leds.append(torch.from_numpy(led))
            targets.append(torch.from_numpy(row["label"]))
        width = min(h.shape[0] for h in histories)
        z = torch.stack([h[-width:] for h in histories]).to(config.device)
        led = torch.stack([l[-width:] for l in leds]).to(config.device)
        target = torch.stack(targets).to(config.device)

        committed, conditioning = commit_inputs(z, rng, config)
        features, _, _ = world(None, led, committed, conditioning)
        last = features[:, -1:]
        logits = predict_logits(
            world, head, last.expand(len(z), 17, *last.shape[2:]),
            candidates[None].expand(len(z), -1),
        )
        positive = float(target.sum())
        weight = torch.tensor(max(target.numel() - positive, 1.0) / max(positive, 1.0),
                              device=config.device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=weight)
        opt.zero_grad()
        loss.backward()
        for group in opt.param_groups:
            group["lr"] = config.learning_rate * min(1.0, (step + 1) / config.warmup)
        torch.nn.utils.clip_grad_norm_(
            [p for m in (world, head) for p in m.parameters()], config.grad_clip)
        opt.step()
        curve.append(float(loss.detach()))
        positives += int(positive)
        if (step + 1) % 500 == 0 or step + 1 == steps:
            window = curve[-500:]
            rate = (step + 1) / (time.time() - started)
            print(f"  {step+1}/{steps} loss={sum(window)/len(window):.4f} "
                  f"positives={positives:,} [{time.time()-started:.0f}s, "
                  f"{(steps-step-1)/rate:.0f}s left]", flush=True)
        if not args.preflight and (step + 1) in tuple(args.milestones):
            save(args.out / f"model_{step+1:06d}.pt", config, part0=world, part1=head)

    (args.out / ("preflight.json" if args.preflight else "training_report.json")).write_text(
        json.dumps({"steps": steps, "seconds": time.time() - started,
                    "fit_roots": len(fit), "positive_pairs_seen": positives,
                    "distinct_positive_pairs": int(labels.sum()),
                    "curve_tail": curve[-500:],
                    "objective": "BCE over all 17 simulator outcomes per root"}, indent=2))
    print(f"done in {time.time()-started:.0f}s; positives seen {positives:,}")


if __name__ == "__main__":
    main()
