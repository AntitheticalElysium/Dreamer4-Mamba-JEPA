"""Paired-data scaling: does held-out consequence prediction improve with the number
of distinct paired intervention states?

Everything is held to the factual arm except the count of unique hazard-choice roots
the objective may draw from. Half of every batch is anchored on a hazard-choice root
-- the exact mirror of `train_damage_classifier`'s anchoring -- so the realised
positive loss weight is matched rather than merely intended, and every rung logs what
it actually saw.

Splits are whole-seed. The 965 saved opportunity roots are additionally removed from
every training pool, so no evaluation root enters training even when its seed does.
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
from train_damage_classifier import DamageHead, predict_logits

from d4mj.checkpoint import save
from d4mj.config import Config
from d4mj.train import _share_initialisation, optimizer
from d4mj.transition import World, commit_inputs
from resume import load_state, save_state

DATA = HERE / "branched_damage"


def seed_split(seed: int) -> str:
    draw = int.from_bytes(hashlib.sha256(f"paired-seed:{seed}".encode()).digest()[:8],
                          "little") % 10
    return "fit" if draw < 7 else ("tune" if draw < 8 else "test")


def load_pool():
    """(roots, per-seed trajectories). A root is one recorded state with 17 outcomes."""
    forks = torch.load(ROOT / "artifacts/branched_coverage_gate/branched_forks.pt",
                       weights_only=False)
    reserved = {(int(s), int(t)) for s, t in zip(forks["seed"], forks["step"])}
    trajectories, roots = {}, []
    for path in sorted(DATA.glob("seed-*.pt")):
        payload = torch.load(path, weights_only=False)
        seed = int(payload["seed"])
        trajectories[seed] = (payload["latents"], payload["led_to_action"])
        for row in payload["rows"]:
            label = ((row["health"].numpy() <= -1) | row["dead"].numpy()).astype(np.float32)
            roots.append({
                "seed": seed, "step": int(row["step"]), "label": label,
                "split": seed_split(seed),
                "hazard": bool(label.any() and not label.all()),
                "reserved": (seed, int(row["step"])) in reserved,
            })
    return roots, trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=int, required=True, help="unique hazard-choice roots")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume-every", type=int, default=500)
    args = parser.parse_args()

    config = Config(transition="direct", time_mixer="attention")
    roots, trajectories = load_pool()
    fit = [r for r in roots if r["split"] == "fit" and not r["reserved"]]
    hazard = [r for r in fit if r["hazard"]]
    print(f"pool: {len(roots):,} rescored roots, {len(fit):,} fit and unreserved, "
          f"{len(hazard):,} of them hazard-choice", flush=True)
    if args.roots > len(hazard):
        raise SystemExit(f"rung {args.roots} exceeds the {len(hazard)} available")

    # nested rungs: a stable shuffle, then a prefix, so smaller rungs are subsets
    order = np.random.default_rng(4242).permutation(len(hazard))
    rung = [hazard[i] for i in order[: args.roots]]
    print(f"rung: {len(rung)} hazard roots, "
          f"{int(sum(r['label'].sum() for r in rung)):,} damaging pairs", flush=True)

    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(config.device)
    head = DamageHead(config).to(config.device)
    opt = optimizer([world, head], config)
    rng = torch.Generator(device=config.device).manual_seed(config.seed + 1001)
    draw = np.random.default_rng(config.seed + 91)
    candidates = torch.arange(17, device=config.device)

    def batch_rows(length):
        picked = []
        for row in range(config.batch):
            source = rung if row % 2 == 0 else fit
            picked.append(source[draw.integers(len(source))])
        histories, leds, targets = [], [], []
        for record in picked:
            latents, led = trajectories[record["seed"]]
            t = record["step"]
            start = max(0, t - length + 1)
            histories.append(latents[start : t + 1])
            leds.append(led[start : t + 1])
            targets.append(torch.from_numpy(record["label"]))
        width = min(h.shape[0] for h in histories)
        return (torch.stack([h[-width:] for h in histories]),
                torch.stack([l[-width:] for l in leds]),
                torch.stack(targets), picked)

    args.out.mkdir(parents=True, exist_ok=True)
    steps = 100 if args.preflight else args.steps
    started, curve = time.time(), []
    seen_positive, steps_with_positive, weight_sum, target_total = 0, 0, 0.0, 0
    repeats: dict[tuple[int, int], int] = {}
    resume_path = args.out / "resume.pt"
    begin, extra = (0, {}) if args.preflight else load_state(
        resume_path, {"world": world, "head": head}, opt, rng, draw)
    if begin:
        curve = extra.get("curve", [])
        seen_positive = int(extra.get("seen_positive", 0))
        steps_with_positive = int(extra.get("steps_with_positive", 0))
        weight_sum = float(extra.get("weight_sum", 0.0))
        target_total = int(extra.get("target_total", 0))
        repeats = {tuple(k): v for k, v in extra.get("repeats", [])}
        print(f"resumed from step {begin}", flush=True)
    for step in range(begin, steps):
        finetune = step >= args.steps * (1 - config.long_only_fraction)
        long = finetune or (step + 1) % config.long_batch_every == 0
        length = config.sequence_long if long else config.sequence
        z, led, target, picked = batch_rows(length)
        z = z.to(config.device)
        led = led.to(config.device)
        target = target.to(config.device)
        committed, conditioning = commit_inputs(z, rng, config)
        features, _, _ = world(None, led, committed, conditioning)
        last = features[:, -1:]
        logits = predict_logits(world, head, last.expand(len(z), 17, *last.shape[2:]),
                                candidates[None].expand(len(z), -1))
        positive = float(target.sum())
        negative = float(target.numel() - positive)
        weight = torch.tensor(max(negative, 1.0) / max(positive, 1.0), device=config.device)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target,
                                                              pos_weight=weight)
        opt.zero_grad()
        loss.backward()
        for group in opt.param_groups:
            group["lr"] = config.learning_rate * min(1.0, (step + 1) / config.warmup)
        torch.nn.utils.clip_grad_norm_(
            [p for m in (world, head) for p in m.parameters()], config.grad_clip)
        opt.step()
        curve.append(float(loss.detach()))
        seen_positive += int(positive)
        target_total += int(target.numel())
        if positive > 0:
            steps_with_positive += 1
            weight_sum += 0.5          # pos_weight = N_neg/N_pos equalises the two masses
        for record in picked:
            if record["hazard"]:
                key = (record["seed"], record["step"])
                repeats[key] = repeats.get(key, 0) + 1
        if not args.preflight and ((step + 1) % args.resume_every == 0 or step + 1 == steps):
            save_state(resume_path, step + 1, {"world": world, "head": head}, opt, rng,
                       draw, extra={"curve": curve[-500:], "seen_positive": seen_positive,
                                    "steps_with_positive": steps_with_positive,
                                    "weight_sum": weight_sum, "target_total": target_total,
                                    "repeats": [(list(k), v) for k, v in repeats.items()]})
        if (step + 1) % 500 == 0 or step + 1 == steps:
            print(f"  {step+1}/{steps} loss={sum(curve[-500:])/len(curve[-500:]):.4f} "
                  f"positives={seen_positive:,} [{time.time()-started:.0f}s]", flush=True)

    counts = np.array(list(repeats.values())) if repeats else np.array([0])
    realised = {
        "rung_unique_hazard_roots": len(rung),
        "distinct_damaging_pairs_in_rung": int(sum(r["label"].sum() for r in rung)),
        "total_positive_presentations": seen_positive,
        "scored_targets": target_total,
        "positive_fraction_of_targets": seen_positive / max(target_total, 1),
        "steps_with_a_positive": steps_with_positive / steps,
        "realised_positive_loss_fraction": weight_sum / steps,
        "hazard_roots_drawn": len(repeats),
        "repeats_per_hazard_root_mean": float(counts.mean()),
        "repeats_per_hazard_root_median": float(np.median(counts)),
        "seconds": time.time() - started,
    }
    print("\nrealised balance: " + json.dumps(realised, indent=2), flush=True)
    if not args.preflight:
        save(args.out / "model.pt", config, part0=world, part1=head)
    (args.out / ("preflight.json" if args.preflight else "training_report.json")).write_text(
        json.dumps({"realised": realised, "curve_tail": curve[-500:]}, indent=2))


if __name__ == "__main__":
    main()
