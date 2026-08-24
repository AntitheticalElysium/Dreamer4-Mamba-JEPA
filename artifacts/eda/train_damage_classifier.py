"""Step 2: can frozen Z* + history + action predict damage when asked to?

The production Direct-Attention backbone from its registered initialization, with
the Phase-1A encoder frozen and the cached latents as input. The only change to the
objective is the target: instead of the next latent, a factual binary label
`1[health decreases]` for the transition each block's outgoing action causes.

Damaging transitions are 1.4% of the corpus, so half of every batch is anchored on
one. The head mirrors `World.predict`'s readout -- same pooling, same action
conditioning -- with a scalar logit codomain, so nothing about how the action enters
the prediction differs from production.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import corpus

from d4mj.checkpoint import save
from d4mj.config import Config
from d4mj.train import _share_initialisation, optimizer
from d4mj.transition import World, commit_inputs


class DamageHead(nn.Module):
    """`World.predict`'s readout with a scalar logit codomain.

    Pooling and action conditioning are copied from the production path; only the
    output width and the absence of `tanh` differ, since a logit is unbounded.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )
        self.combine = nn.Linear(config.n_spatial, 1)

    def forward(self, pooled: Tensor, context: Tensor) -> Tensor:
        slots = self.readout(torch.cat([pooled, context], dim=-1)).squeeze(-1)
        return self.combine(slots).squeeze(-1)


def predict_logits(world: World, head: DamageHead, features: Tensor, taken: Tensor) -> Tensor:
    spatial = features[:, :, world.spatial]
    register = features[:, :, world.register]
    pooled = world.pool(torch.cat([spatial, register], dim=2).transpose(2, 3)).transpose(2, 3)
    context = world.action_embed(taken)[:, :, None].expand_as(pooled)
    return head(pooled, context)


def build_index(rows, off, damage, length):
    """Window starts, and the damaging transitions each length can still reach."""
    eligible = np.array([r["steps"] + 1 >= length for r in rows])
    spans = np.array([max(r["steps"] + 2 - length, 0) for r in rows]) * eligible
    episode_of = np.repeat(np.arange(len(rows)), np.diff(off))
    local = np.arange(off[-1]) - off[episode_of]
    reachable = damage & eligible[episode_of] & (local <= np.repeat(
        np.array([r["steps"] for r in rows]), np.diff(off)) - 1)
    return spans, episode_of, local, np.where(reachable)[0]


class Sampler:
    """Half the rows anchored on a damaging transition, half drawn as production does."""

    def __init__(self, rows, off, damage, config: Config, seed: int):
        self.rows, self.off, self.damage, self.config = rows, off, damage, config
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        for length in (config.sequence, config.sequence_long):
            spans, episode_of, local, positives = build_index(rows, off, damage, length)
            weights = spans / spans.sum()
            self.cache[length] = (spans, episode_of, local, positives, weights)

    def draw(self, step: int, total: int, batch: int, length: int):
        spans, episode_of, local, positives, weights = self.cache[length]
        chosen = []
        for row in range(batch):
            if row % 2 == 0 and len(positives):
                flat = positives[self.rng.integers(len(positives))]
                episode = int(episode_of[flat])
                target = int(local[flat])
                low = max(0, target - length + 2)
                high = min(int(spans[episode]) - 1, target)
                start = int(self.rng.integers(low, high + 1)) if high >= low else max(low, 0)
            else:
                episode = int(self.rng.choice(len(self.rows), p=weights))
                start = int(self.rng.integers(0, max(int(spans[episode]), 1)))
            chosen.append((episode, start))
        return chosen


def gather(chosen, length, latents_of, off, damage, rows, config: Config):
    """Block arrays under the led-to convention, matching `data._window`."""
    z, actions, labels, valid = [], [], [], []
    for episode, start in chosen:
        z_ep = latents_of(episode)                      # (n+1, n_spatial, d_spatial)
        z.append(z_ep[start : start + length].clone())
        act = np.zeros(length, dtype=np.int64)
        lab = np.zeros(length, dtype=np.float32)
        ok = np.zeros(length, dtype=bool)
        n = rows[episode]["steps"]
        for j in range(length - 1):
            t = start + j
            if t < n:
                act[j + 1] = int(action_of(episode, t))
                lab[j] = float(damage[off[episode] + t])
                ok[j] = True
        act[0] = config.n_actions if start == 0 else int(action_of(episode, start - 1))
        actions.append(torch.from_numpy(act))
        labels.append(torch.from_numpy(lab))
        valid.append(torch.from_numpy(ok))
    return (torch.stack(z), torch.stack(actions), torch.stack(labels), torch.stack(valid))


ACTIONS: dict[int, np.ndarray] = {}


def action_of(episode: int, t: int) -> int:
    return int(ACTIONS[episode][t])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=HERE / "damage_classifier")
    parser.add_argument("--milestones", type=int, nargs="+", default=(5_000, 10_000, 20_000))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    config = Config(transition="direct", time_mixer="attention")
    rows = corpus.train_rows()
    corpus.verify_order(rows)
    blob = np.load(HERE / "damage_labels.npz")
    off, damage = blob["offsets"], blob["damage"]
    print(f"corpus {len(rows)} episodes / {off[-1]:,} transitions, "
          f"{int(damage.sum()):,} damaging ({damage.mean():.3%})", flush=True)

    # memory-mapped: the shard storages stay mapped for as long as a tensor
    # references them, so the 6.4 GB cache is never resident all at once.
    latents: dict[int, torch.Tensor] = {}
    for index, value in corpus.iter_cached_latents():
        latents[index] = value
    print(f"cached latent tensors mapped: {len(latents)}", flush=True)

    import json as _json

    manifest = _json.loads((corpus.SUPPORT / "manifest.json").read_text())
    archive = torch.load(corpus.ARCHIVE, weights_only=False, mmap=True)
    by_shard: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        by_shard.setdefault(row["shard"], []).append(i)
    for shard, indices in sorted(by_shard.items()):
        payload = (torch.load(corpus.SUPPORT / manifest["shards"][shard]["file"],
                              weights_only=False, mmap=True) if shard >= 0 else None)
        for i in indices:
            ACTIONS[i] = (payload["episodes"][rows[i]["slot"]]["actions_taken"].numpy()
                          if payload is not None else archive[rows[i]["slot"]]["actions"].numpy())
        del payload

    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(config.device)
    head = DamageHead(config).to(config.device)
    opt = optimizer([world, head], config)
    rng = torch.Generator(device=config.device).manual_seed(config.seed + 1001)
    sampler = Sampler(rows, off, damage, config, seed=config.seed + 91)

    args.out.mkdir(parents=True, exist_ok=True)
    steps = 200 if args.preflight else args.steps
    started = time.time()
    curve, positives_seen = [], 0
    for step in range(steps):
        finetune = args.steps > 0 and step >= args.steps * (1 - config.long_only_fraction)
        long = finetune or (step + 1) % config.long_batch_every == 0
        length = config.sequence_long if long else config.sequence
        chosen = sampler.draw(step, args.steps, config.batch, length)
        z, actions, labels, valid = gather(chosen, length, lambda e: latents[e], off,
                                           damage, rows, config)
        z = z.to(config.device)   # (B, T, n_spatial, d_spatial)
        actions = actions.to(config.device)
        labels = labels.to(config.device)
        valid = valid.to(config.device)
        committed, conditioning = commit_inputs(z, rng, config)
        features, _, _ = world(None, actions, committed, conditioning)
        logits = predict_logits(world, head, features[:, :-1], actions[:, 1:])
        target = labels[:, : length - 1]
        mask = valid[:, : length - 1].float()
        positive = float((target * mask).sum())
        negative = float(mask.sum() - positive)
        weight = torch.tensor(max(negative, 1.0) / max(positive, 1.0), device=config.device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target, weight=mask, pos_weight=weight, reduction="sum"
        ) / mask.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        for group in opt.param_groups:
            group["lr"] = config.learning_rate * min(1.0, (step + 1) / config.warmup)
        torch.nn.utils.clip_grad_norm_(
            [p for m in (world, head) for p in m.parameters()], config.grad_clip
        )
        opt.step()
        curve.append(float(loss.detach()))
        positives_seen += int(positive)
        if (step + 1) % 500 == 0 or step + 1 == steps:
            window = curve[-500:]
            rate = (step + 1) / (time.time() - started)
            print(f"  {step + 1}/{steps} loss={sum(window)/len(window):.4f} "
                  f"positives={positives_seen:,} "
                  f"[{time.time()-started:.0f}s, {(steps-step-1)/rate:.0f}s left]", flush=True)
        if not args.preflight and (step + 1) in tuple(args.milestones):
            save(args.out / f"model_{step + 1:06d}.pt", config, part0=world, part1=head)

    report = {
        "steps": steps, "preflight": args.preflight,
        "positive_targets_seen": positives_seen,
        "seconds": time.time() - started,
        "curve_tail": curve[-500:],
        "objective": "BCE on 1[health decreases], pos_weight balanced per batch",
        "sampling": "half of every batch anchored on a damaging transition",
    }
    (args.out / ("preflight.json" if args.preflight else "training_report.json")).write_text(
        json.dumps(report, indent=2)
    )
    print(f"done in {time.time() - started:.0f}s; positives seen {positives_seen:,}")


if __name__ == "__main__":
    main()
