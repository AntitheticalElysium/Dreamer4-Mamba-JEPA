"""Is the consequence the world model misses directly learnable from frozen Z*?

Trains the production Direct-Attention backbone end to end on the scalar the fork
gate measures -- `y_t = d^T(z_{t+1} - z_t)` along the fixed TRAIN fatality
direction -- rather than on the 512-dimensional next latent. `Z*` stays frozen,
so a strong result says the representation and data carry usable consequence and
the latent-prediction objective fails to express it; a weak result implicates the
representation itself.

The readout is the scalar analogue of `World.predict`: identical pooling and
action conditioning, one scalar per spatial slot, then a learned combination
across slots. Replacing only the final projection would emit `n_spatial` scalars
per transition, not one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    cached_train,
    data_digests,
    file_digest,
    implementation_digests,
)
from d4mj.checkpoint import load, save
from d4mj.config import Config
from d4mj.data import sample_batch
from d4mj.train import _generators, _to, _update, optimizer
from d4mj.transition import World, commit_inputs

VERSION = "consequence-learnability-benchmark-v1"


class ConsequenceHead(nn.Module):
    """`World.predict`'s readout with a scalar codomain and no `tanh`.

    `y` is an unbounded projection, so the production squashing would cap it. The
    slot combination is the one addition: pooling yields `n_spatial` slots and the
    benchmark's target is one number per transition.
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


def predict_scalar(
    world: World, head: ConsequenceHead, features: Tensor, taken: Tensor
) -> Tensor:
    """The production Direct path up to the readout, then the scalar head."""
    spatial = features[:, :, world.spatial]
    register = features[:, :, world.register]
    pooled = world.pool(torch.cat([spatial, register], dim=2).transpose(2, 3)).transpose(2, 3)
    context = world.action_embed(taken)[:, :, None].expand_as(pooled)
    return head(pooled, context)


def consequence_target(latents: Tensor, direction: Tensor) -> Tensor:
    """`d^T(z_{t+1} - z_t)`. The gate subtracts the same action mean from both
    endpoints, so it cancels in the difference and no centring is needed here."""
    flat = latents.flatten(2)
    projected = flat @ direction.to(flat.device, flat.dtype)
    return projected[:, 1:] - projected[:, :-1]


def benchmark_loss(
    world: World,
    head: ConsequenceHead,
    batch,
    rng: torch.Generator,
    config: Config,
    direction: Tensor,
) -> Tensor:
    committed, conditioning = commit_inputs(batch.latents, rng, config)
    features, _, _ = world(None, batch.led_to_action, committed, conditioning)
    taken = batch.led_to_action[:, 1:]
    predicted = predict_scalar(world, head, features[:, :-1], taken)
    target = consequence_target(batch.latents, direction)
    per_row = (predicted - target).pow(2).mean(dim=1)
    mask = batch.rows("dynamics").to(per_row.device).float()
    return (per_row * mask).sum() / mask.sum().clamp(min=1.0)


def scalar_contrast(
    predicted: Tensor, true: Tensor, fatal: Tensor, group: Tensor, *, samples: int, seed: int
) -> dict:
    """The gate's within-state conditional consequence, for scalars.

    `delta_metrics` projects latents internally and cannot accept `y` directly, so
    this reproduces its statistic: per-state fatal-minus-safe means, averaged over
    states, with a whole-state bootstrap.
    """
    rows = []
    for value in group.unique():
        selected = group == value
        label = fatal[selected]
        if not bool(label.any()) or not bool((~label).any()):
            continue
        truth, estimate = true[selected], predicted[selected]
        rows.append(
            (
                float(truth[label].mean() - truth[~label].mean()),
                float(estimate[label].mean() - estimate[~label].mean()),
            )
        )
    if not rows:
        return {}
    values = torch.tensor(rows, dtype=torch.float)
    true_contrast = float(values[:, 0].mean())
    predicted_contrast = float(values[:, 1].mean())
    generator = torch.Generator().manual_seed(seed)
    draws = []
    for _ in range(samples):
        chosen = torch.randint(len(values), (len(values),), generator=generator)
        draws.append(values[chosen][:, 1].mean())
    draws = torch.stack(draws)
    recovered = predicted_contrast / max(abs(true_contrast), 1e-12)
    return {
        "groups": len(values),
        "true_fatal_minus_safe": true_contrast,
        "predicted_fatal_minus_safe": predicted_contrast,
        "recovered_fraction": recovered,
        "predicted_contrast_ci95": [
            float(draws.quantile(0.025)),
            float(draws.quantile(0.975)),
        ],
        "recovered_fraction_ci95": [
            float(draws.quantile(0.025) / max(abs(true_contrast), 1e-12)),
            float(draws.quantile(0.975) / max(abs(true_contrast), 1e-12)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--milestones", type=int, nargs="+", default=(5_000, 10_000, 20_000))
    parser.add_argument("--probe", type=int, default=0)
    args = parser.parse_args()

    config = Config(transition="direct", time_mixer="attention")
    base = Config()
    prepared = torch.load(args.prepared, weights_only=False)
    direction = prepared["direction"].float().to(config.device).flatten()
    if abs(float(direction.norm()) - 1.0) > 1e-4:
        raise ValueError("fatality direction is not unit norm")

    encoder, episodes = cached_train(
        args.phase1a, base, args.expert, support=args.support, cache=args.cache
    )
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    corpus_report = {
        "train_episodes": len(episodes),
        "train_transitions": sum(len(episode) for episode in episodes),
        "bc_eligible_episodes": sum(bool(e.bc_eligible) for e in episodes),
        "terminal_episodes": sum(bool(e.terminated.any()) for e in episodes),
    }
    print(f"TRAIN corpus loaded: {corpus_report}", flush=True)

    torch.manual_seed(config.seed + 1)
    world = World(config).to(config.device)
    head = ConsequenceHead(config).to(config.device)
    opt = optimizer([world, head], config)
    sampler, rng = _generators(config, 2)

    args.out.mkdir(parents=True, exist_ok=True)
    contract = {
        "version": VERSION,
        "phase1a": file_digest(args.phase1a),
        "prepared": file_digest(args.prepared),
        "data": data_digests(),
        "implementation": implementation_digests(Path(__file__)),
        "steps": args.steps,
        "target": "y_t = d^T (z_{t+1} - z_t), fixed TRAIN fatality direction",
        "representation": "frozen Phase-1A Z*; encoder receives no gradient",
        "trained": "Direct-Attention backbone and scalar head, end to end",
        "initialization": "registered Direct initialization, not the trained checkpoint",
        "mixture": False,
        "corpus": corpus_report,
    }

    steps = args.probe or args.steps
    curve = []
    for step in range(steps):
        batch = _to(sample_batch(episodes, sampler, config, step, args.steps, mixture=False), config.device)
        loss = benchmark_loss(world, head, batch, rng, config, direction)
        _update(opt, loss, [world, head], config, step)
        curve.append(float(loss.detach()))
        completed = step + 1
        if completed % 500 == 0 or completed == steps:
            window = curve[-500:]
            print(f"benchmark {completed}/{steps} loss={sum(window)/len(window):.6f}", flush=True)
        if completed in tuple(args.milestones):
            save(args.out / f"model_{completed:06d}.pt", config, part0=world, part1=head)

    atomic_json(
        args.out / "training_report.json",
        {"contract": contract, "curve": curve[-2000:]},
    )
    print(f"complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
