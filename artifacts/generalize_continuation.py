"""Train continuation alone on all TRAIN terminal tails and evaluate on DEV."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from artifacts.run_stage_a import corpus
from d4mj.agent import Heads, head_targets
from d4mj.config import Config
from d4mj.data import Batch, Episode, _terminal_start, _window
from d4mj.train import (
    _to,
    cache_latents,
    train_dynamics,
    train_representation,
)
from d4mj.transition import commit_inputs, transition_loss


def continuation_logits(heads: Heads, agent: torch.Tensor) -> torch.Tensor:
    """Run only the continuation stack, avoiding unused policy/reward heads."""
    pooled = agent.mean(dim=2)
    features = heads.model_body(pooled)
    return heads.continuation(features)[..., 0]


def auc(score: torch.Tensor, target: torch.Tensor) -> float:
    positive = score[target]
    negative = score[~target]

    if not len(positive) or not len(negative):
        return 0.5

    difference = positive[:, None] - negative[None]
    return float(
        (
            difference.gt(0).float()
            + 0.5 * difference.eq(0).float()
        ).mean()
    )


@torch.no_grad()
def metrics(
    heads: Heads,
    agent: torch.Tensor,
    continuation: torch.Tensor,
    device: str,
) -> dict[str, float]:
    heads.eval()

    probabilities = []
    chunk = 256

    for start in range(0, len(agent), chunk):
        logits = continuation_logits(
            heads,
            agent[start : start + chunk].to(device),
        )
        probabilities.append(logits.sigmoid().cpu())

    probability = torch.cat(probabilities).flatten()
    truth = continuation.flatten().bool()

    alive = truth
    dead = ~truth
    death_score = 1.0 - probability

    return {
        "examples": int(len(truth)),
        "alive": int(alive.sum()),
        "dead": int(dead.sum()),
        "bce": float(
            F.binary_cross_entropy(
                probability.clamp(1e-7, 1 - 1e-7),
                truth.float(),
            )
        ),
        "death_auc": auc(death_score, dead),
        "accuracy": float(
            ((probability >= 0.5) == truth).float().mean()
        ),
        "continuation_on_alive": float(probability[alive].mean()),
        "continuation_on_dead": float(probability[dead].mean()),
    }


def terminal_episodes(
    episodes: list[Episode],
    config: Config,
) -> list[Episode]:
    """Each retained episode contributes exactly one alive/dead pair."""
    length = config.sequence

    return [
        episode
        for episode in episodes
        if episode.uniform_eligible
        and episode.latents is not None
        and len(episode) + 1 >= length
        and bool(episode.terminated.any())
    ]


def make_terminal_batch(
    episodes: list[Episode],
    config: Config,
) -> Batch:
    """Build unique short terminal tails without sampling with replacement."""
    length = config.sequence

    rows = [
        _window(
            episode,
            _terminal_start(episode, length),
            length,
            config,
        )
        for episode in episodes
    ]

    stack = {
        field: torch.stack([row[field] for row in rows])
        for field in rows[0]
    }

    count = len(episodes)

    return Batch(
        burn_in=0,
        relevant=torch.zeros(count, dtype=torch.bool),
        support=torch.ones(count, dtype=torch.bool),
        **stack,
    )


@torch.no_grad()
def collect_features(
    episodes: list[Episode],
    world,
    config: Config,
    *,
    seed: int,
    chunk_size: int,
    world_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return generated features, observed features, and balanced labels."""
    selected = terminal_episodes(episodes, config)

    if not selected:
        raise RuntimeError("No eligible terminal episodes")

    rng = torch.Generator(device=config.device).manual_seed(seed)

    generated_rows = []
    observed_rows = []
    target_rows = []

    for start in range(0, len(selected), chunk_size):
        subset = selected[start : start + chunk_size]
        batch = _to(make_terminal_batch(subset, config), config.device)

        # Direct's ordinary Phase-2 route. Its final two readouts are replaced
        # by recursively generated readouts.
        _, generated_full = transition_loss(
            world,
            batch,
            rng,
            config,
            return_agent=True,
            step=world_steps,
        )

        # Teacher-forced observed path for the same terminal tails.
        committed, conditioning = commit_inputs(
            batch.latents,
            rng,
            config,
        )
        _, observed_full, _ = world(
            None,
            batch.led_to_action,
            committed,
            conditioning,
        )

        targets = head_targets(batch, config)["continuation"][:, -2:, 0]

        assert bool(targets[:, 0].eq(1).all()), (
            "Penultimate target is not alive"
        )
        assert bool(targets[:, 1].eq(0).all()), (
            "Final target is not dead"
        )

        generated_rows.append(generated_full[:, -2:].cpu())
        observed_rows.append(observed_full[:, -2:].cpu())
        target_rows.append(targets.cpu())

    generated = torch.cat(generated_rows)
    observed = torch.cat(observed_rows)
    target = torch.cat(target_rows)

    # [episodes, 2, agent_tokens, width]
    # becomes [2 * episodes, 1, agent_tokens, width].
    generated = generated.flatten(0, 1).unsqueeze(1)
    observed = observed.flatten(0, 1).unsqueeze(1)
    target = target.flatten().unsqueeze(1)

    return generated, observed, target


def fit(
    train_agent: torch.Tensor,
    train_target: torch.Tensor,
    evaluations: dict[str, tuple[torch.Tensor, torch.Tensor]],
    config: Config,
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> dict:
    """Train only the same model-body + continuation stack used in Phase 2."""
    torch.manual_seed(seed)

    heads = Heads(config).to(config.device)

    for parameter in heads.parameters():
        parameter.requires_grad_(False)

    for parameter in heads.model_body.parameters():
        parameter.requires_grad_(True)

    for parameter in heads.continuation.parameters():
        parameter.requires_grad_(True)

    parameters = [
        parameter
        for parameter in heads.parameters()
        if parameter.requires_grad
    ]

    optimiser = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=0.0,
    )

    sampler = torch.Generator().manual_seed(seed + 1)
    count = len(train_agent)

    heads.train()

    for step in range(steps):
        indices = torch.randint(
            count,
            (min(batch_size, count),),
            generator=sampler,
        )

        agent = train_agent[indices].to(config.device)
        target = train_target[indices].to(config.device)

        logits = continuation_logits(heads, agent)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step in {0, 99, 499, 999, steps - 1}:
            print(
                f"step={step + 1:4d} loss={float(loss):.6f}",
                flush=True,
            )

    return {
        name: metrics(heads, agent, target, config.device)
        for name, (agent, target) in evaluations.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch", type=int, default=32)
    parser.add_argument("--tokenizer-steps", type=int, default=3000)
    parser.add_argument("--dynamics-steps", type=int, default=20000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/continuation_generalization.json"),
    )

    args = parser.parse_args()

    base = Config()
    config = replace(
        base,
        transition="direct",
        time_mixer="attention",
    )

    train_set, dev_set = corpus(base, args.expert, print)

    encoder, _, cached_train = train_representation(
        train_set,
        args.tokenizer_steps,
        base,
        checkpoint=args.reuse / "phase1a.pt",
    )
    cached_dev = cache_latents(encoder, dev_set, base)

    encoder.cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    world = train_dynamics(
        cached_train,
        args.dynamics_steps,
        config,
        checkpoint=args.reuse / "direct-attention.1b.pt",
    ).eval()

    for parameter in world.parameters():
        parameter.requires_grad_(False)

    train_generated, train_observed, train_target = collect_features(
        cached_train,
        world,
        config,
        seed=config.seed + 800,
        chunk_size=args.feature_batch,
        world_steps=args.dynamics_steps,
    )

    dev_generated, dev_observed, dev_target = collect_features(
        cached_dev,
        world,
        config,
        seed=config.seed + 900,
        chunk_size=args.feature_batch,
        world_steps=args.dynamics_steps,
    )

    print(
        f"TRAIN: {len(train_target) // 2} terminal episodes, "
        f"{len(train_target)} balanced examples"
    )
    print(
        f"DEV: {len(dev_target) // 2} terminal episodes, "
        f"{len(dev_target)} balanced examples"
    )

    print("\nFit generated path", flush=True)
    generated_fit = fit(
        train_generated,
        train_target,
        {
            "train_generated": (train_generated, train_target),
            "dev_generated": (dev_generated, dev_target),
            "train_observed": (train_observed, train_target),
            "dev_observed": (dev_observed, dev_target),
        },
        config,
        steps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=config.seed + 1000,
    )

    print("\nFit observed path", flush=True)
    observed_fit = fit(
        train_observed,
        train_target,
        {
            "train_observed": (train_observed, train_target),
            "dev_observed": (dev_observed, dev_target),
            "train_generated": (train_generated, train_target),
            "dev_generated": (dev_generated, dev_target),
        },
        config,
        steps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=config.seed + 1100,
    )

    report = {
        "train_terminal_episodes": len(train_target) // 2,
        "dev_terminal_episodes": len(dev_target) // 2,
        "generated_fit": generated_fit,
        "observed_fit": observed_fit,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()