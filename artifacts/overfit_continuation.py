"""Check whether continuation can memorize matched alive/dead examples."""

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
from d4mj.data import sample_terminal_batch
from d4mj.train import _to, train_dynamics, train_representation
from d4mj.transition import commit_inputs, transition_loss


def auc(score: torch.Tensor, target: torch.Tensor) -> float:
    positive = score[target]
    negative = score[~target]

    if not len(positive) or not len(negative):
        return 0.5

    comparisons = positive[:, None] - negative[None]
    return float(
        (
            comparisons.gt(0).float()
            + 0.5 * comparisons.eq(0).float()
        ).mean()
    )


@torch.no_grad()
def metrics(
    heads: Heads,
    agent: torch.Tensor,
    continuation: torch.Tensor,
) -> dict[str, float]:
    probability = (
        heads(agent)["continuation"][..., 0]
        .sigmoid()
        .flatten()
    )
    truth = continuation.flatten().bool()

    alive = truth
    dead = ~truth
    death_score = 1.0 - probability

    return {
        "bce": float(
            F.binary_cross_entropy(probability, truth.float())
        ),
        "death_auc": auc(death_score, dead),
        "accuracy": float(
            ((probability >= 0.5) == truth).float().mean()
        ),
        "continuation_on_alive": float(probability[alive].mean()),
        "continuation_on_dead": float(probability[dead].mean()),
    }


def fit_head(
    train_agent: torch.Tensor,
    cross_agent: torch.Tensor,
    continuation: torch.Tensor,
    config: Config,
    steps: int,
    learning_rate: float,
) -> dict[str, dict[str, float]]:
    """Fit only the continuation body and output layer."""

    torch.manual_seed(config.seed + 77)
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
    target = continuation.float()

    heads.train()

    for step in range(steps):
        logits = heads(train_agent)["continuation"][..., 0]
        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step in {0, 99, 499, steps - 1}:
            print(
                f"step={step + 1:4d} loss={float(loss):.6f}",
                flush=True,
            )

    heads.eval()

    return {
        "fitted_path": metrics(heads, train_agent, continuation),
        "cross_path": metrics(heads, cross_agent, continuation),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--tokenizer-steps", type=int, default=3000)
    parser.add_argument("--dynamics-steps", type=int, default=20000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/continuation_overfit.json"),
    )

    args = parser.parse_args()

    base = Config()
    config = replace(
        base,
        transition="direct",
        time_mixer="attention",
    )

    # Sampling-only configuration. Do not use this to load the checkpoint,
    # because terminal_batch is part of the checkpoint configuration.
    sampling_config = replace(
        config,
        terminal_batch=args.examples,
    )

    train_set, _ = corpus(base, args.expert, print)

    # Restore Phase 1A and rebuild the frozen latent cache.
    encoder, _, cached_train = train_representation(
        train_set,
        args.tokenizer_steps,
        base,
        checkpoint=args.reuse / "phase1a.pt",
    )

    encoder.cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore Direct-Attention Phase 1B. No Phase 2 is run.
    world = train_dynamics(
        cached_train,
        args.dynamics_steps,
        config,
        checkpoint=args.reuse / "direct-attention.1b.pt",
    ).eval()

    for parameter in world.parameters():
        parameter.requires_grad_(False)

    sampler = torch.Generator().manual_seed(config.seed + 700)
    model_rng = torch.Generator(
        device=config.device
    ).manual_seed(config.seed + 701)

    batch = _to(
        sample_terminal_batch(
            cached_train,
            sampler,
            sampling_config,
            step=0,
            total=args.steps,
        ),
        config.device,
    )

    # Final two positions of each tail:
    # [-2] continuing, [-1] terminal.
    continuation = head_targets(
        batch,
        config,
    )["continuation"][:, -2:, 0]

    assert bool(continuation[:, 0].eq(1).all()), (
        "Penultimate targets are not all alive"
    )
    assert bool(continuation[:, 1].eq(0).all()), (
        "Final targets are not all terminal"
    )

    with torch.no_grad():
        # Direct Phase-2 route. The final two agent readouts are replaced
        # with the two-step generated-prefix readouts.
        _, generated_full = transition_loss(
            world,
            batch,
            model_rng,
            config,
            return_agent=True,
            step=args.dynamics_steps,
        )
        generated = generated_full[:, -2:].detach()

        # Same examples using their real, teacher-forced latent states.
        committed, conditioning = commit_inputs(
            batch.latents,
            model_rng,
            config,
        )
        _, observed_full, _ = world(
            None,
            batch.led_to_action,
            committed,
            conditioning,
        )
        observed = observed_full[:, -2:].detach()

    print("\nFitting generated readouts", flush=True)
    generated_fit = fit_head(
        generated,
        observed,
        continuation,
        config,
        args.steps,
        args.lr,
    )

    print("\nFitting observed readouts", flush=True)
    observed_fit = fit_head(
        observed,
        generated,
        continuation,
        config,
        args.steps,
        args.lr,
    )

    report = {
        "examples_per_class": args.examples,
        "generated_fit": generated_fit,
        "observed_fit": observed_fit,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()