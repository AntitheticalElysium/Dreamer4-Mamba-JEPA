"""Synthetic known-branch successor-mode controls (Phase C).

The environment has two known, context/action-conditional successor modes. This
isolates mixture mechanics from representation failure and reports MoP-style
codebook, shuffled-context, router-gating, precision, and route-validity checks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))

from model import FuturePredictor, ModelConfig, cosine_distance  # noqa: E402


@dataclass
class BranchingProcess:
    contexts: Tensor
    true_modes: Tensor
    positive_probability: Tensor

    @classmethod
    def create(cls, seed: int, conditions: int, actions: int, dim: int, device):
        generator = torch.Generator(device=device).manual_seed(seed)
        contexts = F.normalize(
            torch.randn(conditions, dim, generator=generator, device=device), dim=-1
        )
        transform = torch.randn(dim, dim, generator=generator, device=device) / math.sqrt(dim)
        action_shift = torch.randn(actions, dim, generator=generator, device=device)
        branch = torch.randn(
            conditions, actions, dim, generator=generator, device=device
        )
        base = torch.einsum("cd,de->ce", contexts, transform)
        base = F.normalize(base[:, None] + action_shift[None], dim=-1)
        # Make the two outcomes genuinely separated rather than relying on an
        # arbitrary random-vector scale. Orthogonal equal-norm branches have
        # cosine distance 1 from one another and ~0.293 from their mean.
        branch = branch - (branch * base).sum(-1, keepdim=True) * base
        branch = F.normalize(branch, dim=-1)
        negative = F.normalize(base - branch, dim=-1)
        positive = F.normalize(base + branch, dim=-1)
        true_modes = torch.stack([negative, positive], dim=2)
        ids = torch.arange(conditions, device=device)[:, None]
        action_ids = torch.arange(actions, device=device)[None]
        positive_probability = torch.where(
            (ids + action_ids) % 2 == 0,
            torch.full((conditions, actions), 0.8, device=device),
            torch.full((conditions, actions), 0.2, device=device),
        )
        return cls(contexts, true_modes, positive_probability)

    def sample(self, count: int, generator: torch.Generator):
        conditions, actions = self.positive_probability.shape
        condition = torch.randint(
            conditions, (count,), generator=generator, device=self.contexts.device
        )
        action = torch.randint(
            actions, (count,), generator=generator, device=self.contexts.device
        )
        probability = self.positive_probability[condition, action]
        branch = (torch.rand(count, generator=generator, device=probability.device) < probability).long()
        target = self.true_modes[condition, action, branch]
        target = F.normalize(
            target
            + 0.02
            * torch.randn(
                target.shape,
                generator=generator,
                device=target.device,
                dtype=target.dtype,
            ),
            dim=-1,
        )
        context = self.contexts[condition]
        return condition, context[:, None], action, branch, target[:, None]

    def all_conditions(self):
        condition_count, action_count = self.positive_probability.shape
        condition = torch.arange(condition_count, device=self.contexts.device).repeat_interleave(action_count)
        action = torch.arange(action_count, device=self.contexts.device).repeat(condition_count)
        return condition, self.contexts[condition, None], action


class CodebookControl(nn.Module):
    def __init__(self, modes: int, dim: int, conditions: int, actions: int):
        super().__init__()
        self.codebook = nn.Parameter(F.normalize(torch.randn(modes, 1, dim), dim=-1))
        self.condition = nn.Embedding(conditions, dim)
        self.action = nn.Embedding(actions, dim)
        self.router = nn.Linear(2 * dim, modes)
        self.modes = modes

    def predictions(self, condition: Tensor, context: Tensor, action: Tensor):
        modes = self.codebook[None].expand(len(condition), -1, -1, -1)
        logits = self.router(
            torch.cat([self.condition(condition), self.action(action)], dim=-1)
        )
        return modes, logits


def predictor_config(dim: int, actions: int, modes: int, kind: str):
    return ModelConfig(
        token_dim=dim,
        spatial_heads=4,
        temporal_backend="gru",
        action_dim=actions,
        predictor="deterministic" if kind == "deterministic" else "mixture",
        predictor_depth=2,
        modes=max(2, modes),
        horizon_bins=2,
        mode_balance_temperature=0.1,
    )


def model_predictions(model, condition, context, action):
    if isinstance(model, CodebookControl):
        return model.predictions(condition, context, action)
    horizon = torch.ones(len(action), dtype=torch.long, device=action.device)
    return model.all_predictions(context, action, horizon)


def train_model(
    kind: str,
    modes: int,
    process: BranchingProcess,
    seed: int,
    steps: int,
    batch_size: int,
    balance_weight: float,
):
    device = process.contexts.device
    dim = process.contexts.shape[-1]
    action_count = process.positive_probability.shape[1]
    torch.manual_seed(seed + 1000)
    if kind == "codebook":
        model = CodebookControl(
            modes, dim, len(process.contexts), action_count
        ).to(device)
    else:
        model = FuturePredictor(
            predictor_config(dim, action_count, modes, kind)
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 2000)
    recent = []
    for _ in range(steps):
        condition, context, action, _, target = process.sample(batch_size, generator)
        if isinstance(model, CodebookControl):
            predictions, logits = model.predictions(condition, context, action)
            distances = cosine_distance(predictions, target[:, None]).mean(-1)
            assignment = distances.argmin(-1)
            regression = distances.gather(1, assignment[:, None]).mean()
            router = F.cross_entropy(logits, assignment.detach())
            soft = torch.softmax(-distances / 0.1, dim=-1).mean(0).clamp_min(1e-8)
            balance = (soft * (soft.log() + math.log(modes))).sum()
        else:
            horizon = torch.ones(batch_size, dtype=torch.long, device=device)
            output = model(context, action, horizon, target)
            regression = output.regression
            router = output.router_loss
            balance = output.balance_loss
        loss = regression + 0.1 * router + balance_weight * balance
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        recent.append(float(loss.detach()))
    return model, float(np.mean(recent[-100:]))


@torch.no_grad()
def evaluate(
    model,
    process: BranchingProcess,
    seed: int,
    samples: int = 4096,
    validity_threshold: float = 0.15,
):
    device = process.contexts.device
    generator = torch.Generator(device=device).manual_seed(seed + 3000)
    condition, context, action, _, target = process.sample(samples, generator)
    predictions, logits = model_predictions(model, condition, context, action)
    distances = cosine_distance(predictions, target[:, None]).mean(-1)
    heldout_best = distances.min(-1).values.mean()

    permutation = torch.randperm(samples, generator=generator, device=device)
    shuffled_predictions, _ = model_predictions(
        model, condition[permutation], context[permutation], action[permutation]
    )
    shuffled_error = cosine_distance(
        shuffled_predictions, target[:, None]
    ).mean(-1).min(-1).values.mean()

    all_condition, all_context, all_action = process.all_conditions()
    candidates, all_logits = model_predictions(
        model, all_condition, all_context, all_action
    )
    true = process.true_modes[all_condition, all_action, :, None, :]
    # [condition-action, K, true-mode]
    candidate_to_true = cosine_distance(
        candidates[:, :, None], true[:, None]
    ).mean(-1)
    valid = candidate_to_true.min(-1).values < validity_threshold
    covered = candidate_to_true.min(1).values < validity_threshold
    probabilities = all_logits.softmax(-1)
    active = probabilities > (0.5 / candidates.shape[1])
    gated_valid = valid & active
    gated_covered = (
        candidate_to_true.masked_fill(~active[:, :, None], float("inf")).min(1).values
        < validity_threshold
    )
    active_count = active.sum().clamp_min(1)

    nearest_mode = candidate_to_true.argmin(-1)
    predicted_branch_probability = torch.zeros(len(candidates), device=device)
    predicted_branch_probability.scatter_add_(
        0,
        torch.arange(len(candidates), device=device).repeat_interleave(candidates.shape[1]),
        (probabilities * (nearest_mode == 1) * valid).flatten(),
    )
    true_positive_probability = process.positive_probability[
        all_condition, all_action
    ]
    total_variation = (predicted_branch_probability - true_positive_probability).abs()

    hard_assignment = distances.argmin(-1)
    usage = F.one_hot(hard_assignment, candidates.shape[1]).float().mean(0)
    return {
        "heldout_best_of_k_cosine": float(heldout_best),
        "shuffled_context_best_of_k_cosine": float(shuffled_error),
        "shuffled_context_degradation": float(shuffled_error - heldout_best),
        "raw_mode_coverage": float(covered.float().mean()),
        "raw_transition_precision": float(valid.float().mean()),
        "router_gated_mode_coverage": float(gated_covered.float().mean()),
        "router_gated_transition_precision": float(
            gated_valid.float().sum() / active_count
        ),
        "route_valid_probability_mass": float((probabilities * valid).sum(-1).mean()),
        "router_probability_total_variation": float(total_variation.mean()),
        "mean_active_heads": float(active.float().sum(-1).mean()),
        "hard_assignment_usage": [float(value) for value in usage],
    }


def run_arm(
    kind,
    modes,
    balance_weight,
    seed,
    steps,
    batch,
    conditions,
    actions,
    dim,
    device,
):
    process = BranchingProcess.create(seed, conditions, actions, dim, device)
    model, final_loss = train_model(
        kind, modes, process, seed, steps, batch, balance_weight
    )
    result = evaluate(model, process, seed)
    result.update(
        {
            "kind": kind,
            "modes": 1 if kind == "deterministic" else modes,
            "balance_weight": balance_weight,
            "seed": seed,
            "final_train_loss_mean_100": final_loss,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    del model
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--conditions", type=int, default=8)
    parser.add_argument("--actions", type=int, default=2)
    parser.add_argument("--dim", type=int, default=16)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms = [
        ("deterministic", 1, 0.0),
        ("mixture", 2, 0.0),
        ("mixture", 2, 0.05),
        ("mixture", 4, 0.05),
        ("codebook", 2, 0.05),
    ]
    results = [
        run_arm(
            kind,
            modes,
            balance,
            seed,
            args.steps,
            args.batch,
            args.conditions,
            args.actions,
            args.dim,
            device,
        )
        for seed in args.seeds
        for kind, modes, balance in arms
    ]
    print(
        json.dumps(
            {
                "protocol": {
                    "known_modes_per_condition": 2,
                    "conditions": args.conditions,
                    "actions": args.actions,
                    "target_noise_std": 0.02,
                    "router_probabilities": [0.2, 0.8],
                    "validity_cosine_threshold": 0.15,
                    "steps": args.steps,
                    "batch": args.batch,
                    "seeds": args.seeds,
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
