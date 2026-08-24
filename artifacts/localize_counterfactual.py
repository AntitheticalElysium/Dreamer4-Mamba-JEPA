"""Localize failure on the exact counterfactual outcome-gate forks.

No training code is modified. The script:

1. Loads the exact saved S76 fork truth table.
2. Replays the saved (seed, step) states with the Phase-2 Direct-Attention
   world/policy and verifies the true outcomes and production-head predictions.
3. For every action at every terminal-opportunity state, extracts:
      observed successor latent
      observed successor agent readout
      generated successor latent
      generated successor agent readout
4. Runs state-held-out probes. Each test fold is one whole pre-action state, so
   the probe must rank fatal versus safe actions *within an unseen state*.
5. Reports the actual production continuation head and fatal/safe latent MSE.

Decision:
- observed latent/readout strong, generated latent weak -> Direct transition fails
  to preserve action-conditioned fatality.
- generated latent strong, generated readout weak -> world readout loses it.
- generated readout strong, production head weak -> Phase-2 continuation training.
- everything strong but saved gate differs -> gate/replay alignment bug.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.representation import Encoder
from d4mj.transition import World, advance, observe


@dataclass
class ForkData:
    observed_latent: Tensor
    observed_readout: Tensor
    generated_latent: Tensor
    generated_readout: Tensor
    target: Tensor
    action: Tensor
    group: Tensor
    production_generated: Tensor
    production_observed: Tensor


def auc(score: Tensor, target: Tensor) -> float:
    s = score.detach().float().cpu().flatten()
    y = target.detach().bool().cpu().flatten()
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return 0.5
    delta = pos[:, None] - neg[None]
    return float((delta.gt(0).float() + 0.5 * delta.eq(0).float()).mean())


def binary_metrics(probability: Tensor, target: Tensor) -> dict[str, float]:
    p = probability.detach().float().cpu().flatten().clamp(1e-7, 1 - 1e-7)
    y = target.detach().float().cpu().flatten()
    return {
        "bce": float(F.binary_cross_entropy(p, y)),
        "auc": auc(p, y.bool()),
        "accuracy": float(((p >= 0.5) == y.bool()).float().mean()),
        "mean_on_dead": float(p[y.bool()].mean()),
        "mean_on_alive": float(p[~y.bool()].mean()),
    }


def continuation_death_probability(heads: Heads, agent: Tensor) -> Tensor:
    readout = heads(agent)
    return 1.0 - readout["continuation"][..., 0].sigmoid()


def continuation_death_logits(heads: Heads, agent: Tensor) -> Tensor:
    pooled = agent.mean(dim=2)
    features = heads.model_body(pooled)
    # continuation logit is alive; negate it to obtain a death logit.
    return -heads.continuation(features)[..., 0]


def load_models(
    phase1a: Path,
    phase2: Path,
    base: Config,
    config: Config,
) -> tuple[Encoder, World, Heads]:
    encoder = Encoder(base).to(base.device)
    load(phase1a, base, part0=encoder)
    encoder.eval()

    world = World(config).to(config.device)
    heads = Heads(config).to(config.device)
    load(phase2, config, part0=world, part1=heads)
    world.eval()
    heads.eval()

    for module in (encoder, world, heads):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return encoder, world, heads


@torch.no_grad()
def extract_exact_forks(
    saved: dict,
    encoder: Encoder,
    world: World,
    heads: Heads,
    config: Config,
) -> tuple[ForkData, dict]:
    saved_seed = saved["seed"].long()
    saved_step = saved["step"].long()
    true_death = saved["true_death"].bool()
    death_varies = true_death.any(1) & (~true_death).any(1)
    selected_rows = death_varies.nonzero().flatten().tolist()

    if not selected_rows:
        raise RuntimeError("Saved fork file has no terminal-opportunity states")

    key_to_row = {
        (int(saved_seed[row]), int(saved_step[row])): row
        for row in selected_rows
    }
    if len(key_to_row) != len(selected_rows):
        raise RuntimeError("Duplicate (seed, step) terminal-opportunity fork")

    by_seed: dict[int, set[int]] = {}
    for seed, step in key_to_row:
        by_seed.setdefault(seed, set()).add(step)

    observed_latent, observed_readout = [], []
    generated_latent, generated_readout = [], []
    labels, actions, groups = [], [], []
    production_generated, production_observed = [], []

    replay_model_max_error = 0.0
    replay_observed_max_error = 0.0
    reproduced = set()

    # group index follows the saved opportunity-state order, not replay order.
    row_to_group = {row: group for group, row in enumerate(selected_rows)}

    for seed in sorted(by_seed):
        wanted = by_seed[seed]
        last = max(wanted)

        observation, env_state = reset(seed)
        state = None
        incoming = torch.full(
            (1, 1),
            config.n_actions,
            dtype=torch.long,
            device=config.device,
        )
        world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)

        for index in range(last + 1):
            patches = patchify(observation[None, None], config.patch).to(config.device)
            state, agent = observe(
                world, encoder, state, incoming, patches, world_rng, config
            )

            key = (seed, index)
            if key in key_to_row:
                row = key_to_row[key]
                group = row_to_group[row]
                saved_truth = true_death[row]

                for action in range(config.n_actions):
                    # Ground-truth simulator successor from the exact gate state.
                    successor_obs, _, _, terminated, _ = env_step(
                        env_state, action, seed + index + 1
                    )
                    if bool(terminated) != bool(saved_truth[action]):
                        raise AssertionError(
                            f"truth replay mismatch seed={seed} step={index} "
                            f"action={action}: replay={bool(terminated)} "
                            f"saved={bool(saved_truth[action])}"
                        )

                    chosen = torch.tensor(
                        [[action]], dtype=torch.long, device=config.device
                    )

                    # Exact generated path used by the counterfactual gate.
                    generated_rng = torch.Generator(
                        device=config.device
                    ).manual_seed(
                        config.seed + 2**23 + seed * 4099 + index * 17
                    )
                    generated_state, generated_agent = advance(
                        world, state.world, chosen, generated_rng, config
                    )
                    generated_death = continuation_death_probability(
                        heads, generated_agent
                    )[0, -1]

                    # Exact observed-successor path used by the gate diagnostic.
                    observed_rng = torch.Generator(
                        device=config.device
                    ).manual_seed(
                        config.seed + 2**24 + seed * 4099 + index * 17
                    )
                    successor_patches = patchify(
                        successor_obs[None, None], config.patch
                    ).to(config.device)
                    observed_state, observed_agent = observe(
                        world,
                        encoder,
                        state,
                        chosen,
                        successor_patches,
                        observed_rng,
                        config,
                    )
                    observed_death = continuation_death_probability(
                        heads, observed_agent
                    )[0, -1]

                    observed_latent.append(
                        observed_state.world.latent[0, -1].detach().cpu()
                    )
                    observed_readout.append(
                        observed_agent[0, -1].detach().cpu()
                    )
                    generated_latent.append(
                        generated_state.latent[0, -1].detach().cpu()
                    )
                    generated_readout.append(
                        generated_agent[0, -1].detach().cpu()
                    )
                    labels.append(float(terminated))
                    actions.append(action)
                    groups.append(group)
                    production_generated.append(float(generated_death))
                    production_observed.append(float(observed_death))

                    replay_model_max_error = max(
                        replay_model_max_error,
                        abs(float(generated_death) - float(saved["model_death"][row, action])),
                    )
                    if saved.get("observed_death") is not None:
                        replay_observed_max_error = max(
                            replay_observed_max_error,
                            abs(
                                float(observed_death)
                                - float(saved["observed_death"][row, action])
                            ),
                        )

                reproduced.add(key)

            # Follow the original gate's policy trajectory to the next state.
            logits = heads(agent)["policy"][:, -1, 0]
            action = int(
                torch.multinomial(
                    logits.softmax(-1), 1, generator=policy_rng
                )
            )
            previous = env_state
            observation, env_state, _, terminated, truncated = env_step(
                env_state, action, seed + index + 1
            )
            incoming.fill_(action)

            if terminated or truncated:
                if index < last:
                    missing = sorted(
                        step for step in wanted if step > index
                    )
                    raise RuntimeError(
                        f"Replay ended before saved steps for seed {seed}: {missing}"
                    )
                break

    missing_keys = sorted(set(key_to_row) - reproduced)
    if missing_keys:
        raise RuntimeError(f"Failed to replay saved fork states: {missing_keys}")

    # Direct is deterministic. If these differ, this diagnostic is not testing
    # the exact saved gate and must stop.
    if replay_model_max_error > 1e-5:
        raise AssertionError(
            f"generated gate replay mismatch: max abs error "
            f"{replay_model_max_error:.3e}"
        )
    if saved.get("observed_death") is not None and replay_observed_max_error > 1e-5:
        raise AssertionError(
            f"observed gate replay mismatch: max abs error "
            f"{replay_observed_max_error:.3e}"
        )

    data = ForkData(
        observed_latent=torch.stack(observed_latent),
        observed_readout=torch.stack(observed_readout),
        generated_latent=torch.stack(generated_latent),
        generated_readout=torch.stack(generated_readout),
        target=torch.tensor(labels, dtype=torch.float32),
        action=torch.tensor(actions, dtype=torch.long),
        group=torch.tensor(groups, dtype=torch.long),
        production_generated=torch.tensor(production_generated),
        production_observed=torch.tensor(production_observed),
    )
    replay = {
        "terminal_opportunity_states": len(selected_rows),
        "examples": len(labels),
        "generated_prediction_max_abs_error_vs_saved": replay_model_max_error,
        "observed_prediction_max_abs_error_vs_saved": replay_observed_max_error,
    }
    return data, replay


def flatten(x: Tensor) -> Tensor:
    return x.float().flatten(1)


def standardize(
    fit: Tensor,
    validation: Tensor,
    test: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    mean = fit.mean(0, keepdim=True)
    std = fit.std(0, unbiased=False, keepdim=True).clamp(min=1e-5)
    return (fit - mean) / std, (validation - mean) / std, (test - mean) / std


def weighted_bce(logits: Tensor, target: Tensor) -> Tensor:
    y = target.float()
    positive = y.sum().clamp(min=1.0)
    negative = (1.0 - y).sum().clamp(min=1.0)
    return F.binary_cross_entropy_with_logits(
        logits,
        y,
        pos_weight=(negative / positive).detach(),
    )


def validation_key(probability: Tensor, target: Tensor) -> tuple[float, float]:
    return (
        auc(probability, target),
        -float(
            F.binary_cross_entropy(
                probability.clamp(1e-7, 1 - 1e-7),
                target.float(),
            )
        ),
    )


def fit_linear_once(
    fit_x: Tensor,
    fit_y: Tensor,
    val_x: Tensor,
    val_y: Tensor,
    test_x: Tensor,
    *,
    seed: int,
    device: str,
    steps: int,
    lr: float,
    weight_decay: float,
) -> Tensor:
    fit_x, val_x, test_x = standardize(fit_x, val_x, test_x)

    torch.manual_seed(seed)
    model = nn.Linear(fit_x.shape[1], 1).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_key = (-1.0, float("-inf"))
    best = None

    x = fit_x.to(device)
    y = fit_y.to(device)

    for step in range(steps):
        logits = model(x)[:, 0]
        loss = weighted_bce(logits, y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                probability = model(val_x.to(device))[:, 0].sigmoid().cpu()
            key = validation_key(probability, val_y)
            if key > best_key:
                best_key = key
                best = copy.deepcopy(model.state_dict())

    assert best is not None
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        return model(test_x.to(device))[:, 0].sigmoid().cpu()


def fit_mlp_once(
    fit_agent: Tensor,
    fit_y: Tensor,
    val_agent: Tensor,
    val_y: Tensor,
    test_agent: Tensor,
    config: Config,
    *,
    seed: int,
    steps: int,
    lr: float,
    weight_decay: float,
) -> Tensor:
    torch.manual_seed(seed)
    heads = Heads(config).to(config.device)

    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    for parameter in heads.model_body.parameters():
        parameter.requires_grad_(True)
    for parameter in heads.continuation.parameters():
        parameter.requires_grad_(True)

    params = [p for p in heads.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay
    )

    best_key = (-1.0, float("-inf"))
    best_body = None
    best_continuation = None

    x = fit_agent[:, None].to(config.device)
    y = fit_y.to(config.device)

    for step in range(steps):
        logits = continuation_death_logits(heads, x).flatten()
        loss = weighted_bce(logits, y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % 10 == 0 or step == steps - 1:
            heads.eval()
            with torch.no_grad():
                p = continuation_death_logits(
                    heads, val_agent[:, None].to(config.device)
                ).flatten().sigmoid().cpu()
            heads.train()
            key = validation_key(p, val_y)
            if key > best_key:
                best_key = key
                best_body = copy.deepcopy(heads.model_body.state_dict())
                best_continuation = copy.deepcopy(
                    heads.continuation.state_dict()
                )

    assert best_body is not None and best_continuation is not None
    heads.model_body.load_state_dict(best_body)
    heads.continuation.load_state_dict(best_continuation)
    heads.eval()

    with torch.no_grad():
        return continuation_death_logits(
            heads, test_agent[:, None].to(config.device)
        ).flatten().sigmoid().cpu()


def grouped_probe(
    x: Tensor,
    target: Tensor,
    group: Tensor,
    *,
    seeds: list[int],
    device: str,
    steps: int,
    lr: float,
    weight_decay: float,
) -> dict:
    """One whole pre-action state is test; another is validation."""
    groups = sorted(set(group.tolist()))
    if len(groups) < 3:
        raise RuntimeError("Need at least three opportunity states for grouped probing")

    prediction = torch.zeros_like(target)
    per_state = {}

    for test_group in groups:
        remaining = [g for g in groups if g != test_group]
        test_mask = group == test_group
        seed_predictions = []

        for seed_index, seed in enumerate(seeds):
            val_group = remaining[seed_index % len(remaining)]
            fit_mask = (group != test_group) & (group != val_group)
            val_mask = group == val_group

            seed_predictions.append(
                fit_linear_once(
                    x[fit_mask],
                    target[fit_mask],
                    x[val_mask],
                    target[val_mask],
                    x[test_mask],
                    seed=seed + test_group * 101,
                    device=device,
                    steps=steps,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            )

        mean_prediction = torch.stack(seed_predictions).mean(0)
        prediction[test_mask] = mean_prediction
        per_state[str(test_group)] = binary_metrics(
            mean_prediction, target[test_mask]
        )

    return {
        "pooled": binary_metrics(prediction, target),
        "mean_state_auc": float(
            torch.tensor([row["auc"] for row in per_state.values()]).mean()
        ),
        "min_state_auc": min(row["auc"] for row in per_state.values()),
        "max_state_auc": max(row["auc"] for row in per_state.values()),
        "per_state": per_state,
    }


def grouped_mlp_probe(
    agent: Tensor,
    target: Tensor,
    group: Tensor,
    config: Config,
    *,
    seeds: list[int],
    steps: int,
    lr: float,
    weight_decay: float,
) -> dict:
    groups = sorted(set(group.tolist()))
    prediction = torch.zeros_like(target)
    per_state = {}

    for test_group in groups:
        remaining = [g for g in groups if g != test_group]
        test_mask = group == test_group
        seed_predictions = []

        for seed_index, seed in enumerate(seeds):
            val_group = remaining[seed_index % len(remaining)]
            fit_mask = (group != test_group) & (group != val_group)
            val_mask = group == val_group

            seed_predictions.append(
                fit_mlp_once(
                    agent[fit_mask],
                    target[fit_mask],
                    agent[val_mask],
                    target[val_mask],
                    agent[test_mask],
                    config,
                    seed=seed + test_group * 101,
                    steps=steps,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            )

        mean_prediction = torch.stack(seed_predictions).mean(0)
        prediction[test_mask] = mean_prediction
        per_state[str(test_group)] = binary_metrics(
            mean_prediction, target[test_mask]
        )

    return {
        "pooled": binary_metrics(prediction, target),
        "mean_state_auc": float(
            torch.tensor([row["auc"] for row in per_state.values()]).mean()
        ),
        "min_state_auc": min(row["auc"] for row in per_state.values()),
        "max_state_auc": max(row["auc"] for row in per_state.values()),
        "per_state": per_state,
    }


def action_only(data: ForkData, n_actions: int) -> Tensor:
    return F.one_hot(data.action, num_classes=n_actions).float()


def latent_error(data: ForkData) -> dict[str, float]:
    mse = (
        data.generated_latent - data.observed_latent
    ).pow(2).flatten(1).mean(1)
    dead = data.target.bool()
    return {
        "all": float(mse.mean()),
        "fatal": float(mse[dead].mean()),
        "safe": float(mse[~dead].mean()),
        "fatal_over_safe": float(
            mse[dead].mean() / mse[~dead].mean().clamp(min=1e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--linear-steps", type=int, default=600)
    parser.add_argument("--linear-lr", type=float, default=3e-3)
    parser.add_argument("--linear-weight-decay", type=float, default=1e-3)
    parser.add_argument("--mlp-steps", type=int, default=800)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/counterfactual_localization.json"),
    )
    args = parser.parse_args()

    base = Config()
    config = replace(
        base,
        transition="direct",
        time_mixer="attention",
    )

    saved = torch.load(args.forks, weights_only=False)
    encoder, world, heads = load_models(
        args.phase1a, args.phase2, base, config
    )

    data, replay = extract_exact_forks(
        saved, encoder, world, heads, config
    )

    print(
        f"replayed {replay['terminal_opportunity_states']} opportunity states, "
        f"{replay['examples']} state-action examples",
        flush=True,
    )

    seeds = [config.seed + 3000 + i for i in range(args.seeds)]

    production = {
        "generated": binary_metrics(
            data.production_generated, data.target
        ),
        "observed": binary_metrics(
            data.production_observed, data.target
        ),
    }

    linear = {}
    for name, tensor in (
        ("observed_latent", data.observed_latent),
        ("observed_readout", data.observed_readout),
        ("generated_latent", data.generated_latent),
        ("generated_readout", data.generated_readout),
        ("action_only", action_only(data, config.n_actions)),
    ):
        print(f"grouped linear probe: {name}", flush=True)
        linear[name] = grouped_probe(
            flatten(tensor),
            data.target,
            data.group,
            seeds=seeds,
            device=config.device,
            steps=args.linear_steps,
            lr=args.linear_lr,
            weight_decay=args.linear_weight_decay,
        )

    print("grouped production-shaped MLP: observed_readout", flush=True)
    mlp_observed = grouped_mlp_probe(
        data.observed_readout,
        data.target,
        data.group,
        config,
        seeds=seeds,
        steps=args.mlp_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
    )

    print("grouped production-shaped MLP: generated_readout", flush=True)
    mlp_generated = grouped_mlp_probe(
        data.generated_readout,
        data.target,
        data.group,
        config,
        seeds=seeds,
        steps=args.mlp_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
    )

    report = {
        "contract": {
            "arm": "direct-attention",
            "uses_exact_saved_gate_states": True,
            "all_17_actions_per_state": True,
            "test_split": "leave-one-pre-action-state-out",
            "validation_split": "one different whole pre-action state",
            "validation_rotates_across_probe_seeds": True,
            "probe_seeds": args.seeds,
        },
        "replay": replay,
        "production_head": production,
        "linear_probes": linear,
        "production_shaped_mlp": {
            "observed_readout": mlp_observed,
            "generated_readout": mlp_generated,
        },
        "latent_generation_error": latent_error(data),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
