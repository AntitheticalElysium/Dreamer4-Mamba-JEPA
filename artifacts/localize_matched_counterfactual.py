"""Localize terminal prediction on states fixed by a baseline policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from artifacts.localize_counterfactual import (
    ForkData,
    binary_metrics,
    flatten,
    latent_error,
    load_models,
)
from artifacts.localize_counterfactual_interaction import (
    centered_oof_linear,
    oof_mlp,
    report_score,
)
from artifacts.evaluate_matched_counterfactual import ARMS, arm_config
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.transition import advance, observe


@torch.no_grad()
def extract_matched_forks(
    saved,
    encoder,
    trajectory_world,
    trajectory_heads,
    world,
    heads,
    trajectory_config,
    evaluation_config,
    *,
    verify_saved_predictions=False,
) -> tuple[ForkData, dict]:
    if evaluation_config.transition != "direct":
        raise ValueError("feature localization is deterministic-Direct only; use the S35 probe for Flow")
    if trajectory_config.device != evaluation_config.device:
        raise ValueError("trajectory and evaluation arms must use one device")
    device = evaluation_config.device
    terminal_opportunity = saved["true_death"].any(1) & (~saved["true_death"]).any(1)
    selected = terminal_opportunity.nonzero().flatten().tolist()
    if not selected:
        raise RuntimeError("saved forks contain no terminal-opportunity states")

    key_to_row = {
        (int(saved["seed"][row]), int(saved["step"][row])): row
        for row in selected
    }
    if len(key_to_row) != len(selected):
        raise RuntimeError("saved forks contain duplicate terminal-opportunity states")
    row_to_group = {row: group for group, row in enumerate(selected)}
    by_seed: dict[int, set[int]] = {}
    for seed, step in key_to_row:
        by_seed.setdefault(seed, set()).add(step)

    observed_latent, observed_readout = [], []
    generated_latent, generated_readout = [], []
    labels, actions, groups = [], [], []
    production_generated, production_observed = [], []
    reproduced = set()
    trajectory_action_by_group = {}
    saved_predictions = "model_death" in saved and "observed_death" in saved
    generated_max_error = observed_max_error = 0.0

    for seed in sorted(by_seed):
        wanted = by_seed[seed]
        last = max(wanted)
        observation, env_state = reset(seed)
        trajectory_state = evaluation_state = None
        incoming = torch.full(
            (1, 1), evaluation_config.n_actions, dtype=torch.long, device=device
        )
        trajectory_rng = torch.Generator(device=device).manual_seed(seed + 2**21)
        evaluation_rng = torch.Generator(device=device).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=device).manual_seed(seed + 2**20)

        for index in range(last + 1):
            patches = patchify(observation[None, None], evaluation_config.patch).to(device)
            trajectory_state, trajectory_agent = observe(
                trajectory_world,
                encoder,
                trajectory_state,
                incoming,
                patches,
                trajectory_rng,
                trajectory_config,
            )
            evaluation_state, _ = observe(
                world,
                encoder,
                evaluation_state,
                incoming,
                patches,
                evaluation_rng,
                evaluation_config,
            )

            key = (seed, index)
            if key in key_to_row:
                row = key_to_row[key]
                group = row_to_group[row]
                for action in range(evaluation_config.n_actions):
                    successor, _, _, terminated, _ = env_step(
                        env_state, action, seed + index + 1
                    )
                    if bool(terminated) != bool(saved["true_death"][row, action]):
                        raise AssertionError(
                            f"truth mismatch seed={seed} step={index} action={action}"
                        )

                    chosen = torch.tensor([[action]], device=device)
                    generated_rng = torch.Generator(device=device).manual_seed(
                        evaluation_config.seed + 2**23 + seed * 4099 + index * 17
                    )
                    generated_state, generated_agent = advance(
                        world,
                        evaluation_state.world,
                        chosen,
                        generated_rng,
                        evaluation_config,
                    )
                    generated_death = float(
                        1.0
                        - heads(generated_agent)["continuation"][:, -1, 0].sigmoid()[0]
                    )

                    observed_rng = torch.Generator(device=device).manual_seed(
                        evaluation_config.seed + 2**24 + seed * 4099 + index * 17
                    )
                    successor_patches = patchify(
                        successor[None, None], evaluation_config.patch
                    ).to(device)
                    observed_state, observed_agent = observe(
                        world,
                        encoder,
                        evaluation_state,
                        chosen,
                        successor_patches,
                        observed_rng,
                        evaluation_config,
                    )
                    observed_death = float(
                        1.0
                        - heads(observed_agent)["continuation"][:, -1, 0].sigmoid()[0]
                    )

                    if saved_predictions:
                        generated_max_error = max(
                            generated_max_error,
                            abs(generated_death - float(saved["model_death"][row, action])),
                        )
                        observed_max_error = max(
                            observed_max_error,
                            abs(observed_death - float(saved["observed_death"][row, action])),
                        )
                    observed_latent.append(observed_state.world.latent[0, -1].cpu())
                    observed_readout.append(observed_agent[0, -1].cpu())
                    generated_latent.append(generated_state.latent[0, -1].cpu())
                    generated_readout.append(generated_agent[0, -1].cpu())
                    labels.append(float(terminated))
                    actions.append(action)
                    groups.append(group)
                    production_generated.append(generated_death)
                    production_observed.append(observed_death)
                reproduced.add(key)

            logits = trajectory_heads(trajectory_agent)["policy"][:, -1, 0]
            action = int(
                torch.multinomial(logits.softmax(-1), 1, generator=policy_rng)
            )
            if key in key_to_row:
                group = row_to_group[key_to_row[key]]
                trajectory_action_by_group[group] = action
            observation, env_state, _, terminated, truncated = env_step(
                env_state, action, seed + index + 1
            )
            incoming.fill_(action)
            if terminated or truncated:
                if index < last:
                    missing = sorted(step for step in wanted if step > index)
                    raise RuntimeError(
                        f"baseline trajectory ended before saved steps: {seed=} {missing=}"
                    )
                break

    missing = sorted(set(key_to_row) - reproduced)
    if missing:
        raise RuntimeError(f"failed to reproduce terminal-opportunity states: {missing}")
    if set(trajectory_action_by_group) != set(range(len(selected))):
        raise RuntimeError("failed to reproduce the trajectory action for every fork state")
    if verify_saved_predictions and not saved_predictions:
        raise ValueError("saved prediction verification requested without saved predictions")
    if verify_saved_predictions and (
        generated_max_error > 1e-5 or observed_max_error > 1e-5
    ):
        raise AssertionError(
            "production replay mismatch: "
            f"generated={generated_max_error:.3e}, observed={observed_max_error:.3e}"
        )

    data = ForkData(
        observed_latent=torch.stack(observed_latent),
        observed_readout=torch.stack(observed_readout),
        generated_latent=torch.stack(generated_latent),
        generated_readout=torch.stack(generated_readout),
        target=torch.tensor(labels),
        action=torch.tensor(actions),
        group=torch.tensor(groups),
        production_generated=torch.tensor(production_generated),
        production_observed=torch.tensor(production_observed),
    )
    replay = {
        "terminal_opportunity_states": len(selected),
        "examples": len(labels),
        "truth_replayed_exactly": True,
        "generated_prediction_max_abs_error_vs_saved": generated_max_error,
        "observed_prediction_max_abs_error_vs_saved": observed_max_error,
        "saved_prediction_equality_required": verify_saved_predictions,
        "saved_predictions_available": saved_predictions,
        "trajectory_action_by_group": [
            trajectory_action_by_group[group] for group in range(len(selected))
        ],
        "trajectory_action_definition": (
            "action sampled by the fixed trajectory policy and executed from this state"
        ),
    }
    return data, replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--eval-phase2", type=Path, required=True)
    parser.add_argument("--trajectory-arm", choices=ARMS, default="direct-attention")
    parser.add_argument("--eval-arm", choices=ARMS, default="direct-attention")
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--linear-steps", type=int, default=600)
    parser.add_argument("--mlp-steps", type=int, default=800)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--verify-saved-predictions", action="store_true")
    args = parser.parse_args()

    base = Config()
    trajectory_config = arm_config(base, args.trajectory_arm)
    config = arm_config(base, args.eval_arm)
    encoder, trajectory_world, trajectory_heads = load_models(
        args.phase1a, args.trajectory_phase2, base, trajectory_config
    )
    _, world, heads = load_models(args.phase1a, args.eval_phase2, base, config)
    saved = torch.load(args.forks, weights_only=False)
    data, replay = extract_matched_forks(
        saved,
        encoder,
        trajectory_world,
        trajectory_heads,
        world,
        heads,
        trajectory_config,
        config,
        verify_saved_predictions=args.verify_saved_predictions,
    )
    print(
        f"exact matched replay: {replay['terminal_opportunity_states']} states, "
        f"{replay['examples']} examples",
        flush=True,
    )
    args.features.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vars(data), args.features)

    seeds = [config.seed + 4000 + index for index in range(args.seeds)]
    predictions = {
        "production_generated": data.production_generated,
        "production_observed": data.production_observed,
        "action_identity_only": data.action.float(),
    }
    for name, feature in (
        ("observed_latent", data.observed_latent),
        ("observed_readout", data.observed_readout),
        ("generated_latent", data.generated_latent),
        ("generated_readout", data.generated_readout),
    ):
        print(f"action-centered OOF linear: {name}", flush=True)
        predictions[f"linear_{name}"] = centered_oof_linear(
            feature,
            data.target,
            data.action,
            data.group,
            config,
            seeds=seeds,
            steps=args.linear_steps,
            lr=3e-3,
            weight_decay=1e-3,
        )

    for name, feature in (
        ("observed_readout", data.observed_readout),
        ("generated_readout", data.generated_readout),
    ):
        print(f"OOF production-shaped MLP: {name}", flush=True)
        predictions[f"mlp_{name}"] = oof_mlp(
            feature,
            data.target,
            data.group,
            config,
            seeds=seeds,
            steps=args.mlp_steps,
            lr=1e-3,
            weight_decay=1e-4,
        )

    scores = {}
    for index, (name, score) in enumerate(predictions.items()):
        print(f"conditional metric: {name}", flush=True)
        scores[name] = report_score(
            score,
            data.target,
            data.action,
            permutations=args.permutations,
            seed=config.seed + 5000 + index,
        )

    report = {
        "contract": {
            "trajectory_policy": str(args.trajectory_phase2.resolve()),
            "trajectory_arm": args.trajectory_arm,
            "evaluated_model": str(args.eval_phase2.resolve()),
            "evaluated_arm": args.eval_arm,
            "uses_exact_matched_states": True,
            "all_17_actions_replayed": True,
            "evaluation": "dead-vs-safe comparisons within the same action",
            "test_split": "leave-one-pre-action-state-out",
            "probe_seeds": args.seeds,
            "permutations": args.permutations,
        },
        "replay": replay,
        "production_head": {
            "generated": binary_metrics(data.production_generated, data.target),
            "observed": binary_metrics(data.production_observed, data.target),
        },
        "latent_generation_error": latent_error(data),
        "scores": scores,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
