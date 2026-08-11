"""Freeze the S76 Direct world and vary only continuation terminal routing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from artifacts.localize_counterfactual import binary_metrics
from artifacts.localize_counterfactual_interaction import report_score
from artifacts.localize_matched_counterfactual import extract_matched_forks
from artifacts.run_stage_a import ARCHIVE, SUPPORT, corpus
from d4mj.agent import Heads, head_loss, head_targets
from d4mj.checkpoint import load, save
from d4mj.config import Config
from d4mj.data import sample_batch, sample_terminal_batch
from d4mj.train import (
    _balance,
    _generators,
    _to,
    _update,
    generator_state,
    optimizer,
    train_representation,
)
from d4mj.transition import World, transition_loss

VARIANTS = ("generated_only", "observed_only", "shared_paired")
ROOT = Path(__file__).resolve().parent.parent


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _implementation_digests() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "d4mj" / "agent.py",
        ROOT / "d4mj" / "config.py",
        ROOT / "d4mj" / "data.py",
        ROOT / "d4mj" / "train.py",
        ROOT / "d4mj" / "transition.py",
    )
    return {str(path.relative_to(ROOT)): _file_digest(path) for path in paths}


def continuation_logits(heads: Heads, agent: torch.Tensor) -> torch.Tensor:
    pooled = agent.mean(dim=2)
    return heads.continuation(heads.model_body(pooled))


def terminal_objectives(
    generated_logits: torch.Tensor,
    observed_logits: torch.Tensor,
    continuation: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The three arms differ only in which tail domain supplies alive/dead BCE."""
    generated_logits = generated_logits[:, -2:]
    observed_logits = observed_logits[:, -2:]
    continuation = continuation[:, -2:]
    valid = valid[:, -2:]
    assert bool((valid > 0).all())
    assert bool((continuation[:, -2, 0] == 1).all())
    assert bool((continuation[:, -1, 0] == 0).all())

    def score(logits: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, continuation, reduction="none"
        )
        return (loss * valid).sum() / valid.sum().clamp(min=1.0)

    generated = score(generated_logits)
    observed = score(observed_logits)
    return {
        "generated_only": generated,
        "observed_only": observed,
        "shared_paired": (generated + observed) / 2,
    }


def _configure_head(heads: Heads) -> None:
    """Match the production reward/continuation trunk and freeze unrelated heads."""
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    for module in (heads.model_body, heads.reward, heads.continuation):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _checkpoint_objects(heads, optimisers, balances, streams, meta) -> dict:
    objects = {
        "balances": balances,
        "streams": streams,
        "meta": meta,
    }
    for name in VARIANTS:
        objects[f"head_{name}"] = heads[name]
        objects[f"optimiser_{name}"] = optimisers[name]
    return objects


def train_heads(
    episodes,
    world: World,
    config: Config,
    *,
    steps: int,
    world_steps: int,
    checkpoint: Path,
    contract: str,
) -> tuple[dict[str, Heads], dict]:
    """One frozen-world pass feeds all variants, giving them identical batches."""
    torch.manual_seed(config.seed + 2)
    reference = Heads(config).to(config.device)
    heads = {name: copy.deepcopy(reference) for name in VARIANTS}
    del reference
    for module in heads.values():
        _configure_head(module)

    initial = {_module_digest(module) for module in heads.values()}
    assert len(initial) == 1, "continuation variants did not start identically"
    initial_digest = initial.pop()

    optimisers = {name: optimizer([heads[name]], config) for name in VARIANTS}
    balances = {name: {} for name in VARIANTS}
    sampler, model_rng = _generators(config, 2)
    stream_state: dict = {}
    meta: dict = {}
    resume = 0

    if checkpoint.exists():
        load(
            checkpoint,
            config,
            **_checkpoint_objects(
                heads, optimisers, balances, stream_state, meta
            ),
        )
        if meta.get("contract") != contract:
            raise ValueError("frozen-head checkpoint contract changed")
        resume = int(meta["step"])
        sampler.set_state(stream_state["sampler"])
        model_rng.set_state(stream_state["model"])

    world.eval()
    frozen_digest = _module_digest(world)
    for step in range(resume, steps):
        main = _to(
            sample_batch(episodes, sampler, config, step, steps, mixture=True),
            config.device,
        )
        terminal = _to(
            sample_terminal_batch(episodes, sampler, config, step, steps),
            config.device,
        )
        with torch.no_grad():
            _, main_agent = transition_loss(
                world,
                main,
                model_rng,
                config,
                return_agent=True,
                step=world_steps + step,
            )
            _, generated_agent, observed_agent = transition_loss(
                world,
                terminal,
                model_rng,
                config,
                return_agent=True,
                return_observed=True,
                step=world_steps + step,
            )

        main_targets = head_targets(main, config)
        terminal_targets = head_targets(terminal, config)
        for name in VARIANTS:
            module = heads[name]
            main_losses = head_loss(
                module(main_agent) | {"centers": module.centers},
                main_targets,
                config,
            )
            tail = terminal_objectives(
                continuation_logits(module, generated_agent),
                continuation_logits(module, observed_agent),
                terminal_targets["continuation"],
                terminal_targets["continuation_valid"],
            )[name]
            continuation = (
                (1.0 - config.terminal_loss_mass) * main_losses["continuation"]
                + config.terminal_loss_mass * tail
            )
            loss = _balance(
                {
                    "reward": main_losses["reward"],
                    "continuation": continuation,
                },
                balances[name],
                config,
            )
            _update(optimisers[name], loss, [module], config, step)

        if (step + 1) % 100 == 0 or step + 1 == steps:
            values = " ".join(
                f"{name}={balances[name]['continuation'] ** 0.5:.4f}"
                for name in VARIANTS
            )
            print(f"step {step + 1}/{steps} continuation_rms {values}", flush=True)

        if (step + 1) % config.checkpoint_every == 0 or step + 1 == steps:
            meta = {"contract": contract, "step": step + 1}
            stream_state = generator_state(sampler=sampler, model=model_rng)
            save(
                checkpoint,
                config,
                **_checkpoint_objects(
                    heads, optimisers, balances, stream_state, meta
                ),
            )

    assert _module_digest(world) == frozen_digest, "frozen world weights moved"
    for module in heads.values():
        module.eval()
    return heads, {
        "steps": steps,
        "resumed_from": resume,
        "initial_head_sha256": initial_digest,
        "frozen_world_sha256": frozen_digest,
        "final_head_sha256": {
            name: _module_digest(heads[name]) for name in VARIANTS
        },
        "trainable": [
            name
            for name, parameter in heads[VARIANTS[0]].named_parameters()
            if parameter.requires_grad
        ],
        "final_rms": {
            name: {key: value**0.5 for key, value in balances[name].items()}
            for name in VARIANTS
        },
    }


@torch.no_grad()
def death_score(heads: Heads, agent: torch.Tensor, device: str) -> torch.Tensor:
    scores = []
    for start in range(0, len(agent), 256):
        batch = agent[start : start + 256, None].to(device)
        logits = continuation_logits(heads, batch)
        scores.append((1.0 - logits[:, -1, 0].sigmoid()).cpu())
    return torch.cat(scores)


def score_variant(
    heads: Heads,
    generated: torch.Tensor,
    observed: torch.Tensor,
    target: torch.Tensor,
    action: torch.Tensor,
    config: Config,
    *,
    permutations: int,
    seed: int,
) -> dict:
    heads.to(config.device).eval()
    generated_score = death_score(heads, generated, config.device)
    observed_score = death_score(heads, observed, config.device)
    heads.cpu()
    return {
        "generated": {
            "binary": binary_metrics(generated_score, target),
            "same_action": report_score(
                generated_score,
                target,
                action,
                permutations=permutations,
                seed=seed,
            ),
        },
        "observed": {
            "binary": binary_metrics(observed_score, target),
            "same_action": report_score(
                observed_score,
                target,
                action,
                permutations=permutations,
                seed=seed + 1,
            ),
        },
    }


def _validate_reference(
    data,
    replay: dict,
    production: Heads,
    report: dict,
    reference_report: Path,
    reference_features: Path,
    config: Config,
) -> dict:
    expected_report = json.loads(reference_report.read_text())
    expected = expected_report["same_action_interaction"]
    expected_features = torch.load(reference_features, weights_only=False)
    production.to(config.device).eval()

    for field in ("target", "action", "group"):
        if not torch.equal(getattr(data, field), expected_features[field]):
            raise AssertionError(f"reference {field} changed")

    errors = {}
    feature_errors = {}
    for domain in ("generated", "observed"):
        for feature in ("latent", "readout"):
            field = f"{domain}_{feature}"
            error = float(
                (getattr(data, field) - expected_features[field]).abs().max()
            )
            feature_errors[field] = error
            if error > 1e-5:
                raise AssertionError(
                    f"reference {field} changed: {error:.3e}"
                )

        current = getattr(data, f"production_{domain}")
        recorded = expected_features[f"production_{domain}"]
        errors[domain] = float((current - recorded).abs().max())
        if errors[domain] > 1e-5:
            raise AssertionError(
                f"production {domain} predictions changed: {errors[domain]:.3e}"
            )

        rescored = death_score(
            production, getattr(data, f"{domain}_readout"), config.device
        )
        error = float((rescored - current).abs().max())
        if error > 1e-6:
            raise AssertionError(
                f"production {domain} readout was not replayed: {error:.3e}"
            )

        got = report[domain]["same_action"]["conditional"]["pooled_pair_auc"]
        want = expected[domain]["pooled_pair_auc"]
        if abs(got - want) > 1e-6:
            raise AssertionError(
                f"production {domain} AUC differs: {got:.8f} != {want:.8f}"
            )

    expected_states = expected_report["contract"]["terminal_opportunity_states"]
    if replay["terminal_opportunity_states"] != expected_states:
        raise AssertionError("terminal-opportunity state count changed")
    if replay["examples"] != expected_states * config.n_actions:
        raise AssertionError("fork example count changed")
    production.cpu()
    return {
        "all_labels_actions_groups_equal": True,
        "max_abs_feature_error": feature_errors,
        "max_abs_prediction_error": errors,
        "same_action_auc_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--world-phase2", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--tokenizer-steps", type=int, default=3000)
    parser.add_argument("--world-steps", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    config = Config(transition="direct", time_mixer="attention")
    base = Config()
    train_set, _ = corpus(base, args.expert, print)
    encoder, decoder, cached_train = train_representation(
        train_set,
        args.tokenizer_steps,
        base,
        checkpoint=args.phase1a,
    )
    decoder.cpu()
    encoder.cpu()
    del decoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    world = World(config).to(config.device)
    production = Heads(config).to(config.device)
    load(args.world_phase2, config, part0=world, part1=production)
    world.eval()
    production.eval()
    for module in (world, production):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    contract = json.dumps(
        {
            "version": "frozen-continuation-domains-v1",
            "steps": args.steps,
            "world_steps": args.world_steps,
            "phase1a": _file_digest(args.phase1a),
            "world_phase2": _file_digest(args.world_phase2),
            "expert_manifest": _file_digest(Path(f"{ARCHIVE}.manifest.json")),
            "support_manifest": _file_digest(Path(f"{SUPPORT}.manifest.json")),
            "implementation": _implementation_digests(),
            "variants": VARIANTS,
            "main": "identical Phase-2 reward and continuation likelihood",
            "trainable": "model_body+reward+continuation",
        },
        sort_keys=True,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    heads, training = train_heads(
        cached_train,
        world,
        config,
        steps=args.steps,
        world_steps=args.world_steps,
        checkpoint=args.checkpoint,
        contract=contract,
    )
    for module in heads.values():
        module.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    encoder.to(config.device).eval()
    trajectory_world = World(config).to(config.device)
    trajectory_heads = Heads(config).to(config.device)
    load(
        args.trajectory_phase2,
        config,
        part0=trajectory_world,
        part1=trajectory_heads,
    )
    trajectory_world.eval()
    trajectory_heads.eval()
    saved = torch.load(args.forks, weights_only=False)
    data, replay = extract_matched_forks(
        saved,
        encoder,
        trajectory_world,
        trajectory_heads,
        world,
        production,
        config,
        config,
    )

    production_report = score_variant(
        production,
        data.generated_readout,
        data.observed_readout,
        data.target,
        data.action,
        config,
        permutations=args.permutations,
        seed=config.seed + 6000,
    )
    reference_validation = _validate_reference(
        data,
        replay,
        production,
        production_report,
        args.reference_report,
        args.reference_features,
        config,
    )
    encoder.cpu()
    trajectory_world.cpu()
    trajectory_heads.cpu()
    world.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    variants = {
        name: score_variant(
            heads[name],
            data.generated_readout,
            data.observed_readout,
            data.target,
            data.action,
            config,
            permutations=args.permutations,
            seed=config.seed + 6100 + index * 10,
        )
        for index, name in enumerate(VARIANTS)
    }

    report = {
        "contract": json.loads(contract),
        "inputs": {
            "phase1a": str(args.phase1a.resolve()),
            "world_phase2": str(args.world_phase2.resolve()),
            "trajectory_phase2": str(args.trajectory_phase2.resolve()),
            "forks": str(args.forks.resolve()),
            "reference_report": str(args.reference_report.resolve()),
            "reference_features": str(args.reference_features.resolve()),
            "phase1a_sha256": _file_digest(args.phase1a),
            "world_phase2_sha256": _file_digest(args.world_phase2),
            "trajectory_phase2_sha256": _file_digest(args.trajectory_phase2),
            "forks_sha256": _file_digest(args.forks),
            "reference_report_sha256": _file_digest(args.reference_report),
            "reference_features_sha256": _file_digest(args.reference_features),
            "checkpoint_sha256": _file_digest(args.checkpoint),
        },
        "replay": replay,
        "reference_validation": reference_validation,
        "training": training,
        "production_control": production_report,
        "variants": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
