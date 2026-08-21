"""Train the matched generated-latent terminal-gradient ablation."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    cached_train,
    data_digests,
    file_digest,
    implementation_digests,
    state_digest,
)
from artifacts.train_phase1b_geometry_factorial import (
    admit_terminal_batch,
    combine_strata,
    tensor_digest,
)
from artifacts.train_predictor_flow_attribution import validate_direct_control
from artifacts.train_terminal_diversity_scaling import (
    balanced_terminal_schedule,
    stratified_terminal_ranking,
    terminal_metadata,
    terminal_tail_batch,
)
from d4mj.checkpoint import load, save
from d4mj.config import Config
from d4mj.data import sample_batch
from d4mj.state import WorldState
from d4mj.train import (
    _balance,
    _generators,
    _share_initialisation,
    _to,
    generator_state,
    optimizer,
)
from d4mj.transition import World, advance, commit_inputs, transition_loss


VARIANTS = ("allowed", "stopped")
SOURCE_PAPERS = (
    Path("third_party/papers/2509.24527v1.pdf"),
    Path("third_party/papers/2410.08893v4.pdf"),
    Path("third_party/papers/2603.02765v1.pdf"),
    Path("third_party/papers/2603.07083v2.pdf"),
    Path("third_party/papers/2506.09985v1.pdf"),
    Path("third_party/papers/2606.27326v1.pdf"),
)


class LatentContinuationHead(nn.Module):
    """A small continuation readout over one packed successor latent."""

    def __init__(self, config: Config):
        super().__init__()
        width = config.n_spatial * config.d_spatial
        self.net = nn.Sequential(
            nn.Linear(width, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent.flatten(-2)).squeeze(-1)


def generated_terminal_successors(
    world: World,
    batch,
    rng: torch.Generator,
    config: Config,
) -> torch.Tensor:
    """Return Direct's two recursively generated terminal-tail successors."""
    if config.transition != "direct" or batch.latents.shape[1] < 3:
        raise ValueError("terminal successor extraction requires Direct and T >= 3")
    committed, conditioning = commit_inputs(batch.latents, rng, config)
    prefix, _, memory = world(
        None,
        batch.led_to_action[:, :-2],
        committed[:, :-2],
        conditioning[:, :-2],
    )
    length = batch.latents.shape[1]
    state = WorldState(
        batch.latents[:, -3:-2],
        memory,
        length - 2,
        prefix[:, -1:],
    )
    first, _ = advance(
        world,
        state,
        batch.led_to_action[:, -2:-1],
        rng,
        config,
    )
    second, _ = advance(
        world,
        first,
        batch.led_to_action[:, -1:],
        rng,
        config,
    )
    return torch.cat([first.latent, second.latent], dim=1)


def terminal_objective(
    world: World,
    head: LatentContinuationHead,
    batch,
    rng: torch.Generator,
    config: Config,
    *,
    stop_generated: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score matching observed/generated alive-dead successor pairs."""
    predicted = generated_terminal_successors(world, batch, rng, config)
    observed = batch.latents[:, -2:]
    target = (~batch.terminated[:, -2:]).float()
    valid = batch.valid[:, -2:]
    if not bool(valid.all()):
        raise AssertionError("terminal outcome pair contains padding")
    if not bool(target[:, 0].eq(1).all() and target[:, 1].eq(0).all()):
        raise AssertionError("terminal outcome pair is not alive then dead")
    generated_input = predicted.detach() if stop_generated else predicted
    generated_logits = head(generated_input)
    observed_logits = head(observed)
    generated_loss = F.binary_cross_entropy_with_logits(generated_logits, target)
    observed_loss = F.binary_cross_entropy_with_logits(observed_logits, target)
    return (generated_loss + observed_loss) / 2, {
        "predicted": predicted,
        "observed": observed,
        "target": target,
        "generated_logits": generated_logits,
        "observed_logits": observed_logits,
    }


def gradient_preflight(
    world: World,
    head: LatentContinuationHead,
    batch,
    config: Config,
) -> dict[str, dict[str, float]]:
    result = {}
    for stopped in (False, True):
        rng = torch.Generator(device=config.device).manual_seed(config.seed + 18_101)
        loss, values = terminal_objective(
            world,
            head,
            batch,
            rng,
            config,
            stop_generated=stopped,
        )
        world_gradients = torch.autograd.grad(
            loss,
            tuple(world.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
        latent_gradient = torch.autograd.grad(
            loss,
            values["predicted"],
            allow_unused=True,
        )[0]
        result["stopped" if stopped else "allowed"] = {
            "world_gradient_norm": sum(
                float(value.detach().pow(2).sum())
                for value in world_gradients
                if value is not None
            )
            ** 0.5,
            "generated_latent_gradient_norm": (
                0.0
                if latent_gradient is None
                else float(latent_gradient.detach().pow(2).sum().sqrt())
            ),
        }
    if not result["allowed"]["world_gradient_norm"] > 0.0:
        raise AssertionError("allowed terminal objective does not reach the world")
    if not result["allowed"]["generated_latent_gradient_norm"] > 0.0:
        raise AssertionError("allowed terminal objective does not reach generated latent")
    if result["stopped"] != {
        "world_gradient_norm": 0.0,
        "generated_latent_gradient_norm": 0.0,
    }:
        raise AssertionError(f"stopped terminal gradient leaked: {result['stopped']}")
    return result


def _set_learning_rate(current, config: Config, step: int) -> None:
    rate = config.learning_rate * min(1.0, (step + 1) / config.warmup)
    for group in current.param_groups:
        group["lr"] = rate


def update_pair(
    world_optimizer,
    head_optimizer,
    loss: torch.Tensor,
    world: World,
    head: LatentContinuationHead,
    config: Config,
    step: int,
) -> None:
    for current in (world_optimizer, head_optimizer):
        _set_learning_rate(current, config, step)
        current.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(world.parameters(), config.grad_clip)
    torch.nn.utils.clip_grad_norm_(head.parameters(), config.grad_clip)
    world_optimizer.step()
    head_optimizer.step()


def checkpoint_objects(
    worlds,
    heads,
    world_optimizers,
    head_optimizers,
    world_balances,
    head_balances,
    streams,
    meta,
) -> dict:
    objects = {
        "world_balances": world_balances,
        "head_balances": head_balances,
        "streams": streams,
        "meta": meta,
    }
    for name in VARIANTS:
        objects[f"world_{name}"] = worlds[name]
        objects[f"head_{name}"] = heads[name]
        objects[f"world_optimizer_{name}"] = world_optimizers[name]
        objects[f"head_optimizer_{name}"] = head_optimizers[name]
    return objects


def train(
    episodes,
    schedule: torch.Tensor,
    config: Config,
    *,
    steps: int,
    milestones: tuple[int, ...],
    checkpoint: Path,
    out: Path,
    contract: dict,
    control: dict | None,
) -> dict:
    torch.manual_seed(config.seed + 1)
    reference_world = _share_initialisation(World(config), config).to(config.device)
    worlds = {name: copy.deepcopy(reference_world) for name in VARIANTS}
    del reference_world
    torch.manual_seed(config.seed + 18_200)
    reference_head = LatentContinuationHead(config).to(config.device)
    heads = {name: copy.deepcopy(reference_head) for name in VARIANTS}
    del reference_head

    initial_world = {state_digest(module) for module in worlds.values()}
    initial_head = {state_digest(module) for module in heads.values()}
    if len(initial_world) != 1 or len(initial_head) != 1:
        raise AssertionError("outcome-shaping cells do not share initialization")
    if control is not None and initial_world != {control["initial_world_sha256"]}:
        raise AssertionError("outcome-shaping world initialization differs from S78")

    world_optimizers = {name: optimizer([worlds[name]], config) for name in VARIANTS}
    head_optimizers = {name: optimizer([heads[name]], config) for name in VARIANTS}
    world_balances = {name: {} for name in VARIANTS}
    head_balances = {name: {} for name in VARIANTS}
    sampler, _ = _generators(config, 1)
    model_rngs = {
        name: torch.Generator(device=config.device).manual_seed(config.seed + 1001)
        for name in VARIANTS
    }
    terminal_rngs = {
        name: torch.Generator(device=config.device).manual_seed(config.seed + 7101)
        for name in VARIANTS
    }
    outcome_rngs = {
        name: torch.Generator(device=config.device).manual_seed(config.seed + 18_301)
        for name in VARIANTS
    }
    streams, meta = {}, {}
    resume = 0
    curves = {name: [] for name in VARIANTS}
    milestones_recorded = {name: {} for name in VARIANTS}
    control_matches = {}

    if checkpoint.exists():
        load(
            checkpoint,
            config,
            **checkpoint_objects(
                worlds,
                heads,
                world_optimizers,
                head_optimizers,
                world_balances,
                head_balances,
                streams,
                meta,
            ),
        )
        if meta.get("contract") != contract:
            raise ValueError("generated-latent outcome checkpoint contract changed")
        resume = int(meta["step"])
        curves = meta["curves"]
        milestones_recorded = meta["milestones"]
        control_matches = meta["control_matches"]
        sampler.set_state(streams["sampler"])
        for name in VARIANTS:
            model_rngs[name].set_state(streams["models"][name])
            terminal_rngs[name].set_state(streams["terminals"][name])
            outcome_rngs[name].set_state(streams["outcomes"][name])

    gate_batch = _to(
        terminal_tail_batch(episodes[int(schedule[0])], config, 0, steps),
        config.device,
    )
    routing = gradient_preflight(worlds["allowed"], heads["allowed"], gate_batch, config)

    raw = {
        name: {"main": [], "tail": [], "outcome": []}
        for name in VARIANTS
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        (out / name).mkdir(parents=True, exist_ok=True)
    for step in range(resume, steps):
        ordinary_batch = _to(
            sample_batch(episodes, sampler, config, step, steps),
            config.device,
        )
        tail_batch = _to(
            terminal_tail_batch(episodes[int(schedule[step])], config, step, steps),
            config.device,
        )
        admitted = admit_terminal_batch(tail_batch)
        for name in VARIANTS:
            ordinary = transition_loss(
                worlds[name], ordinary_batch, model_rngs[name], config, step=step
            )
            tail = transition_loss(
                worlds[name], admitted, terminal_rngs[name], config, step=step
            )
            dynamics = combine_strata(ordinary, tail, config.terminal_loss_mass)
            outcome, _ = terminal_objective(
                worlds[name],
                heads[name],
                tail_batch,
                outcome_rngs[name],
                config,
                stop_generated=name == "stopped",
            )
            world_loss = _balance(
                {"dynamics": dynamics}, world_balances[name], config
            )
            head_loss = _balance(
                {"terminal_outcome": outcome}, head_balances[name], config
            )
            update_pair(
                world_optimizers[name],
                head_optimizers[name],
                world_loss + head_loss,
                worlds[name],
                heads[name],
                config,
                step,
            )
            raw[name]["main"].append(float(ordinary.detach()))
            raw[name]["tail"].append(float(tail.detach()))
            raw[name]["outcome"].append(float(outcome.detach()))

        completed = step + 1
        report_every = min(500, steps)
        if completed % report_every == 0 or completed == steps:
            text = []
            for name in VARIANTS:
                row = {
                    "step": completed,
                    "main_raw_mean": sum(raw[name]["main"]) / len(raw[name]["main"]),
                    "tail_raw_mean": sum(raw[name]["tail"]) / len(raw[name]["tail"]),
                    "outcome_raw_mean": sum(raw[name]["outcome"]) / len(raw[name]["outcome"]),
                    "dynamics_rms": world_balances[name]["dynamics"] ** 0.5,
                    "outcome_rms": head_balances[name]["terminal_outcome"] ** 0.5,
                }
                curves[name].append(row)
                for values in raw[name].values():
                    values.clear()
                text.append(
                    f"{name}=({row['main_raw_mean']:.5f},"
                    f"{row['tail_raw_mean']:.5f},{row['outcome_raw_mean']:.5f})"
                )
            print(f"step {completed}/{steps} " + " ".join(text), flush=True)

        if completed in milestones:
            for name in VARIANTS:
                world_path = out / name / f"world_{completed:06d}.pt"
                model_path = out / name / f"model_{completed:06d}.pt"
                save(
                    world_path,
                    config,
                    part0=worlds[name],
                    experiment=contract,
                    step=completed,
                )
                save(
                    model_path,
                    config,
                    part0=worlds[name],
                    part1=heads[name],
                    experiment=contract,
                    step=completed,
                )
                milestones_recorded[name][str(completed)] = {
                    "world": str(world_path.resolve()),
                    "model": str(model_path.resolve()),
                    "world_sha256": state_digest(worlds[name]),
                    "head_sha256": state_digest(heads[name]),
                }
            if control is not None:
                expected = control["contract"]["milestones"]
                expected_digest = control["milestone_world_sha256"][str(completed)]
                actual = milestones_recorded["stopped"][str(completed)]["world_sha256"]
                matched = actual == expected_digest and completed in expected
                control_matches[str(completed)] = matched
                if not matched:
                    raise AssertionError(
                        f"stopped world diverged from S78 at {completed}: "
                        f"{actual} != {expected_digest}"
                    )

        if completed % config.checkpoint_every == 0 or completed == steps:
            streams = {
                "sampler": sampler.get_state(),
                "models": generator_state(**model_rngs),
                "terminals": generator_state(**terminal_rngs),
                "outcomes": generator_state(**outcome_rngs),
            }
            meta = {
                "contract": contract,
                "step": completed,
                "curves": curves,
                "milestones": milestones_recorded,
                "control_matches": control_matches,
            }
            save(
                checkpoint,
                config,
                **checkpoint_objects(
                    worlds,
                    heads,
                    world_optimizers,
                    head_optimizers,
                    world_balances,
                    head_balances,
                    streams,
                    meta,
                ),
            )

    world_difference = state_digest(worlds["allowed"]) != state_digest(worlds["stopped"])
    if not world_difference:
        raise AssertionError("outcome gradient did not change the allowed world")
    return {
        "contract": contract,
        "resumed_from": resume,
        "initial_world_sha256": initial_world.pop(),
        "initial_head_sha256": initial_head.pop(),
        "gradient_preflight": routing,
        "milestones": milestones_recorded,
        "control_matches": control_matches,
        "worlds_differ": world_difference,
        "training_curves": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--reference-phase1b", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--milestones", type=int, nargs="+", default=(5_000, 10_000, 20_000)
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    milestones = tuple(sorted(set(args.milestones)))
    if milestones[-1] != args.steps:
        parser.error("the final step must be a milestone")
    config = Config(transition="direct", time_mixer="attention")
    control = None
    if not args.smoke:
        control = validate_direct_control(args.control, config, args.steps)
        control_report = json.loads((args.control / "training_report.json").read_text())
        control["milestone_world_sha256"] = control_report[
            "milestone_world_sha256"
        ]
    encoder, episodes = cached_train(
        args.phase1a,
        Config(),
        args.expert,
        support=args.support,
        cache=args.cache,
    )
    encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadata = terminal_metadata(episodes, config)
    ranking_seed = config.seed + 6200
    schedule_seed = config.seed + 6300
    ranking = stratified_terminal_ranking(metadata, ranking_seed)
    selected = ranking[: min(2, len(ranking))] if args.smoke else ranking
    schedule = balanced_terminal_schedule(selected, args.steps, schedule_seed)
    if control is not None:
        if selected != control["selected_episode_indices"]:
            raise AssertionError("terminal episode selection differs from S78")
        if tensor_digest(schedule) != control["schedule_sha256"]:
            raise AssertionError("terminal schedule differs from S78")

    contract = {
        "version": "generated-latent-outcome-shaping-training-v1",
        "phase1a": file_digest(args.phase1a),
        "reference_phase1b": file_digest(args.reference_phase1b),
        "control": None if args.smoke else {
            "report": control["report_sha256"],
            "milestones": control["milestones"],
        },
        "data": data_digests(args.support),
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/train_terminal_diversity_scaling.py"),
            Path("artifacts/train_phase1b_geometry_factorial.py"),
        ),
        "source_papers": {str(path): file_digest(path) for path in SOURCE_PAPERS},
        "steps": args.steps,
        "milestones": milestones,
        "terminal_candidates": len(selected),
        "ranking_seed": ranking_seed,
        "schedule_seed": schedule_seed,
        "schedule_sha256": tensor_digest(schedule),
        "dynamics": (
            "S78 ordinary Phase-1B plus full-diversity terminal-tail MSE at "
            f"mass {config.terminal_loss_mass}"
        ),
        "outcome": (
            "shared 512->256->1 continuation head on matching observed and "
            "recursively generated final-alive/final-dead successor latents"
        ),
        "variants": {
            "allowed": "generated continuation BCE updates head and Direct world",
            "stopped": "same BCE after generated.detach; world receives MSE only",
        },
        "optimization": (
            "separate AdamW optimizers and clipping; dynamics and outcome each "
            "running-RMS normalized before the shared backward"
        ),
        "scope": (
            "experimental Direct boundary; not attributed to Dreamer 4 and no "
            "production architecture is changed"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report = train(
        episodes,
        schedule,
        config,
        steps=args.steps,
        milestones=milestones,
        checkpoint=args.out / "train.pt",
        out=args.out,
        contract=contract,
        control=control,
    )
    atomic_json(args.out / "training_report.json", report)
    print(f"complete: {args.out / 'training_report.json'}", flush=True)


if __name__ == "__main__":
    main()
