"""Ablate whether terminal outcome gradients may shape the Direct world."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    cached_train,
    data_digests,
    file_digest,
    implementation_digests,
    state_digest,
)
from d4mj.agent import Heads, head_targets, paired_terminal_loss
from d4mj.checkpoint import load, save
from d4mj.config import Config
from d4mj.data import sample_batch, sample_terminal_batch
from d4mj.train import (
    _balance,
    _generators,
    _share_initialisation,
    _to,
    generator_state,
    optimizer,
)
from d4mj.transition import World, transition_loss


VARIANTS = ("world_gradient", "stopped_world_gradient")


def _configure_head(heads: Heads) -> None:
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    for module in (heads.model_body, heads.continuation):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _terminal_readout(heads: Heads, agent: torch.Tensor) -> dict[str, torch.Tensor]:
    pooled = agent.mean(dim=2)
    return {"continuation": heads.continuation(heads.model_body(pooled))}


def terminal_objective(
    world: World,
    heads: Heads,
    batch,
    rng: torch.Generator,
    config: Config,
    *,
    stop_world_gradient: bool,
) -> torch.Tensor:
    """Score the generated alive/dead tail pair; optionally detach its world path."""
    _, agent = transition_loss(
        world, batch, rng, config, return_agent=True, step=0
    )
    if stop_world_gradient:
        agent = agent.detach()
    readout = _terminal_readout(heads, agent)
    return paired_terminal_loss(readout, readout, head_targets(batch, config))


def gradient_preflight(world: World, heads: Heads, batch, config: Config) -> dict:
    values = {}
    for stopped in (False, True):
        rng = torch.Generator(device=config.device).manual_seed(config.seed + 9101)
        loss = terminal_objective(
            world,
            heads,
            batch,
            rng,
            config,
            stop_world_gradient=stopped,
        )
        gradients = torch.autograd.grad(
            loss,
            tuple(world.parameters()),
            allow_unused=True,
        )
        norm = sum(
            float(gradient.detach().pow(2).sum())
            for gradient in gradients
            if gradient is not None
        ) ** 0.5
        values["stopped" if stopped else "allowed"] = norm
    if not values["allowed"] > 0.0 or values["stopped"] != 0.0:
        raise AssertionError(f"terminal gradient routing failed: {values}")
    return values


def _update_pair(
    world_optimizer,
    head_optimizer,
    loss: torch.Tensor,
    world: World,
    heads: Heads,
    config: Config,
    step: int,
) -> None:
    scale = config.learning_rate * min(1.0, (step + 1) / config.warmup)
    for current in (world_optimizer, head_optimizer):
        for group in current.param_groups:
            group["lr"] = scale
        current.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(world.parameters(), config.grad_clip)
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in heads.parameters() if parameter.requires_grad],
        config.grad_clip,
    )
    world_optimizer.step()
    head_optimizer.step()


def _objects(worlds, heads, world_optimizers, head_optimizers, balances, streams, meta):
    objects = {"balances": balances, "streams": streams, "meta": meta}
    for name in VARIANTS:
        objects[f"world_{name}"] = worlds[name]
        objects[f"head_{name}"] = heads[name]
        objects[f"world_optimizer_{name}"] = world_optimizers[name]
        objects[f"head_optimizer_{name}"] = head_optimizers[name]
    return objects


def train(
    episodes,
    config: Config,
    *,
    steps: int,
    checkpoint: Path,
    out: Path,
    contract: dict,
) -> dict:
    torch.manual_seed(config.seed + 1)
    reference_world = _share_initialisation(World(config), config).to(config.device)
    worlds = {name: copy.deepcopy(reference_world) for name in VARIANTS}
    del reference_world

    torch.manual_seed(config.seed + 2)
    reference_head = Heads(config).to(config.device)
    _configure_head(reference_head)
    heads = {name: copy.deepcopy(reference_head) for name in VARIANTS}
    del reference_head

    initial_world = {state_digest(module) for module in worlds.values()}
    initial_head = {state_digest(module) for module in heads.values()}
    assert len(initial_world) == len(initial_head) == 1

    world_optimizers = {name: optimizer([worlds[name]], config) for name in VARIANTS}
    head_optimizers = {name: optimizer([heads[name]], config) for name in VARIANTS}
    balances = {name: {} for name in VARIANTS}
    sampler, _ = _generators(config, 1)
    model_rngs = {
        name: torch.Generator(device=config.device).manual_seed(config.seed + 1001)
        for name in VARIANTS
    }
    streams: dict = {}
    meta: dict = {}
    resume = 0

    if checkpoint.exists():
        load(
            checkpoint,
            config,
            **_objects(
                worlds,
                heads,
                world_optimizers,
                head_optimizers,
                balances,
                streams,
                meta,
            ),
        )
        if meta.get("contract") != contract:
            raise ValueError("consequence-gradient checkpoint contract changed")
        resume = int(meta["step"])
        sampler.set_state(streams["sampler"])
        for name in VARIANTS:
            model_rngs[name].set_state(streams["models"][name])

    gate_sampler = torch.Generator().manual_seed(config.seed + 9100)
    gate_batch = _to(
        sample_terminal_batch(episodes, gate_sampler, config, 0, steps),
        config.device,
    )
    routing = gradient_preflight(
        worlds["world_gradient"], heads["world_gradient"], gate_batch, config
    )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for step in range(resume, steps):
        main = _to(sample_batch(episodes, sampler, config, step, steps), config.device)
        terminal = _to(
            sample_terminal_batch(episodes, sampler, config, step, steps),
            config.device,
        )
        for name in VARIANTS:
            world, head = worlds[name], heads[name]
            dynamics = transition_loss(
                world, main, model_rngs[name], config, step=step
            )
            terminal_loss = terminal_objective(
                world,
                head,
                terminal,
                model_rngs[name],
                config,
                stop_world_gradient=name == "stopped_world_gradient",
            )
            loss = _balance(
                {"dynamics": dynamics, "terminal": terminal_loss},
                balances[name],
                config,
            )
            _update_pair(
                world_optimizers[name],
                head_optimizers[name],
                loss,
                world,
                head,
                config,
                step,
            )

        if (step + 1) % 100 == 0 or step + 1 == steps:
            rms = " ".join(
                f"{name}=({balances[name]['dynamics'] ** 0.5:.4f},"
                f"{balances[name]['terminal'] ** 0.5:.4f})"
                for name in VARIANTS
            )
            print(f"step {step + 1}/{steps} rms(dynamics,terminal) {rms}", flush=True)

        if (step + 1) % config.checkpoint_every == 0 or step + 1 == steps:
            meta = {"contract": contract, "step": step + 1}
            streams = {
                "sampler": sampler.get_state(),
                "models": generator_state(**model_rngs),
            }
            save(
                checkpoint,
                config,
                **_objects(
                    worlds,
                    heads,
                    world_optimizers,
                    head_optimizers,
                    balances,
                    streams,
                    meta,
                ),
            )

    out.mkdir(parents=True, exist_ok=True)
    final = {}
    for name in VARIANTS:
        world_path = out / f"{name}.world.pt"
        model_path = out / f"{name}.model.pt"
        save(world_path, config, part0=worlds[name], experiment=contract)
        save(
            model_path,
            config,
            part0=worlds[name],
            part1=heads[name],
            experiment=contract,
        )
        final[name] = {
            "world": str(world_path.resolve()),
            "model": str(model_path.resolve()),
            "world_sha256": state_digest(worlds[name]),
            "head_sha256": state_digest(heads[name]),
            "rms": {key: value**0.5 for key, value in balances[name].items()},
        }
    if final[VARIANTS[0]]["world_sha256"] == final[VARIANTS[1]]["world_sha256"]:
        raise AssertionError("allowed and stopped world gradients produced identical worlds")
    return {
        "contract": contract,
        "resumed_from": resume,
        "initial_world_sha256": initial_world.pop(),
        "initial_head_sha256": initial_head.pop(),
        "gradient_preflight": routing,
        "final": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--steps", type=int, default=20_000)
    args = parser.parse_args()

    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    encoder, episodes = cached_train(args.phase1a, base, args.expert)
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    contract = {
        "version": "direct-phase1b-consequence-gradient-v1",
        "phase1a": file_digest(args.phase1a),
        "data": data_digests(),
        "implementation": implementation_digests(Path(__file__)),
        "steps": args.steps,
        "ordinary_dynamics": "identical Phase-1B batches and objective",
        "terminal_auxiliary": (
            "balanced generated final-alive/final-dead BCE on TRAIN terminal tails"
        ),
        "variants": {
            "world_gradient": "terminal BCE trains head and world",
            "stopped_world_gradient": "terminal BCE trains head after agent.detach",
        },
        "optimizers": "separate world and head AdamW with separate clipping",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report = train(
        episodes,
        config,
        steps=args.steps,
        checkpoint=args.out / "train.pt",
        out=args.out,
        contract=contract,
    )
    atomic_json(args.out / "training_report.json", report)
    print(f"complete: {args.out / 'training_report.json'}", flush=True)


if __name__ == "__main__":
    main()

