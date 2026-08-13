"""Extend the exact 20k Direct Phase-1B control to an 80k learning curve."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    cached_train,
    data_digests,
    file_digest,
    implementation_digests,
    state_digest,
    stored_state_digest,
)
from d4mj.checkpoint import FORMAT, load, save
from d4mj.config import Config
from d4mj.data import sample_batch
from d4mj.train import (
    _balance,
    _generators,
    _share_initialisation,
    _to,
    _update,
    generator_state,
    optimizer,
)
from d4mj.transition import World, transition_loss


def _reference_digest(path: Path, config: Config, schedule_cycle: int) -> str:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if payload["format"] != FORMAT:
        raise ValueError(f"not a {FORMAT} checkpoint: {path}")
    if payload["config"] != asdict(config):
        raise ValueError("reference Phase-1B config differs")
    if payload["modules"].get("contract") != f"1B:{schedule_cycle}":
        raise ValueError("reference is not the completed schedule-cycle control")
    return stored_state_digest(payload["modules"]["part0"])


def train(
    episodes,
    config: Config,
    *,
    steps: int,
    schedule_cycle: int,
    milestones: tuple[int, ...],
    reference_phase1b: Path,
    checkpoint: Path,
    out: Path,
    contract: dict,
) -> dict:
    if steps < schedule_cycle or schedule_cycle not in milestones:
        raise ValueError("the schedule-cycle endpoint must be a milestone")
    reference_digest = _reference_digest(reference_phase1b, config, schedule_cycle)

    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(config.device)
    optimiser = optimizer([world], config)
    balance: dict[str, float] = {}
    sampler, model_rng = _generators(config, 1)
    streams: dict = {}
    meta: dict = {}
    resume = 0
    curve: list[dict] = []
    milestone_digests: dict[str, str] = {}

    if checkpoint.exists():
        load(
            checkpoint,
            config,
            part0=world,
            part1=optimiser,
            balance=balance,
            streams=streams,
            meta=meta,
        )
        if meta.get("contract") != contract:
            raise ValueError("long-control checkpoint contract changed")
        resume = int(meta["step"])
        curve = list(meta.get("curve", []))
        milestone_digests = dict(meta.get("milestone_digests", {}))
        sampler.set_state(streams["sampler"])
        model_rng.set_state(streams["model"])

    out.mkdir(parents=True, exist_ok=True)
    window_losses: list[float] = []
    for step in range(resume, steps):
        schedule_step = step % schedule_cycle
        batch = _to(
            sample_batch(
                episodes,
                sampler,
                config,
                schedule_step,
                schedule_cycle,
            ),
            config.device,
        )
        dynamics = transition_loss(
            world, batch, model_rng, config, step=schedule_step
        )
        window_losses.append(float(dynamics.detach()))
        _update(
            optimiser,
            _balance({"dynamics": dynamics}, balance, config),
            [world],
            config,
            step,
        )
        completed = step + 1

        if completed % 100 == 0:
            curve.append(
                {
                    "step": completed,
                    "raw_dynamics_mean": sum(window_losses) / len(window_losses),
                    "dynamics_rms": balance["dynamics"] ** 0.5,
                }
            )
            window_losses.clear()
            print(
                f"step {completed}/{steps} raw={curve[-1]['raw_dynamics_mean']:.6f} "
                f"rms={curve[-1]['dynamics_rms']:.6f}",
                flush=True,
            )

        if completed in milestones:
            digest = state_digest(world)
            milestone_digests[str(completed)] = digest
            if completed == schedule_cycle and digest != reference_digest:
                mismatch = {
                    "step": completed,
                    "expected_reference_sha256": reference_digest,
                    "actual_world_sha256": digest,
                    "matched": False,
                }
                atomic_json(out / "reference_mismatch.json", mismatch)
                raise AssertionError(
                    "the 20k point did not reproduce the ordinary Phase-1B control"
                )
            save(
                out / f"world_{completed:06d}.pt",
                config,
                part0=world,
                experiment=contract,
                step=completed,
            )

        if completed % config.checkpoint_every == 0 or completed == steps:
            streams = generator_state(sampler=sampler, model=model_rng)
            meta = {
                "contract": contract,
                "step": completed,
                "curve": curve,
                "milestone_digests": milestone_digests,
            }
            save(
                checkpoint,
                config,
                part0=world,
                part1=optimiser,
                balance=balance,
                streams=streams,
                meta=meta,
            )

    return {
        "contract": contract,
        "resumed_from": resume,
        "reference_20k_world_sha256": reference_digest,
        "milestone_world_sha256": milestone_digests,
        "reference_20k_reproduced_exactly": (
            milestone_digests.get(str(schedule_cycle)) == reference_digest
        ),
        "training_curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--reference-phase1b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--schedule-cycle", type=int, default=20_000)
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=(5_000, 10_000, 20_000, 40_000, 60_000, 80_000),
    )
    args = parser.parse_args()

    milestones = tuple(sorted(set(args.milestones)))
    if milestones[-1] != args.steps:
        parser.error("the final step must be included in --milestones")

    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    encoder, episodes = cached_train(args.phase1a, base, args.expert)
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    contract = {
        "version": "long-direct-phase1b-v1",
        "phase1a": file_digest(args.phase1a),
        "reference_phase1b": file_digest(args.reference_phase1b),
        "data": data_digests(),
        "implementation": implementation_digests(Path(__file__)),
        "steps": args.steps,
        "milestones": milestones,
        "schedule": (
            f"repeat the exact {args.schedule_cycle}-step production short/long "
            "schedule; the first cycle must reproduce the saved control bitwise"
        ),
    }
    report = train(
        episodes,
        config,
        steps=args.steps,
        schedule_cycle=args.schedule_cycle,
        milestones=milestones,
        reference_phase1b=args.reference_phase1b,
        checkpoint=args.out / "train.pt",
        out=args.out,
        contract=contract,
    )
    atomic_json(args.out / "training_report.json", report)
    print(f"complete: {args.out / 'training_report.json'}", flush=True)


if __name__ == "__main__":
    main()
