"""Matched Direct Phase-1B curve with only the target-space metric changed."""

from __future__ import annotations

import argparse
import json
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
from artifacts.phase1b_geometry_common import direct_metric_loss
from d4mj.checkpoint import load, save
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
from d4mj.transition import World


def train(
    episodes,
    precision,
    config: Config,
    *,
    steps: int,
    schedule_cycle: int,
    milestones: tuple[int, ...],
    checkpoint: Path,
    out: Path,
    contract: dict,
) -> dict:
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(config.device)
    initial_digest = state_digest(world)
    optimiser = optimizer([world], config)
    balance: dict[str, float] = {}
    sampler, model_rng = _generators(config, 1)
    streams, meta = {}, {}
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
            raise ValueError("whitened Phase-1B checkpoint contract changed")
        resume = int(meta["step"])
        curve = list(meta.get("curve", []))
        milestone_digests = dict(meta.get("milestone_digests", {}))
        if meta.get("initial_world_sha256") != initial_digest:
            raise ValueError("whitened run initialization changed")
        sampler.set_state(streams["sampler"])
        model_rng.set_state(streams["model"])

    window_losses = []
    for step in range(resume, steps):
        schedule_step = step % schedule_cycle
        batch = _to(
            sample_batch(
                episodes, sampler, config, schedule_step, schedule_cycle
            ),
            config.device,
        )
        dynamics = direct_metric_loss(
            world, batch, model_rng, config, precision
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
                    "raw_whitened_dynamics_mean": sum(window_losses) / len(window_losses),
                    "dynamics_rms": balance["dynamics"] ** 0.5,
                }
            )
            window_losses.clear()
            print(
                f"whitened {completed}/{steps} "
                f"raw={curve[-1]['raw_whitened_dynamics_mean']:.6f}",
                flush=True,
            )
        if completed in milestones:
            digest = state_digest(world)
            milestone_digests[str(completed)] = digest
            save(
                out / f"world_{completed:06d}.pt",
                config,
                part0=world,
                experiment=contract,
                step=completed,
            )
        if completed % config.checkpoint_every == 0 or completed == steps:
            save(
                checkpoint,
                config,
                part0=world,
                part1=optimiser,
                balance=balance,
                streams=generator_state(sampler=sampler, model=model_rng),
                meta={
                    "contract": contract,
                    "step": completed,
                    "initial_world_sha256": initial_digest,
                    "curve": curve,
                    "milestone_digests": milestone_digests,
                },
            )
    return {
        "contract": contract,
        "resumed_from": resume,
        "initial_world_sha256": initial_digest,
        "milestone_world_sha256": milestone_digests,
        "training_curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--ordinary-control-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--schedule-cycle", type=int, default=20000)
    parser.add_argument(
        "--milestones", type=int, nargs="+", default=(5000, 20000, 80000)
    )
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if not gate.get("eligible", False):
        report = {
            "status": "skipped_by_preregistered_gate",
            "gate": gate,
            "gate_sha256": file_digest(args.gate),
        }
        atomic_json(args.out / "training_report.json", report)
        print("whitened Phase-1B skipped: preregistered geometry gate failed", flush=True)
        return

    milestones = tuple(sorted(set(args.milestones)))
    if milestones[-1] != args.steps:
        parser.error("final step must be a milestone")
    prepared = torch.load(args.prepared, weights_only=False)
    precision = prepared["precision"]
    if not torch.allclose(precision, precision.T, atol=1e-5, rtol=0.0):
        raise ValueError("precision is not symmetric")
    if float(torch.linalg.eigvalsh(precision.double()).min()) <= 0:
        raise ValueError("precision is not positive definite")
    ordinary = json.loads(args.ordinary_control_report.read_text())
    if not ordinary.get("reference_20k_reproduced_exactly", False):
        raise ValueError("ordinary control did not reproduce the production 20k world")

    config = Config(transition="direct", time_mixer="attention")
    encoder, episodes = cached_train(args.phase1a, Config(), args.expert)
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    contract = {
        "version": "whitened-direct-phase1b-v1",
        "phase1a": file_digest(args.phase1a),
        "prepared": file_digest(args.prepared),
        "gate": file_digest(args.gate),
        "ordinary_control_report": file_digest(args.ordinary_control_report),
        "data": data_digests(),
        "implementation": implementation_digests(
            Path(__file__), Path("artifacts/phase1b_geometry_common.py")
        ),
        "steps": args.steps,
        "milestones": milestones,
        "schedule_cycle": args.schedule_cycle,
        "only_moved_axis": (
            "replace each Direct teacher/rollout Euclidean squared error with the "
            "fixed TRAIN shrinkage-precision quadratic; inputs, targets, batches, "
            "RNG streams, optimizer, RMS balance and rollout terms are unchanged"
        ),
    }
    report = train(
        episodes,
        precision,
        config,
        steps=args.steps,
        schedule_cycle=args.schedule_cycle,
        milestones=milestones,
        checkpoint=args.out / "train.pt",
        out=args.out,
        contract=contract,
    )
    atomic_json(args.out / "training_report.json", report)
    print(f"complete: {args.out / 'training_report.json'}", flush=True)


if __name__ == "__main__":
    main()
