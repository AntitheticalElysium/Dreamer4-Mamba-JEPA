"""Shared construction and training for the Stage-2F operator control."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from checkpoint import sprint_candidate_config  # noqa: E402
from fork_oracle_v2 import ENCODER_CKPT  # noqa: E402
from model import (  # noqa: E402
    M3HJWM,
    ModelConfig,
    assert_encoder_frozen,
    enforce_frozen_encoder,
    frozen_dynamics_recipe,
)
from ssl_ijepa import IJEPAPretrainer  # noqa: E402
from stage1b_equal_update_control import state_digest  # noqa: E402
from stage2_ab import (  # noqa: E402
    BATCH,
    PREFIX,
    SEED,
    make_batch,
)
from stage2_objectives import (  # noqa: E402
    GeneratedLossWeights,
    generated_step_components,
    weighted_generated_loss,
)


PROTOCOL = "reviews/2026-07-19-stage2f-reward-operator-protocol.md"
FULL_UPDATES = 16_000
SMOKE_UPDATES = 64
GENERATED_STEPS = 2
GENERATED_WEIGHTS = GeneratedLossWeights(
    latent=1.0, reward=0.10, continuation=0.0
)
LOCAL_FINGERPRINT = {
    "schedule_sha256": (
        "427eb8a311ac9a99ec7f5fd529added9035777a1146864c4ab53d68c2c1295d0"
    ),
    "init_digest": (
        "55e31261de2ced792bab1754d9060cefefb682d4964324fbca5643da8d2c7260"
    ),
    "final_digest": (
        "92048e7311cda13ff178ed921b129ad5c85f95c56fba9c3046bc8c8d00b17415"
    ),
    "history_sha256": (
        "ce25a57adab01d26bcf516ba929bfc6f81617b9748cf03c786d810d7d23a1a3b"
    ),
}
HISTORY_FIELDS = (
    "total",
    "base_jepa",
    "base_reward",
    "base_continuation",
    "base_rollout",
    "generated_latent",
    "generated_reward",
    "generated_weighted",
)


def build_operator_world(
    device: torch.device,
    operator: str,
    *,
    zero_output: bool,
) -> M3HJWM:
    """Exact Stage-2C initialization with an explicit reward operator."""
    torch.manual_seed(SEED)
    cfg = replace(
        sprint_candidate_config("gru"),
        reward_operator=operator,
    )
    world = M3HJWM(cfg).to(device)
    pretrainer = IJEPAPretrainer(ModelConfig(
        temporal_backend="gru",
        predictor="deterministic",
        mask_ratio=0.0,
    ))
    pretrainer.load_state_dict(
        torch.load(ENCODER_CKPT, weights_only=False)["pretrainer"],
        strict=True,
    )
    encoder = pretrainer.target_encoder.model.state_dict()
    world.online_encoder.load_state_dict(encoder)
    world.target_encoder.model.load_state_dict(encoder)
    enforce_frozen_encoder(world)
    if zero_output:
        zero_reward_output(world)
    return world


def zero_reward_output(world: M3HJWM) -> None:
    output = world.reward.net[-1]
    if not isinstance(output, nn.Linear):
        raise RuntimeError("reward output is no longer the registered Linear")
    if output.out_features != world.cfg.reward_bins:
        raise RuntimeError("reward output width drift")
    with torch.no_grad():
        output.weight.zero_()
        if output.bias is None:
            raise RuntimeError("registered reward output requires a bias")
        output.bias.zero_()


def assert_same_initial_state(first: M3HJWM, second: M3HJWM) -> None:
    first_state = first.state_dict()
    second_state = second.state_dict()
    if first_state.keys() != second_state.keys():
        raise RuntimeError("operator arms have different state keys")
    for name, value in first_state.items():
        if not torch.equal(value, second_state[name]):
            raise RuntimeError(
                f"operator arms differ at initialization: {name}"
            )
    first_names = [name for name, _ in first.named_parameters()]
    second_names = [name for name, _ in second.named_parameters()]
    if first_names != second_names:
        raise RuntimeError("operator arms have different parameter names")


def history_array(histories: dict[str, list[float]]) -> np.ndarray:
    return np.stack(
        [
            np.asarray(histories[name], dtype=np.float64)
            for name in HISTORY_FIELDS
        ],
        axis=1,
    )


def history_sha256(histories: dict[str, list[float]]) -> str:
    return hashlib.sha256(history_array(histories).tobytes()).hexdigest()


@torch.no_grad()
def generated_decode_probe(
    world: M3HJWM,
    batch: dict[str, torch.Tensor],
) -> dict:
    device = batch["obs"].device
    state = world.initial_state(batch["obs"].shape[0], device)
    decoded = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for time_index in range(PREFIX):
            state = world.observe_step(
                batch["obs"][:, time_index],
                batch["previous_actions"][:, time_index],
                state,
            )
        for generated_index in range(GENERATED_STEPS):
            transition = PREFIX - 1 + generated_index
            state, reward_logits, _, _ = world.imagine_step(
                state,
                batch["actions"][:, transition],
                deterministic_mode=True,
            )
            decoded.append(world.reward.decode(reward_logits).float())
    values = torch.cat(decoded)
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "absolute_maximum": float(values.abs().max()),
        "absolute_mean": float(values.abs().mean()),
        "finite": bool(torch.isfinite(values).all()),
    }


def train_world(
    world: M3HJWM,
    train: list[dict],
    schedule: list[tuple[int, int]],
    *,
    updates: int,
    progress_name: str,
    probe_updates: tuple[int, ...] = (),
) -> tuple[M3HJWM, dict]:
    """Run the exact Stage-2C C-LR update with optional smoke probes."""
    if len(schedule) < updates * BATCH:
        raise ValueError("schedule is shorter than requested training")
    weights = frozen_dynamics_recipe()
    trainable_names = [
        name
        for name, parameter in world.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [
        parameter
        for parameter in world.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    histories = {name: [] for name in HISTORY_FIELDS}
    probes = {}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    first_batch = make_batch(train, schedule[:BATCH],
                             next(world.parameters()).device)
    if 0 in probe_updates:
        probes["u0"] = generated_decode_probe(world, first_batch)
    del first_batch
    started = time.perf_counter()

    for update in range(updates):
        picks = schedule[update * BATCH:(update + 1) * BATCH]
        batch = make_batch(
            train, picks, next(world.parameters()).device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base = world(batch, weights)
            components = generated_step_components(
                world,
                batch,
                prefix=PREFIX,
                steps=GENERATED_STEPS,
            )
            generated = weighted_generated_loss(
                components, GENERATED_WEIGHTS
            )
            loss = base.loss + generated
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"{progress_name} non-finite loss at update {update + 1}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(
                f"{progress_name} non-finite gradient at update {update + 1}"
            )
        optimizer.step()
        world.mark_parameters_updated()

        row = {
            "total": float(loss.detach()),
            "base_jepa": float(base.metrics["jepa"]),
            "base_reward": float(base.metrics["reward"]),
            "base_continuation": float(
                base.metrics["continuation"]
            ),
            "base_rollout": float(base.metrics["rollout"]),
            "generated_latent": float(
                components["latent"].detach()
            ),
            "generated_reward": float(
                components["reward"].detach()
            ),
            "generated_weighted": float(generated.detach()),
        }
        for name in HISTORY_FIELDS:
            histories[name].append(row[name])

        completed = update + 1
        if completed in probe_updates:
            probes[f"u{completed}"] = generated_decode_probe(
                world, batch
            )
        if updates <= SMOKE_UPDATES or completed % 1000 == 0:
            for name, parameter in world.named_parameters():
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError(
                        f"{progress_name} non-finite parameter {name} "
                        f"at update {completed}"
                    )
        if completed % 4000 == 0:
            print(
                f"[{progress_name}] {completed}: "
                f"jepa={np.mean(histories['base_jepa'][-500:]):.5f} "
                f"base-reward="
                f"{np.mean(histories['base_reward'][-500:]):.5f} "
                f"gen-reward="
                f"{np.mean(histories['generated_reward'][-500:]):.5f}",
                flush=True,
            )

    assert_encoder_frozen(world, optimizer)
    final_digest = state_digest(world, exclude_heads=False)
    info = {
        "updates": updates,
        "train_seconds": time.perf_counter() - started,
        "final_digest": final_digest,
        "history_sha256": history_sha256(histories),
        "histories": histories,
        "probes": probes,
        "trainable_names": trainable_names,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / 2**20
        ),
        "last500": {
            name: float(np.mean(values[-500:]))
            for name, values in histories.items()
        },
    }
    return world, info


def validate_probe_limits(info: dict) -> None:
    for name, probe in info["probes"].items():
        if not probe["finite"]:
            raise RuntimeError(f"non-finite decoded probe at {name}")
        if probe["absolute_maximum"] > 100.0:
            raise RuntimeError(
                f"decoded smoke reward exceeds 100 at {name}: "
                f"{probe['absolute_maximum']}"
            )
    if info["peak_reserved_mib"] >= 5500:
        raise RuntimeError(
            f"smoke reserved VRAM {info['peak_reserved_mib']} MiB"
        )
