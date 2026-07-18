"""Math and logit collection for frozen categorical reward calibration."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib

import numpy as np
import torch

from model import WorldState, decode_two_hot, two_hot
from phase_e_same_target import (
    BATCH,
    HISTORY,
    HORIZONS,
    WINDOW_OBS,
    suffix_partition,
)
from phase_e_taskheads import clone_world_state


ARM_ORDER = ("E-I", "E-T", "E-Z", "E-TZ")
ZERO_BIN = 127


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@dataclass(frozen=True)
class CalibrationSpec:
    arm: str
    log_temperature: float = 0.0
    zero_bias: float = 0.0

    def __post_init__(self):
        if self.arm not in ARM_ORDER:
            raise ValueError(f"unknown calibration arm {self.arm}")
        if not np.isfinite(self.log_temperature):
            raise ValueError("non-finite log temperature")
        if not np.isfinite(self.zero_bias):
            raise ValueError("non-finite zero-bin bias")

    @property
    def temperature(self) -> float:
        return float(np.exp(self.log_temperature))

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "temperature": self.temperature,
        }


def calibrate_logits(
    logits: torch.Tensor,
    spec: CalibrationSpec,
) -> torch.Tensor:
    if logits.shape[-1] != 255:
        raise ValueError("Stage-2E expects the registered 255 reward bins")
    temperature = logits.new_tensor(spec.temperature)
    if not bool(torch.isfinite(temperature)) or not bool(temperature > 0):
        raise ValueError("calibration temperature must be finite and positive")
    output = logits / temperature
    if spec.zero_bias:
        bias = torch.zeros(
            logits.shape[-1], dtype=logits.dtype, device=logits.device
        )
        bias[ZERO_BIN] = spec.zero_bias
        output = output + bias
    return output


def decode_calibrated(
    logits: torch.Tensor,
    spec: CalibrationSpec,
    *,
    low: float,
    high: float,
) -> torch.Tensor:
    return decode_two_hot(
        calibrate_logits(logits, spec), low, high
    )


def calibration_nll(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    spec: CalibrationSpec,
    *,
    low: float,
    high: float,
) -> torch.Tensor:
    targets = two_hot(rewards, logits.shape[-1], low, high)
    log_probabilities = calibrate_logits(
        logits, spec
    ).log_softmax(-1)
    return -(targets * log_probabilities).sum(-1)


def _parameterized_logits(
    logits: torch.Tensor,
    log_temperature: torch.Tensor,
    zero_bias: torch.Tensor,
) -> torch.Tensor:
    temperature = log_temperature.exp()
    bias = torch.zeros(
        logits.shape[-1], dtype=logits.dtype, device=logits.device
    )
    bias[ZERO_BIN] = zero_bias
    return logits / temperature + bias


def fit_calibrator(
    arm: str,
    logits: torch.Tensor,
    rewards: torch.Tensor,
    *,
    low: float,
    high: float,
    max_iter: int = 200,
) -> tuple[CalibrationSpec, dict]:
    """Fit one registered arm with deterministic full-batch LBFGS."""
    if arm not in ARM_ORDER:
        raise ValueError(arm)
    logits = logits.detach().cpu().double().contiguous()
    rewards = rewards.detach().cpu().double().contiguous()
    if logits.ndim != 2 or logits.shape[-1] != 255:
        raise ValueError("logits must be [N,255]")
    if rewards.shape != logits.shape[:1]:
        raise ValueError("reward/logit row mismatch")
    targets = two_hot(rewards, logits.shape[-1], low, high)
    identity = CalibrationSpec("E-I")
    initial_nll = float(calibration_nll(
        logits, rewards, identity, low=low, high=high
    ).mean())
    if arm == "E-I":
        return identity, {
            "initial_nll": initial_nll,
            "final_nll": initial_nll,
            "closure_calls": 0,
        }

    fit_temperature = arm in ("E-T", "E-TZ")
    fit_zero_bias = arm in ("E-Z", "E-TZ")
    log_temperature = torch.zeros(
        (), dtype=torch.float64, requires_grad=fit_temperature
    )
    zero_bias = torch.zeros(
        (), dtype=torch.float64, requires_grad=fit_zero_bias
    )
    parameters = [
        parameter
        for parameter in (log_temperature, zero_bias)
        if parameter.requires_grad
    ]
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure():
        nonlocal closure_calls
        optimizer.zero_grad(set_to_none=True)
        calibrated = _parameterized_logits(
            logits, log_temperature, zero_bias
        )
        loss = -(
            targets * calibrated.log_softmax(-1)
        ).sum(-1).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite calibration objective")
        loss.backward()
        closure_calls += 1
        return loss

    optimizer.step(closure)
    spec = CalibrationSpec(
        arm=arm,
        log_temperature=(
            float(log_temperature.detach()) if fit_temperature else 0.0
        ),
        zero_bias=(
            float(zero_bias.detach()) if fit_zero_bias else 0.0
        ),
    )
    final_nll = float(calibration_nll(
        logits, rewards, spec, low=low, high=high
    ).mean())
    if not np.isfinite(final_nll):
        raise RuntimeError("non-finite fitted calibration NLL")
    if final_nll > initial_nll + 1e-10:
        raise RuntimeError(
            f"{arm} calibration worsened NLL "
            f"{initial_nll} -> {final_nll}"
        )
    return spec, {
        "initial_nll": initial_nll,
        "final_nll": final_nll,
        "closure_calls": closure_calls,
    }


def select_calibrator(fits: dict[str, dict]) -> str:
    """Lowest finite CAL NLL; ties favor registered lower capacity."""
    missing = set(ARM_ORDER) - set(fits)
    if missing:
        raise ValueError(f"missing calibration arms: {sorted(missing)}")
    best = None
    best_nll = float("inf")
    for arm in ARM_ORDER:
        value = float(fits[arm]["optimization"]["final_nll"])
        if not np.isfinite(value):
            continue
        if value < best_nll - 1e-10:
            best = arm
            best_nll = value
    if best is None:
        raise RuntimeError("no finite calibration arm")
    return best


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def collect_same_target_logits(
    world,
    arrays: dict,
    device: torch.device,
    *,
    batch_size: int = BATCH,
) -> dict[str, torch.Tensor]:
    """Frozen reward logits for the registered K0/1/2/4/8 construction."""
    output = {f"k{depth}": [] for depth in HORIZONS}
    total = len(arrays["obs"])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        obs = torch.from_numpy(arrays["obs"][start:stop]).to(device)
        actions = torch.from_numpy(
            arrays["actions"][start:stop]
        ).to(device)
        previous = torch.from_numpy(
            arrays["previous_actions"][start:stop]
        ).to(device)
        batch = stop - start

        encoded = []
        with _autocast(device):
            for time_index in range(WINDOW_OBS):
                encoded.append(
                    world.online_encoder(obs[:, time_index])
                )
            state = world.initial_state(batch, device)
            for time_index in range(HISTORY):
                index = world._previous_action_indices(
                    previous[:, time_index]
                )
                value = (
                    encoded[time_index]
                    + world.action_input(index)[:, None]
                )
                temporal_output, temporal = world.temporal.step(
                    value, state.temporal
                )
                state = WorldState(
                    temporal, temporal_output, state.revision
                )
        base = state

        for depth in HORIZONS:
            state = clone_world_state(base)
            real_times, imagined_actions = suffix_partition(depth)
            logits = None
            with _autocast(device):
                for time_index in real_times:
                    index = world._previous_action_indices(
                        previous[:, time_index]
                    )
                    value = (
                        encoded[time_index]
                        + world.action_input(index)[:, None]
                    )
                    temporal_output, temporal = world.temporal.step(
                        value, state.temporal
                    )
                    state = WorldState(
                        temporal, temporal_output, state.revision
                    )
                for action_index in imagined_actions:
                    state, logits, _, _ = world.imagine_step(
                        state,
                        actions[:, action_index],
                        deterministic_mode=True,
                    )
                if depth == 0:
                    logits = world.reward(world.pool(state.tokens))
            if logits is None:
                raise RuntimeError(f"no logits collected at depth {depth}")
            output[f"k{depth}"].append(logits.float().cpu())

    return {
        key: torch.cat(values, dim=0)
        for key, values in output.items()
    }
