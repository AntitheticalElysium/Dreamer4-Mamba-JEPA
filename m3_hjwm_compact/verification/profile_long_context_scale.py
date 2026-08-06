"""Pre-outcome resource and numerical gate for Step 4b.

This script consumes only the pinned training replay.  It does not open the
fresh long-prefix monitor bundle and therefore cannot influence arm selection.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from data import Episode, EpisodeReplay  # noqa: E402
from long_context_scale import (  # noqa: E402
    LONG_CONTEXT,
    build_long_world,
    temporal_parameter_count,
)
from model import frozen_dynamics_recipe  # noqa: E402
from step3_temporal import load_scaled_data  # noqa: E402
from step4_runner import shared_state_digest  # noqa: E402
from consolidation import build_world  # noqa: E402


OUT = REPO_ROOT / "reviews" / "artifacts" / "long_context_feasibility.json"
ARMS = ("LS-G64", "LS-M64", "LL-G", "LL-M")


def _cache_mib(state) -> float:
    entries = []
    for item in state.cache or []:
        entries.extend(item if isinstance(item, (tuple, list)) else (item,))
    return sum(value.numel() * value.element_size() for value in entries) / 2**20


def _synchronize():
    torch.cuda.synchronize()


def _timed(repetitions, function):
    for _ in range(2):
        function()
    _synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        function()
    _synchronize()
    return 1e3 * (time.perf_counter() - started) / repetitions


def profile_arm(arm, reference_shared, batch, device):
    torch.cuda.empty_cache()
    torch.manual_seed(404)
    world = build_long_world(arm, 404, reference_shared, device).train()
    trainable = [parameter for parameter in world.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    initial_shared_digest = shared_state_digest(world)

    def train_once():
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = world(batch, frozen_dynamics_recipe())
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 100.0)
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        optimizer.step()
        world.mark_parameters_updated()
        _synchronize()
        return {
            "loss": float(output.loss.detach()),
            "grad_norm_before_clip": float(grad_norm),
            "finite_loss": bool(torch.isfinite(output.loss)),
            "finite_gradients": finite_gradients,
            "milliseconds": 1e3 * (time.perf_counter() - started),
        }

    cold_step = train_once()
    torch.cuda.reset_peak_memory_stats()
    steady_step = train_once()
    training_memory = {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }

    world.eval()
    core = world.temporal.impl
    streams, dim = world.streams, world.cfg.token_dim
    sequence_input = torch.randn(1, LONG_CONTEXT, streams, dim, device=device)
    with torch.no_grad():
        def sequence_call():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                core.sequence(sequence_input)

        sequence_ms = _timed(5, sequence_call)

        state = core.init_state(1, streams, device, torch.float32)
        cache_mib_b1 = _cache_mib(state)
        step_input = sequence_input[:, 0]

        def step_call():
            nonlocal state
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, state = core.step(step_input, state)

        recurrent_step_ms = _timed(20, step_call)

        obs = batch["obs"][:, 0]
        previous = batch["previous_actions"][:, 0]
        state_world = world.initial_state(1, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state_world = world.observe_step(obs, previous, state_world)
        actions = batch["actions"][:, :8]

        def imagination_call():
            state_local = state_world
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for index in range(8):
                    state_local, _, _, _ = world.imagine_step(
                        state_local, actions[:, index], deterministic_mode=True)

        imagination_h8_ms = _timed(5, imagination_call)

    result = {
        "arm": arm,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "temporal_parameters": temporal_parameter_count(core),
        "initial_shared_state_digest": initial_shared_digest,
        "cold_train_step": cold_step,
        "steady_train_step": steady_step,
        **training_memory,
        "temporal_sequence_ms_B1_T128": sequence_ms,
        "temporal_recurrent_step_ms_B1": recurrent_step_ms,
        "temporal_cache_mib_B1_fp32": cache_mib_b1,
        "world_imagination_ms_B1_H8": imagination_h8_ms,
    }
    del optimizer, world
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Step 4b requires the actual CUDA target")
    device = torch.device("cuda")
    train, _ = load_scaled_data()
    replay = EpisodeReplay(capacity_steps=500_000)
    for episode in train:
        replay.add(Episode(**episode))
    batch = replay.sample(
        batch=1, observations=LONG_CONTEXT, device=device,
        rng=np.random.default_rng(404))

    torch.manual_seed(404)
    reference = build_world("global_gru", 64, device)
    reference_shared = {
        name: value.detach().cpu().clone()
        for name, value in reference.state_dict().items()
        if not name.startswith("temporal.")
    }
    del reference
    torch.cuda.empty_cache()

    report = {
        "protocol": "reviews/2026-07-16-long-context-scale-protocol.md",
        "outcome_blind": True,
        "batch": 1,
        "observations": LONG_CONTEXT,
        "dtype": "BF16 autocast, FP32 parameters/cache",
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "arms": {},
    }
    for arm in ARMS:
        report["arms"][arm] = profile_arm(
            arm, reference_shared, batch, device)
        print(json.dumps(report["arms"][arm], indent=2), flush=True)
    shared = {
        entry["initial_shared_state_digest"]
        for entry in report["arms"].values()
    }
    report["shared_state_pairing_pass"] = len(shared) == 1
    report["all_numerical_gates_pass"] = all(
        entry["cold_train_step"]["finite_loss"]
        and entry["cold_train_step"]["finite_gradients"]
        and entry["steady_train_step"]["finite_loss"]
        and entry["steady_train_step"]["finite_gradients"]
        for entry in report["arms"].values())
    report["all_vram_gates_pass"] = all(
        entry["peak_reserved_mib"] < 5000
        for entry in report["arms"].values())
    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
