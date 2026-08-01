import time

import torch
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

from .agent import Heads, head_targets
from .config import Config
from .data import Batch
from .state import WorldState
from .transition import World, advance, initial, transition_loss


@torch.no_grad()
def multistep_error(
    world: World, batch: Batch, rng: torch.Generator, config: Config, successors: Tensor | None = None
) -> dict[str, list[float]]:
    """Per-step error under the real runtime path, from a *committed* prefix.

    Mean error alone cannot adjudicate the direct arm: under squared loss the
    optimal deterministic predictor is the conditional mean, so the collapsed
    solution is the one that minimises exactly this number. When `successors`
    (B, M, ...) samples of the true next latent are supplied, the nearest-mode and
    mean distances are reported alongside -- a predictor sitting between modes
    shows a large gap, and one on a mode shows none.
    """
    state, _ = initial(world, batch.latents[:, :1], batch.led_to_action[:, :1], rng, config)
    report: dict[str, list[float]] = {"mean_error": []}
    for step in range(1, batch.latents.shape[1]):
        state, _ = advance(world, state, batch.led_to_action[:, step : step + 1], rng, config)
        report["mean_error"].append(float((state.latent - batch.latents[:, step : step + 1]).pow(2).mean()))

    if successors is not None:
        gap = (state.latent[:, 0, None] - successors).pow(2).flatten(2).mean(-1)
        report["nearest_mode"] = [float(gap.min(dim=1).values.mean())]
        report["mode_mean"] = [float((state.latent[:, 0] - successors.mean(1)).pow(2).mean())]
    return report


@torch.no_grad()
def latent_stats(world: World, batch: Batch, rng: torch.Generator, config: Config) -> dict[str, float]:
    """Range *and* scale. A bounded readout fixes the range; it does nothing about
    contraction toward the conditional mean, which is the failure that looks like a
    working model in every one-step metric.

    The comparison is against the matched next-state target, not the whole batch:
    sequence-level diversity in the denominator would make a sound one-step
    prediction look contracted."""
    real = batch.latents[:, 1:2]
    state, _ = initial(world, batch.latents[:, :1], batch.led_to_action[:, :1], rng, config)
    state, _ = advance(world, state, batch.led_to_action[:, 1:2], rng, config)
    predicted = state.latent
    return {
        "real_std": float(real.std()),
        "predicted_std": float(predicted.std()),
        "contraction": float(predicted.std() / real.std().clamp(min=1e-8)),
        "outside_unit": float((predicted.abs() > 1.0).float().mean()),
    }


@torch.no_grad()
def head_calibration(heads: Heads, agent: Tensor, batch: Batch, config: Config) -> dict[str, float]:
    """Reward and continuation against their targets at lead 0, which is the only
    lead deployment reads."""
    readout, targets = heads(agent), head_targets(batch, config)
    valid = targets["valid"][..., 0]
    probability = readout["continuation"][..., 0].sigmoid()
    mean = (readout["reward"][..., 0, :].softmax(-1) * heads.centers).sum(-1)
    predicted = mean.sign() * torch.expm1(mean.abs())
    return {
        "reward_mae": float(((predicted - targets["reward"][..., 0]).abs() * valid).sum() / valid.sum()),
        "continuation_mean": float((probability * valid).sum() / valid.sum()),
        "continuation_target": float((targets["continuation"][..., 0] * valid).sum() / valid.sum()),
    }


def cost(modules: dict[str, nn.Module], world: World, config: Config) -> dict[str, float]:
    """Deployed against training-only parameters, and the two state sizes apart.

    `effective_horizon` is how far back a perturbation still moves the prediction.
    Both trajectories are re-rolled from scratch at each distance under one seed, so
    a difference is history rather than accumulated divergence or unmatched noise.
    The ladder doubles, so the figure is a power-of-two *lower bound*, capped at
    twice the context -- named `effective_horizon_at_least` because a truly
    long-memory arm reports the cap and must not be read as converging there.
    Attention is hard-bounded at `dynamics_context`; an SSM state has no cutoff, so
    reporting it is what keeps a Mamba win from silently meaning "remembers more".

    Mamba fixes the dynamics memory only; the encoder keeps its own bounded cache,
    so a single 'state size' would overstate what the substitution buys. Timing runs
    on the configured device after warm-up, synchronised, since an unsynchronised
    CUDA timer measures queueing rather than compute.

    Both throughputs are reported, because they answer different questions and the
    substitution can move them in opposite directions: `steps_per_second` is one
    imagined step at batch 1, which is what the actor pays, and
    `train_steps_per_second` is a full forward and backward over a real sequence,
    which is what the schedule pays. A flow arm spending `rungs + 1` backbone passes
    per imagined step looks far worse on the first than on the second. Phase 3
    freezes the world, so the training measurement re-enables gradients for its own
    duration and restores the flags; otherwise a world measured after Phase 3 would
    report no training cost at all rather than failing.

    `flops_per_step` is measured, not derived from a parameter count, but it counts
    dispatched aten operations: a fused kernel that never dispatches them is
    invisible to it. That is why it is reported *beside* wall-clock time rather than
    instead of it -- for the Mamba arm the two must be read together, and a FLOP
    figure alone would silently favour whichever arm fuses more.
    """
    device, deployed = config.device, {"encoder", "world", "heads"}
    counts = {name: sum(p.numel() for p in m.parameters()) for name, m in modules.items()}
    latent = torch.randn(1, 1, config.n_spatial, config.d_spatial, device=device)
    action = torch.zeros(1, 1, dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(0)

    with torch.no_grad():
        state, _ = initial(world, latent, action, rng, config)
        for _ in range(4):
            state, _ = advance(world, state, action, rng, config)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(16):
            state, _ = advance(world, state, action, rng, config)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    peak = float(torch.cuda.max_memory_allocated()) if device == "cuda" else 0.0

    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        advance(world, state, action, rng, config)
    flops = counter.get_total_flops()

    probe = Batch(
        led_to_action=torch.zeros(config.batch, config.sequence, dtype=torch.long, device=device),
        reward=torch.zeros(config.batch, config.sequence, device=device),
        terminated=torch.zeros(config.batch, config.sequence, dtype=torch.bool, device=device),
        truncated=torch.zeros(config.batch, config.sequence, dtype=torch.bool, device=device),
        valid=torch.ones(config.batch, config.sequence, dtype=torch.bool, device=device),
        relevant=torch.zeros(config.batch, dtype=torch.bool, device=device),
        burn_in=0,
        latents=torch.randn(
            config.batch, config.sequence, config.n_spatial, config.d_spatial, device=device
        ).tanh(),
    )
    frozen = [p for p in world.parameters() if not p.requires_grad]
    for parameter in frozen:
        parameter.requires_grad_(True)
    for repeat in range(6):
        if repeat == 2:
            if device == "cuda":
                torch.cuda.synchronize()
            train_start = time.perf_counter()
        world.zero_grad()
        transition_loss(world, probe, rng, config).backward()
    if device == "cuda":
        torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - train_start
    world.zero_grad(set_to_none=True)
    for parameter in frozen:
        parameter.requires_grad_(False)

    horizon, other, distance = 0, torch.randn_like(latent).tanh(), 1
    with torch.no_grad():
        while distance <= 2 * config.dynamics_context:
            pair = []
            for start in (latent, other):
                seed = torch.Generator(device=device).manual_seed(0)
                rolled, _ = initial(world, start, action, seed, config)
                for _ in range(distance):
                    rolled, _ = advance(world, rolled, action, seed, config)
                pair.append(rolled.latent)
            if (pair[0] - pair[1]).abs().max() < 1e-6:
                break
            horizon, distance = distance, distance * 2

    encoder = modules.get("encoder")
    return {
        "effective_horizon_at_least": horizon,
        "deployed_parameters": sum(v for k, v in counts.items() if k in deployed),
        "training_only_parameters": sum(v for k, v in counts.items() if k not in deployed),
        "dynamics_state_elements": sum(t.numel() for pair in state.memory for t in pair),
        "encoder_state_elements": config.window
        * (config.n_latents + config.n_patches)
        * config.d_model_encoder
        * (config.depth_encoder // config.time_every)
        * 2
        if encoder is not None
        else 0,
        "peak_bytes": peak,
        "steps_per_second": 16.0 / elapsed,
        "train_steps_per_second": 4.0 / train_elapsed,
        "flops_per_step": float(flops),
        "backbone_passes_per_step": config.rungs + 1 if config.transition == "flow" else 1,
    }
