import time

import torch
from torch import Tensor, nn

from .agent import Heads, head_targets
from .config import Config
from .data import Batch
from .state import WorldState
from .transition import World, advance


@torch.no_grad()
def multistep_error(world: World, batch: Batch, rng: torch.Generator, config: Config) -> list[float]:
    """Per-step prediction error under the real runtime path, from a real prefix.

    This is what the imagination horizon is set from. Inheriting a horizon and
    hoping is how a rollout ends up spending most of its steps outside anything
    the model was trained on.
    """
    context = batch.latents[:, :1]
    state = WorldState(context, None, 0)
    errors = []
    for step in range(1, batch.latents.shape[1]):
        state, _ = advance(world, state, batch.led_to_action[:, step : step + 1], rng, config)
        errors.append(float((state.latent - batch.latents[:, step : step + 1]).pow(2).mean()))
    return errors


@torch.no_grad()
def latent_stats(world: World, batch: Batch, rng: torch.Generator, config: Config) -> dict[str, float]:
    """Range *and* scale. A bounded readout fixes the range; it does nothing about
    contraction toward the conditional mean, which is the failure that looks like a
    working model in every one-step metric."""
    real = batch.latents
    state = WorldState(real[:, :1], None, 0)
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

    Mamba fixes the dynamics memory only; the encoder cache still grows with its
    window, so a single 'state size' would overstate what the substitution buys.
    """
    deployed = {"encoder", "world", "heads"}
    counts = {name: sum(p.numel() for p in m.parameters()) for name, m in modules.items()}
    latent = torch.randn(1, 1, config.n_spatial, config.d_spatial)
    action = torch.zeros(1, 1, dtype=torch.long)
    rng = torch.Generator().manual_seed(0)

    state = WorldState(latent, None, 0)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(8):
            state, _ = advance(world, state, action, rng, config)
    elapsed = time.perf_counter() - start

    return {
        "deployed_parameters": sum(v for k, v in counts.items() if k in deployed),
        "training_only_parameters": sum(v for k, v in counts.items() if k not in deployed),
        "dynamics_state_elements": sum(t.numel() for pair in state.memory for t in pair),
        "steps_per_second": 8.0 / elapsed,
        "passes_per_step": config.rungs + 1 if config.transition == "flow" else 2,
    }
