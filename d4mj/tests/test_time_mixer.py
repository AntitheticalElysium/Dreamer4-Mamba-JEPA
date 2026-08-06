from dataclasses import replace

import pytest
import torch

from d4mj.state import WorldState
from d4mj.time_mixer import time_mixer
from d4mj.transition import World, advance, commit_inputs

MIXERS = ("attention", "mamba")


def cuda_only(config, mixer):
    if mixer == "mamba" and config.device != "cuda":
        pytest.skip("Mamba-2 kernels require CUDA")


def prefix_gradient(config, mixer, detach: bool):
    """Gradient reaching a differentiable prefix through carried memory."""
    module = time_mixer(config, config.d_model, config.dynamics_context).to(config.device)
    prefix = torch.randn(2, 4, config.d_model, device=config.device, requires_grad=True)
    _, memory = module(prefix, None, 0)
    if detach:
        memory = tuple(tensor.detach() for tensor in memory)
    new = torch.randn(2, 1, config.d_model, device=config.device, requires_grad=True)
    out, _ = module(new, memory, 4)
    out.sum().backward()
    return prefix.grad


@pytest.mark.parametrize("mixer", MIXERS)
def test_carried_memory_gradient_is_backend_specific(config, mixer):
    """The asymmetry this exists to contain: official Mamba-2 updates its
    `InferenceParams` cache in place, so its recurrent step is not differentiable
    with respect to the state it receives, while attention's concatenated cache is.
    Leaving that unequalised makes the two arms train different objectives."""
    arm = replace(config, time_mixer=mixer)
    cuda_only(arm, mixer)
    torch.manual_seed(0)
    grad = prefix_gradient(arm, mixer, detach=False)
    if mixer == "attention":
        assert grad is not None and float(grad.abs().sum()) > 0
    else:
        assert grad is None


@pytest.mark.parametrize("mixer", MIXERS)
def test_detaching_equalises_both_arms(config, mixer):
    """`advance` detaches incoming memory so both arms truncate identically."""
    arm = replace(config, time_mixer=mixer)
    cuda_only(arm, mixer)
    torch.manual_seed(0)
    assert prefix_gradient(arm, mixer, detach=True) is None


@pytest.mark.parametrize("mixer", MIXERS)
def test_advance_does_not_backpropagate_into_carried_memory(config, mixer):
    """The end-to-end statement of the same contract, on the runtime path the
    Direct generated-prefix loss actually uses.

    `state.features` is detached here so memory is the *only* remaining route back
    to the prefix. Features are a legitimate gradient path and carry the recurrent
    prediction signal `auto_steps: 2` is for; carried memory is an implementation
    cache, and it is the one the two backends disagree about."""
    arm = replace(config, time_mixer=mixer, transition="direct")
    cuda_only(arm, mixer)
    torch.manual_seed(0)
    world = World(arm).to(arm.device)
    generator = torch.Generator(device=arm.device).manual_seed(1)

    prefix = torch.randn(2, 4, arm.n_spatial, arm.d_spatial, device=arm.device).tanh()
    prefix.requires_grad_(True)
    actions = torch.zeros(2, 4, dtype=torch.long, device=arm.device)
    committed, conditioning = commit_inputs(prefix, generator, arm)
    features, _, memory = world(None, actions, committed, conditioning)

    latent = torch.randn(2, 1, arm.n_spatial, arm.d_spatial, device=arm.device).tanh()
    state = WorldState(latent, memory, 4, features[:, -1:].detach())
    stepped, _ = advance(world, state, actions[:, :1], generator, arm)
    stepped.features.sum().backward()
    assert prefix.grad is None or float(prefix.grad.abs().sum()) == 0.0


@pytest.mark.parametrize("mixer", MIXERS)
def test_step_matches_scan(config, mixer):
    arm = replace(config, time_mixer=mixer)
    cuda_only(arm, mixer)
    torch.manual_seed(0)
    module = time_mixer(arm, arm.d_model, arm.dynamics_context).to(arm.device).eval()
    x = torch.randn(2, 6, arm.d_model, device=arm.device)
    with torch.no_grad():
        scanned, _ = module(x, None, 0)
        stepped, memory = [], None
        for index in range(x.shape[1]):
            out, memory = module(x[:, index : index + 1], memory, index)
            stepped.append(out)
    drift = (torch.cat(stepped, dim=1) - scanned).abs().max() / scanned.abs().max()
    assert drift < 1e-3
