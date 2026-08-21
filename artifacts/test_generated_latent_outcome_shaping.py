from __future__ import annotations

import copy

import torch

from artifacts.train_generated_latent_outcome_shaping import (
    LatentContinuationHead,
    generated_terminal_successors,
    gradient_preflight,
    terminal_objective,
    update_pair,
)
from d4mj.config import Config
from d4mj.data import Batch
from d4mj.train import _balance, _share_initialisation, _update, optimizer
from d4mj.transition import World, transition_loss


def config() -> Config:
    return Config(
        transition="direct",
        time_mixer="attention",
        n_latents=4,
        d_bottleneck=4,
        packing=2,
        d_model=32,
        depth=4,
        n_heads=4,
        time_every=2,
        n_register=2,
        n_agent=1,
        mamba_headdim=16,
        batch=2,
        gradient_checkpointing=False,
        device="cpu",
    )


def batch(cfg: Config) -> Batch:
    torch.manual_seed(9)
    return Batch(
        led_to_action=torch.tensor([[cfg.n_actions, 3, 5]]),
        reward=torch.zeros(1, 3),
        terminated=torch.tensor([[False, False, True]]),
        truncated=torch.zeros(1, 3, dtype=torch.bool),
        valid=torch.ones(1, 3, dtype=torch.bool),
        scored=torch.ones(1, 3, dtype=torch.bool),
        burn_in=0,
        relevant=None,
        support=None,
        latents=torch.randn(1, 3, cfg.n_spatial, cfg.d_spatial).tanh(),
    )


def initialized(cfg: Config) -> tuple[World, LatentContinuationHead]:
    torch.manual_seed(cfg.seed + 1)
    world = _share_initialisation(World(cfg), cfg)
    torch.manual_seed(cfg.seed + 18_200)
    return world, LatentContinuationHead(cfg)


def test_terminal_objective_pairs_matching_alive_dead_successors() -> None:
    cfg = config()
    world, head = initialized(cfg)
    data = batch(cfg)
    loss, values = terminal_objective(
        world,
        head,
        data,
        torch.Generator().manual_seed(1),
        cfg,
        stop_generated=False,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert values["predicted"].shape == values["observed"].shape == (
        1,
        2,
        cfg.n_spatial,
        cfg.d_spatial,
    )
    assert torch.equal(values["target"], torch.tensor([[1.0, 0.0]]))


def test_terminal_gradient_stop_is_exact() -> None:
    cfg = config()
    world, head = initialized(cfg)
    routing = gradient_preflight(world, head, batch(cfg), cfg)
    assert routing["allowed"]["world_gradient_norm"] > 0
    assert routing["allowed"]["generated_latent_gradient_norm"] > 0
    assert routing["stopped"] == {
        "world_gradient_norm": 0.0,
        "generated_latent_gradient_norm": 0.0,
    }


def test_stopped_world_update_is_identical_to_mse_only() -> None:
    cfg = config()
    control, head = initialized(cfg)
    stopped = copy.deepcopy(control)
    control_optimizer = optimizer([control], cfg)
    stopped_optimizer = optimizer([stopped], cfg)
    head_optimizer = optimizer([head], cfg)
    control_balance, stopped_balance, head_balance = {}, {}, {}
    data = batch(cfg)
    rng_control = torch.Generator().manual_seed(2)
    rng_stopped = torch.Generator().manual_seed(2)
    control_dynamics = transition_loss(control, data, rng_control, cfg)
    stopped_dynamics = transition_loss(stopped, data, rng_stopped, cfg)
    assert torch.equal(control_dynamics, stopped_dynamics)
    control_loss = _balance({"dynamics": control_dynamics}, control_balance, cfg)
    stopped_loss = _balance({"dynamics": stopped_dynamics}, stopped_balance, cfg)
    outcome, _ = terminal_objective(
        stopped,
        head,
        data,
        torch.Generator().manual_seed(3),
        cfg,
        stop_generated=True,
    )
    head_loss = _balance({"terminal_outcome": outcome}, head_balance, cfg)
    _update(control_optimizer, control_loss, [control], cfg, 0)
    update_pair(
        stopped_optimizer,
        head_optimizer,
        stopped_loss + head_loss,
        stopped,
        head,
        cfg,
        0,
    )
    for name, value in control.state_dict().items():
        assert torch.equal(value, stopped.state_dict()[name]), name


def test_second_successor_depends_on_both_tail_actions() -> None:
    cfg = config()
    world, _ = initialized(cfg)
    data = batch(cfg)
    original = generated_terminal_successors(
        world, data, torch.Generator().manual_seed(4), cfg
    )
    penultimate = Batch(
        **{**vars(data), "led_to_action": torch.tensor([[cfg.n_actions, 4, 5]])}
    )
    final = Batch(
        **{**vars(data), "led_to_action": torch.tensor([[cfg.n_actions, 3, 6]])}
    )
    changed_first = generated_terminal_successors(
        world, penultimate, torch.Generator().manual_seed(4), cfg
    )
    changed_second = generated_terminal_successors(
        world, final, torch.Generator().manual_seed(4), cfg
    )
    assert not torch.equal(original[:, 0], changed_first[:, 0])
    assert not torch.equal(original[:, 1], changed_first[:, 1])
    assert torch.equal(original[:, 0], changed_second[:, 0])
    assert not torch.equal(original[:, 1], changed_second[:, 1])
