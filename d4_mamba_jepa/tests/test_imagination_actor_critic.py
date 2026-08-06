import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from d4_mamba_jepa.common import BCPolicy
from d4_mamba_jepa.checkpoint import file_sha256
from d4_mamba_jepa.data import Episode, EpisodeReplay
from d4_mamba_jepa.imagination_actor_critic import (
    FORMAT,
    ValueHead,
    ReplayContextSampler,
    _direct_execution_policy,
    actor_critic_update,
    actor_source_report,
    decode_symlog_distribution,
    freeze_module,
    load_imagination_actor_critic,
    module_state_sha256,
    pmpo_loss,
    state_dict_l2_distance,
    td_lambda_returns,
    twohot_symlog_targets,
    unfreeze_module,
)
from d4_mamba_jepa.model import D4LiteWorld
from d4_mamba_jepa.tests.test_baseline import tiny_config


def test_actor_sources_are_exactly_pinned():
    report = actor_source_report()
    assert len(report) == 5
    assert all(len(item["sha256"]) == 64 for item in report.values())


def test_td_lambda_uses_next_reward_value_and_terminal():
    rewards = torch.tensor([[1.0, 2.0]])
    continues = torch.tensor([[1.0, 0.0]])
    values = torch.tensor([[10.0, 20.0, 30.0]])
    returns = td_lambda_returns(
        rewards,
        continues,
        values,
        gamma=0.9,
        lambda_=0.5,
    )
    # R_1 = 2 because c_2=0. R_0 = 1 + .9*(.5*20 + .5*2).
    torch.testing.assert_close(returns, torch.tensor([[10.9, 2.0]]))


def test_twohot_symlog_targets_are_adjacent_distributions():
    centers = torch.linspace(-3.0, 3.0, 13)
    targets = twohot_symlog_targets(
        torch.tensor([0.0, 1.5, -2.25]),
        centers,
    )
    torch.testing.assert_close(targets.sum(dim=-1), torch.ones(3))
    assert (targets > 0).sum(dim=-1).max().item() <= 2
    assert torch.isfinite(targets).all()


def test_pmpo_gradient_increases_positive_and_decreases_negative_actions():
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    prior = torch.zeros_like(logits)
    actions = torch.zeros(1, 2, dtype=torch.long)
    advantages = torch.tensor([[1.0, -1.0]])
    loss, metrics = pmpo_loss(
        logits,
        prior,
        actions,
        advantages,
        alpha=0.5,
        beta=0.0,
    )
    loss.backward()
    # Gradient descent raises chosen action 0 for A>=0 and lowers it for A<0.
    assert logits.grad[0, 0, 0] < 0
    assert logits.grad[0, 1, 0] > 0
    assert metrics["positive_count"].item() == 1
    assert metrics["negative_count"].item() == 1


def test_reverse_prior_kl_is_zero_at_exact_bc_initialization():
    logits = torch.tensor([[[1.0, -0.5], [0.2, 0.4]]])
    actions = torch.tensor([[0, 1]])
    advantages = torch.tensor([[1.0, -1.0]])
    _, metrics = pmpo_loss(
        logits,
        logits.clone(),
        actions,
        advantages,
        alpha=0.5,
        beta=0.3,
    )
    torch.testing.assert_close(metrics["kl_mean"], torch.tensor(0.0))


def test_value_distribution_starts_at_zero_expectation():
    torch.manual_seed(3)
    value = ValueHead(
        d_model=16,
        num_bins=51,
        log_low=-5.0,
        log_high=5.0,
    )
    tokens = torch.randn(4, 7, 2, 16)
    logits, centers = value(tokens)
    assert not logits.any()
    decoded = decode_symlog_distribution(logits, centers)
    torch.testing.assert_close(decoded, torch.zeros_like(decoded))
    wide = ValueHead(
        d_model=16,
        num_bins=255,
        log_low=-10.0,
        log_high=10.0,
    )
    wide_logits, wide_centers = wide(tokens)
    assert (
        decode_symlog_distribution(wide_logits, wide_centers).abs().max()
        <= 1e-6
    )


def test_actor_copy_and_frozen_prior_have_exact_identity():
    torch.manual_seed(5)
    bc = BCPolicy(d_model=16, n_actions=2)
    actor = copy.deepcopy(bc)
    prior = freeze_module(copy.deepcopy(bc))
    assert module_state_sha256(actor) == module_state_sha256(prior)
    assert all(not parameter.requires_grad for parameter in prior.parameters())
    before = copy.deepcopy(actor.state_dict())
    with torch.no_grad():
        next(iter(actor.parameters())).add_(1e-4)
    assert state_dict_l2_distance(before, actor.state_dict()) > 0.0
    assert module_state_sha256(actor) != module_state_sha256(prior)

    frozen_bc = freeze_module(copy.deepcopy(bc))
    trainable_actor = unfreeze_module(copy.deepcopy(frozen_bc))
    assert all(
        parameter.requires_grad
        for parameter in trainable_actor.parameters()
    )
    assert module_state_sha256(trainable_actor) == module_state_sha256(frozen_bc)


def test_context_sampler_never_starts_imagination_from_terminal_state():
    replay = EpisodeReplay()
    for length in (10, 12):
        replay.add(
            Episode(
                obs=np.zeros((length, 3, 8, 8), dtype=np.uint8),
                actions=np.arange(length - 1, dtype=np.int64) % 2,
                rewards=np.ones(length - 1, dtype=np.float32),
                continues=np.asarray(
                    [1.0] * (length - 2) + [0.0],
                    dtype=np.float32,
                ),
            )
        )
    sampler = ReplayContextSampler(
        replay,
        context=8,
        device=torch.device("cpu"),
        seed=7,
    )
    batch = sampler.sample(3)
    assert batch.observations.shape == (3, 8, 3, 8, 8)
    assert batch.led_to_continues[:, -1].eq(1.0).all()


def test_context_sampler_rejects_oversized_unique_batch():
    replay = EpisodeReplay()
    replay.add(
        Episode(
            obs=np.zeros((9, 3, 8, 8), dtype=np.uint8),
            actions=np.zeros(8, dtype=np.int64),
            rewards=np.ones(8, dtype=np.float32),
            continues=np.asarray([1.0] * 7 + [0.0], dtype=np.float32),
        )
    )
    sampler = ReplayContextSampler(
        replay,
        context=8,
        device=torch.device("cpu"),
        seed=11,
    )
    with pytest.raises(ValueError, match="batch size"):
        sampler.sample(2)


def test_one_update_changes_only_actor_and_value():
    torch.manual_seed(13)
    device = torch.device("cpu")
    cfg = tiny_config(n_actions=2)
    world = freeze_module(D4LiteWorld(cfg))
    actor = BCPolicy(d_model=cfg.dynamics_d_model, n_actions=2)
    prior = freeze_module(copy.deepcopy(actor))
    value = ValueHead(
        d_model=cfg.dynamics_d_model,
        num_bins=cfg.reward_bins,
        log_low=cfg.reward_log_low,
        log_high=cfg.reward_log_high,
    )
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(value.parameters()),
        lr=1e-4,
    )
    replay = EpisodeReplay()
    replay.add(
        Episode(
            obs=np.zeros((5, 3, cfg.image_size, cfg.image_size), np.uint8),
            actions=np.asarray([0, 1, 0, 1], np.int64),
            rewards=np.ones(4, np.float32),
            continues=np.asarray([1.0, 1.0, 1.0, 0.0], np.float32),
        )
    )
    context_batch = ReplayContextSampler(
        replay,
        context=2,
        device=device,
        seed=17,
    ).sample(2)
    world_before = module_state_sha256(world)
    prior_before = module_state_sha256(prior)
    actor_before = copy.deepcopy(actor.state_dict())
    value_before = copy.deepcopy(value.state_dict())

    metrics = actor_critic_update(
        world=world,
        actor=actor,
        prior=prior,
        value=value,
        optimizer=optimizer,
        context_batch=context_batch,
        horizon=2,
        denoise_steps=2,
        context=2,
        gamma=0.997,
        lambda_=0.95,
        alpha=0.5,
        beta=0.3,
        gradient_clip=1.0,
        generator=torch.Generator().manual_seed(19),
        device=device,
    )

    assert np.isfinite(metrics["total_loss"])
    assert module_state_sha256(world) == world_before
    assert module_state_sha256(prior) == prior_before
    assert state_dict_l2_distance(actor_before, actor.state_dict()) > 0
    assert state_dict_l2_distance(value_before, value.state_dict()) > 0
    assert all(parameter.grad is None for parameter in world.parameters())
    assert all(parameter.grad is None for parameter in prior.parameters())
    assert any(parameter.grad is not None for parameter in actor.parameters())
    assert any(parameter.grad is not None for parameter in value.parameters())


def _write_actor_checkpoint(
    path: Path,
    *,
    world_sha256: str = "world",
    bc_sha256: str = "bc",
    source_report: dict | None = None,
) -> str:
    torch.manual_seed(23)
    actor = BCPolicy(d_model=16, n_actions=2)
    prior = freeze_module(copy.deepcopy(actor))
    value = ValueHead(
        d_model=16,
        num_bins=17,
        log_low=-10.0,
        log_high=10.0,
    )
    payload = {
        "format": FORMAT,
        "world_checkpoint_sha256": world_sha256,
        "bc_checkpoint_sha256": bc_sha256,
        "actor": actor.state_dict(),
        "prior": prior.state_dict(),
        "value": value.state_dict(),
        "config": {
            "d_model": 16,
            "n_actions": 2,
            "value_bins": 17,
            "value_log_low": -10.0,
            "value_log_high": 10.0,
        },
        "provenance": {
            "actor_sources": (
                actor_source_report()
                if source_report is None
                else source_report
            )
        },
        "frozen_invariants": {
            "prior_tensor_sha256_after": module_state_sha256(prior),
        },
    }
    torch.save(payload, path)
    return file_sha256(path)


def test_actor_checkpoint_rejects_digest_pairing_and_source_drift(tmp_path):
    checkpoint = tmp_path / "actor.pt"
    digest = _write_actor_checkpoint(checkpoint)
    actor, prior, value, _ = load_imagination_actor_critic(
        checkpoint,
        expected_sha256=digest,
        expected_world_sha256="world",
        expected_bc_sha256="bc",
        device=torch.device("cpu"),
    )
    assert all(
        not parameter.requires_grad
        for module in (actor, prior, value)
        for parameter in module.parameters()
    )
    with pytest.raises(RuntimeError, match="digest drift"):
        load_imagination_actor_critic(
            checkpoint,
            expected_sha256="0" * 64,
            expected_world_sha256="world",
            expected_bc_sha256="bc",
            device=torch.device("cpu"),
        )
    with pytest.raises(RuntimeError, match="world pairing drift"):
        load_imagination_actor_critic(
            checkpoint,
            expected_sha256=digest,
            expected_world_sha256="different",
            expected_bc_sha256="bc",
            device=torch.device("cpu"),
        )
    with pytest.raises(RuntimeError, match="BC pairing drift"):
        load_imagination_actor_critic(
            checkpoint,
            expected_sha256=digest,
            expected_world_sha256="world",
            expected_bc_sha256="different",
            device=torch.device("cpu"),
        )

    drifted = tmp_path / "drifted.pt"
    drifted_digest = _write_actor_checkpoint(drifted, source_report={})
    with pytest.raises(RuntimeError, match="source provenance drift"):
        load_imagination_actor_critic(
            drifted,
            expected_sha256=drifted_digest,
            expected_world_sha256="world",
            expected_bc_sha256="bc",
            device=torch.device("cpu"),
        )


def test_direct_actor_evaluation_cannot_route_to_shooting():
    assert _direct_execution_policy("imagination_actor") == "bc_policy"
    assert _direct_execution_policy("bc_policy") == "bc_policy"
    assert _direct_execution_policy("random") == "random"
    assert _direct_execution_policy("oracle_reference") == "oracle_reference"
    with pytest.raises(ValueError, match="unsupported direct policy"):
        _direct_execution_policy("planner")
