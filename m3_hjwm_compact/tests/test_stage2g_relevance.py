"""Permanent correctness tests for the Stage-2G relevance control."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from model import (  # noqa: E402
    M3HJWM,
    ModelConfig,
    enforce_frozen_encoder,
    frozen_dynamics_recipe,
)
from stage2_objectives import (  # noqa: E402
    GeneratedLossWeights,
    generated_step_components,
    weighted_generated_loss,
)
from stage2g_relevance import (  # noqa: E402
    ARM_REWARD_WEIGHTS,
    BATCH,
    PREFIX,
    WINDOW,
    build_auxiliary_contract,
    build_relevance_heads,
    generated_planner_states,
    probe_relevance,
    relevance_loss,
    relevance_pools,
    schedule_label_audit,
)
import stage2g_preflight as preflight_module  # noqa: E402
import stage2g_relevance as relevance_module  # noqa: E402


EXPECTED_PREFLIGHT_SHA256 = (
    "5551ead595a0d1ae71d4e479918176439e1a1405cbcdb11b07d9159919f5b97d"
)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        image_size=16,
        patch_size=8,
        token_dim=16,
        registers=1,
        spatial_heads=4,
        spatial_depth=1,
        temporal_backend="gru",
        temporal_depth=1,
        predictor="deterministic",
        predictor_depth=1,
        mask_ratio=0.0,
        rollout_steps=2,
    )


def indexed_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(190719)
    rewards = torch.zeros(BATCH, WINDOW - 1)
    rewards[1, PREFIX - 1] = 1.0
    rewards[2, PREFIX] = -0.2
    return {
        "obs": torch.randint(
            0,
            256,
            (BATCH, WINDOW, 3, 16, 16),
            dtype=torch.uint8,
            generator=generator,
        ),
        "actions": torch.randint(
            0,
            17,
            (BATCH, WINDOW - 1),
            generator=generator,
        ),
        "rewards": rewards,
        "continues": torch.ones(BATCH, WINDOW - 1),
        "previous_actions": torch.tensor(
            [
                [-1] + [index % 17 for index in range(WINDOW - 1)]
                for _ in range(BATCH)
            ]
        ),
    }


def episode(kind: str) -> dict:
    rewards = np.zeros(WINDOW - 1, dtype=np.float32)
    continues = np.ones(WINDOW - 1, dtype=np.float32)
    if kind == "positive":
        rewards[PREFIX - 1] = 1.0
    elif kind == "negative":
        rewards[PREFIX] = -0.2
    elif kind == "mixed":
        rewards[PREFIX - 1:PREFIX + 1] = [1.0, -0.2]
    elif kind == "terminal":
        rewards[PREFIX - 1] = 1.0
        continues[PREFIX] = 0.0
    elif kind != "zero":
        raise ValueError(kind)
    return {
        "obs": np.zeros((WINDOW, 3, 1, 1), dtype=np.uint8),
        "actions": np.arange(WINDOW - 1, dtype=np.int64) % 17,
        "rewards": rewards,
        "continues": continues,
    }


def gradients_nonzero(module) -> bool:
    return any(
        parameter.grad is not None
        and bool(parameter.grad.detach().abs().any())
        for parameter in module.parameters()
    )


def gradients_zero(module) -> bool:
    return all(
        parameter.grad is None
        or not bool(parameter.grad.detach().abs().any())
        for parameter in module.parameters()
    )


class IndexedWorld:
    def __init__(self):
        self.actions = []

    def initial_state(self, batch: int, device):
        return SimpleNamespace(tokens=torch.zeros(batch, 1, 1))

    def observe_step(self, observation, previous_action, state):
        return state

    def imagine_step(self, state, action, deterministic_mode):
        assert deterministic_mode
        self.actions.append(action.tolist())
        tokens = action.float()[:, None, None]
        return (
            SimpleNamespace(tokens=tokens),
            None,
            None,
            None,
        )

    def pool(self, tokens):
        return tokens[:, 0]


def test_generated_relevance_uses_actions_and_rewards_at_k1_k2():
    batch = indexed_batch()
    batch["actions"][:] = torch.arange(WINDOW - 1)
    batch["rewards"][:] = (
        100 + torch.arange(WINDOW - 1, dtype=torch.float32)
    )
    world = IndexedWorld()
    planner_state, reward = generated_planner_states(world, batch)
    assert world.actions == [
        [PREFIX - 1] * BATCH,
        [PREFIX] * BATCH,
    ]
    assert torch.equal(
        planner_state[..., 0],
        torch.tensor(
            [[PREFIX - 1, PREFIX]] * BATCH,
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        reward,
        torch.tensor(
            [[100 + PREFIX - 1, 100 + PREFIX]] * BATCH,
            dtype=torch.float32,
        ),
    )


def test_pool_partition_excludes_terminal_and_mixed_windows():
    train = [
        episode("zero"),
        episode("positive"),
        episode("negative"),
        episode("mixed"),
        episode("terminal"),
    ]
    pools = relevance_pools(train)
    assert pools.zero == ((0, 0),)
    assert pools.positive == ((1, 0),)
    assert pools.negative == ((2, 0),)
    assert pools.mixed == ((3, 0),)
    assert pools.terminal == ((4, 0),)


def test_auxiliary_schedule_is_deterministic_balanced_and_disjoint():
    train = (
        [episode("zero") for _ in range(40)]
        + [episode("positive") for _ in range(24)]
        + [episode("negative") for _ in range(24)]
    )
    pools = relevance_pools(train)
    first_schedule, first_probe, first_info = build_auxiliary_contract(
        pools, updates=19
    )
    second_schedule, second_probe, second_info = (
        build_auxiliary_contract(pools, updates=19)
    )
    assert first_schedule == second_schedule
    assert first_probe == second_probe
    assert first_info == second_info
    assert not set(first_schedule).intersection(first_probe)
    audit = schedule_label_audit(train, first_schedule)
    assert audit["event_window_fraction"] == 0.5
    assert audit["positive_window_fraction"] == 0.25
    assert audit["negative_window_fraction"] == 0.25
    assert audit["terminal_row_fraction"] == 0.0


def test_factorial_preserves_the_registered_generated_reward_axis():
    assert ARM_REWARD_WEIGHTS == {
        "G-LA": 0.0,
        "G-LRA": 0.10,
    }


def test_preflight_and_training_machinery_cannot_access_evaluation_tiers():
    for module in (preflight_module, relevance_module):
        source = Path(module.__file__).read_text().lower()
        for forbidden in (
            "stage2_eval_bundles",
            "stage2_dev",
            "stage2_final",
            "final_bundle",
            "manifest[\"dev\"]",
            "manifest['dev']",
            "manifest[\"final\"]",
            "manifest['final']",
        ):
            assert forbidden not in source


def test_sealed_preflight_chain_and_coefficient_are_exact():
    path = ARTIFACTS / "stage2g_preflight.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        EXPECTED_PREFLIGHT_SHA256
    )
    preflight = json.loads(path.read_text())
    assert preflight["status"] == "passed"
    assert preflight["local_regression"]["exact"] is True
    assert (
        preflight["base_schedule_sha256"]
        == "427eb8a311ac9a99ec7f5fd529added9035777a1146864c4ab53d68c2c1295d0"
    )
    contract = preflight["auxiliary_contract"]
    assert contract["probe_schedule_overlap"] == 0
    for name in ("schedule_labels", "probe_labels"):
        assert contract[name]["event_window_fraction"] == 0.5
        assert contract[name]["positive_window_fraction"] == 0.25
        assert contract[name]["negative_window_fraction"] == 0.25
        assert contract[name]["terminal_row_fraction"] == 0.0

    gradient = preflight["gradient_registration"]
    expected = (
        0.10
        * gradient["raw_generated_reward_rms"]
        / gradient["raw_auxiliary_rms"]
    )
    assert gradient["lambda_aux"] == expected
    assert 0.01 <= expected <= 10.0
    assert gradient["detached_routes"]["shared"] == 0.0
    assert gradient["detached_routes"]["auxiliary_heads"] > 0.0
    for row in gradient["auxiliary_routes"]:
        for key in (
            "shared",
            "action_input",
            "future",
            "temporal",
            "auxiliary_heads",
        ):
            assert row[key] > 0.0
        for key in (
            "reward_head",
            "continuation_head",
            "online_encoder",
            "target_encoder",
        ):
            assert row[key] == 0.0
    for arm in ("G-LA", "G-LRA"):
        before = preflight["smokes_256"][arm]["probes"]["u0"]
        after = preflight["smokes_256"][arm]["probes"]["u256"]
        assert after["loss"] < before["loss"]
        assert after["event_auroc"] > 0.55
        assert after["sign_auroc"] > 0.55
        assert after["decoded_absolute_maximum"] < 100.0
        assert (
            preflight["smokes_256"][arm]["peak_reserved_mib"]
            < 5500
        )


def test_auxiliary_head_initialization_preserves_global_rng():
    torch.manual_seed(1719)
    before = torch.get_rng_state().clone()
    first = build_relevance_heads(16, torch.device("cpu"))
    after = torch.get_rng_state().clone()
    second = build_relevance_heads(16, torch.device("cpu"))
    assert torch.equal(before, after)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_auxiliary_head_initialization_preserves_cuda_rng():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.cuda.manual_seed(1719)
    before = torch.cuda.get_rng_state().clone()
    first = build_relevance_heads(16, device)
    after = torch.cuda.get_rng_state().clone()
    second = build_relevance_heads(16, device)
    assert torch.equal(before, after)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_fixed_probe_runs_end_to_end_on_cuda():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    source = indexed_batch()
    train = []
    zero_source = copy.deepcopy(source)
    zero_source["rewards"].zero_()
    for index in range(BATCH):
        train.append({
            name: value[index].cpu().numpy()
            for name, value in zero_source.items()
            if name != "previous_actions"
        })
    for index in range(BATCH):
        train.append({
            name: value[index].cpu().numpy()
            for name, value in source.items()
            if name != "previous_actions"
        })
    torch.manual_seed(505)
    world = enforce_frozen_encoder(M3HJWM(tiny_config()).to(device))
    heads = build_relevance_heads(world.cfg.token_dim, device)
    output = probe_relevance(
        world,
        heads,
        train,
        [(index, 0) for index in range(2 * BATCH)],
    )
    assert output["event_rows"] == 2
    assert output["zero_rows"] == 14
    assert output["event_auroc"] is not None
    assert output["sign_auroc"] is not None
    assert np.isfinite(output["loss"])


def test_auxiliary_gradient_reaches_shared_world_but_not_task_heads():
    torch.manual_seed(505)
    world = enforce_frozen_encoder(M3HJWM(tiny_config()))
    heads = build_relevance_heads(
        world.cfg.token_dim, torch.device("cpu")
    )
    output = relevance_loss(world, heads, indexed_batch())
    output["loss"].backward()
    assert gradients_nonzero(world.action_input)
    assert gradients_nonzero(world.future)
    assert gradients_nonzero(world.temporal)
    assert gradients_nonzero(heads)
    assert gradients_zero(world.reward)
    assert gradients_zero(world.continuation)
    assert gradients_zero(world.online_encoder)
    assert gradients_zero(world.target_encoder)


def test_detached_auxiliary_is_vacuous_for_world_gradients():
    torch.manual_seed(505)
    world = enforce_frozen_encoder(M3HJWM(tiny_config()))
    heads = build_relevance_heads(
        world.cfg.token_dim, torch.device("cpu")
    )
    output = relevance_loss(
        world, heads, indexed_batch(), detach_world=True
    )
    output["loss"].backward()
    assert gradients_nonzero(heads)
    for module in (
        world.action_input,
        world.future,
        world.temporal,
        world.reward,
        world.continuation,
        world.online_encoder,
        world.target_encoder,
    ):
        assert gradients_zero(module)


def _one_world_update(world, batch, optimizer, extra=None):
    output = world(batch, frozen_dynamics_recipe())
    components = generated_step_components(
        world, batch, prefix=PREFIX, steps=2
    )
    generated = weighted_generated_loss(
        components,
        GeneratedLossWeights(
            latent=1.0, reward=0.1, continuation=0.0
        ),
    )
    loss = output.loss + generated
    if extra is not None:
        loss = loss + 0.5 * extra
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [
            parameter for parameter in world.parameters()
            if parameter.requires_grad
        ],
        100.0,
    )
    optimizer.step()
    world.mark_parameters_updated()


def test_detached_auxiliary_leaves_world_update_bit_exact():
    batch = indexed_batch()
    torch.manual_seed(505)
    reference = enforce_frozen_encoder(M3HJWM(tiny_config()))
    torch.manual_seed(505)
    observed = enforce_frozen_encoder(M3HJWM(tiny_config()))
    heads = build_relevance_heads(
        observed.cfg.token_dim, torch.device("cpu")
    )
    reference_optimizer = torch.optim.AdamW(
        [
            parameter for parameter in reference.parameters()
            if parameter.requires_grad
        ],
        lr=1e-4,
    )
    observed_optimizer = torch.optim.AdamW(
        [
            parameter for parameter in observed.parameters()
            if parameter.requires_grad
        ],
        lr=1e-4,
    )
    detached = relevance_loss(
        observed, heads, copy.deepcopy(batch), detach_world=True
    )["loss"]
    _one_world_update(reference, copy.deepcopy(batch), reference_optimizer)
    _one_world_update(
        observed, copy.deepcopy(batch), observed_optimizer, detached
    )
    for name, value in reference.state_dict().items():
        assert torch.equal(value, observed.state_dict()[name]), name
