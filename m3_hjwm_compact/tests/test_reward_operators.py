"""Source-alignment and compatibility checks for reward distributions."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from checkpoint import (  # noqa: E402
    load_world_checkpoint,
    save_world_checkpoint,
)
from model import (  # noqa: E402
    LossConfig,
    M3HJWM,
    ModelConfig,
    RewardHead,
    decode_dreamerv3_two_hot,
    decode_two_hot,
    dreamerv3_reward_support,
    dreamerv3_two_hot,
    two_hot,
)
from stage2f_reward_operator import zero_reward_output  # noqa: E402
import stage2f_preflight as preflight_module  # noqa: E402
import stage2f_train as train_module  # noqa: E402


def small_config(operator: str = "local_symlog") -> ModelConfig:
    return ModelConfig(
        image_size=16,
        patch_size=8,
        token_dim=16,
        spatial_heads=4,
        registers=2,
        predictor_depth=1,
        temporal_backend="gru",
        reward_operator=operator,
    )


def independent_support(bins: int = 255) -> np.ndarray:
    half = np.linspace(-20.0, 0.0, (bins - 1) // 2 + 1).astype(
        np.float32
    )
    half = np.sign(half) * np.expm1(np.abs(half))
    return np.concatenate((half, -half[:-1][::-1])).astype(np.float32)


def independent_targets(
    values: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    output = np.zeros((*values.shape, len(support)), dtype=np.float32)
    for row, raw in enumerate(values):
        value = float(np.clip(raw, support[0], support[-1]))
        below = int(np.searchsorted(support, value, side="right") - 1)
        above = int(np.searchsorted(support, value, side="left"))
        below = int(np.clip(below, 0, len(support) - 1))
        above = int(np.clip(above, 0, len(support) - 1))
        if below == above:
            output[row, below] = 1.0
        else:
            distance_below = abs(float(support[below]) - value)
            distance_above = abs(float(support[above]) - value)
            total = distance_below + distance_above
            output[row, below] = distance_above / total
            output[row, above] = distance_below / total
    return output


def test_dreamerv3_support_matches_pinned_odd_bin_equations():
    observed = dreamerv3_reward_support(255, -20.0, 20.0)
    expected = torch.from_numpy(independent_support())
    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
    assert bool(torch.all(observed[1:] > observed[:-1]))
    assert observed[127] == 0
    assert torch.equal(observed, -observed.flip(0))
    assert observed[0] == pytest.approx(-485165184.0)
    assert observed[-1] == pytest.approx(485165184.0)


def test_dreamerv3_targets_match_original_space_reference():
    support = dreamerv3_reward_support(255, -20.0, 20.0)
    values = torch.tensor(
        [-1e12, -0.7, -0.2, 0.0, 0.1, 1.1, 1e12]
    )
    observed = dreamerv3_two_hot(values, support)
    expected = torch.from_numpy(independent_targets(
        values.numpy(), support.numpy()
    ))
    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(
        observed.sum(-1), torch.ones(len(values)), rtol=0, atol=1e-7
    )
    assert bool(torch.all(observed >= 0))
    assert observed[0, 0] == 1
    assert observed[-1, -1] == 1
    assert observed[3, 127] == 1
    assert torch.count_nonzero(observed[3]) == 1


def test_dreamerv3_loss_matches_independent_target_cross_entropy():
    generator = torch.Generator().manual_seed(1907)
    logits = torch.randn(7, 255, generator=generator)
    rewards = torch.tensor([-0.7, -0.2, -0.1, 0.0, 0.1, 1.0, 1.1])
    head = RewardHead(small_config("dreamerv3_symexp"))
    observed = head.loss(logits, rewards)
    target = torch.from_numpy(independent_targets(
        rewards.numpy(), independent_support()
    ))
    expected = -(target * logits.log_softmax(-1)).sum(-1)
    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-6)


def test_symmetric_decoder_is_exact_zero_and_operators_diverge():
    support = dreamerv3_reward_support(255, -20.0, 20.0)
    uniform = torch.zeros(4, 255)
    assert torch.equal(
        decode_dreamerv3_two_hot(uniform, support), torch.zeros(4)
    )

    logits = torch.full((1, 255), -20.0)
    logits[0, 120] = 0.0
    logits[0, 130] = 0.0
    dreamerv3 = decode_dreamerv3_two_hot(logits, support)
    local = decode_two_hot(logits, -20.0, 20.0)
    assert not torch.allclose(dreamerv3, local)


def test_local_operator_remains_the_exact_historical_math():
    generator = torch.Generator().manual_seed(77)
    logits = torch.randn(9, 255, generator=generator)
    rewards = torch.randn(9, generator=generator)
    head = RewardHead(small_config())
    expected_target = two_hot(rewards, 255, -20.0, 20.0)
    expected_loss = -(
        expected_target * logits.log_softmax(-1)
    ).sum(-1)
    assert torch.equal(head.loss(logits, rewards), expected_loss)
    assert torch.equal(
        head.decode(logits), decode_two_hot(logits, -20.0, 20.0)
    )


def test_operator_axis_does_not_change_parameter_initialization():
    local_cfg = small_config()
    dreamer_cfg = replace(
        local_cfg, reward_operator="dreamerv3_symexp"
    )
    torch.manual_seed(911)
    local = RewardHead(local_cfg)
    torch.manual_seed(911)
    dreamer = RewardHead(dreamer_cfg)
    assert local.state_dict().keys() == dreamer.state_dict().keys()
    for key, value in local.state_dict().items():
        assert torch.equal(value, dreamer.state_dict()[key])
    assert "_dreamerv3_support" not in local.state_dict()
    assert "_dreamerv3_support" not in dreamer.state_dict()


def test_zero_output_initialization_is_matched_and_decodes_zero():
    heads = []
    for operator in ("local_symlog", "dreamerv3_symexp"):
        torch.manual_seed(81)
        head = RewardHead(small_config(operator))
        with torch.no_grad():
            head.net[-1].weight.zero_()
            head.net[-1].bias.zero_()
        heads.append(head)
        context = torch.randn(5, 16)
        logits = head(context)
        assert torch.count_nonzero(logits) == 0
        decoded = head.decode(logits)
        if operator == "dreamerv3_symexp":
            assert torch.equal(decoded, torch.zeros(5))
        else:
            # Preserve the historical local left-to-right reduction exactly;
            # unlike DreamerV3's paired sum it is only numerically near zero.
            assert torch.equal(
                decoded, decode_two_hot(logits, -20.0, 20.0)
            )
            assert float(decoded.detach().abs().max()) < 1e-6
    for key, value in heads[0].state_dict().items():
        assert torch.equal(value, heads[1].state_dict()[key])


def test_zero_output_initialization_changes_only_last_reward_linear():
    torch.manual_seed(33)
    world = M3HJWM(small_config())
    before = {
        name: value.clone() for name, value in world.state_dict().items()
    }
    zero_reward_output(world)
    changed = {
        name
        for name, value in world.state_dict().items()
        if not torch.equal(value, before[name])
    }
    assert changed == {
        "reward.net.3.weight",
        "reward.net.3.bias",
    }
    assert torch.count_nonzero(world.reward.net[-1].weight) == 0
    assert torch.count_nonzero(world.reward.net[-1].bias) == 0


def test_legacy_checkpoint_defaults_to_local_operator(tmp_path):
    cfg = small_config()
    world = M3HJWM(cfg)
    path = tmp_path / "legacy.pt"
    save_world_checkpoint(path, world, LossConfig())
    payload = torch.load(path, weights_only=False)
    payload["model_config"].pop("reward_operator")
    torch.save(payload, path)

    loaded, _ = load_world_checkpoint(
        path, torch.device("cpu"), expect_config=cfg
    )
    assert loaded.cfg.reward_operator == "local_symlog"
    for key, value in world.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_dreamerv3_checkpoint_roundtrip_and_config_drift(tmp_path):
    cfg = small_config("dreamerv3_symexp")
    world = M3HJWM(cfg)
    path = tmp_path / "dreamerv3.pt"
    digest = save_world_checkpoint(path, world, LossConfig())
    loaded, _ = load_world_checkpoint(
        path,
        torch.device("cpu"),
        expect_config=cfg,
        expect_sha256=digest,
    )
    assert loaded.cfg.reward_operator == "dreamerv3_symexp"
    with pytest.raises(RuntimeError, match="reward_operator"):
        load_world_checkpoint(
            path,
            torch.device("cpu"),
            expect_config=replace(
                cfg, reward_operator="local_symlog"
            ),
        )


def test_dreamerv3_config_rejects_non_source_support():
    with pytest.raises(ValueError, match="odd"):
        M3HJWM(replace(
            small_config("dreamerv3_symexp"), reward_bins=254
        ))
    with pytest.raises(ValueError, match="symmetric"):
        M3HJWM(replace(
            small_config("dreamerv3_symexp"), reward_high=19.0
        ))


def test_preflight_and_training_cannot_access_evaluation_tiers():
    for module in (preflight_module, train_module):
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
    preflight = (
        COMPACT_ROOT.parent
        / "reviews/artifacts/stage2f_preflight.json"
    )
    assert train_module.EXPECTED_PREFLIGHT_SHA256 == hashlib.sha256(
        preflight.read_bytes()
    ).hexdigest()
