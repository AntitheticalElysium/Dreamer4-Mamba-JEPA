import numpy as np
import torch

from d4_mamba_jepa.cartpole_baseline import (
    CartPoleBCPolicy,
    paired_bootstrap_interval,
    preprocess_rgb,
    sample_cartpole_sequences,
)
from d4_mamba_jepa.data import Episode, EpisodeReplay
from d4_mamba_jepa.source import GYMNASIUM_CARTPOLE, verify_installed_cartpole


def _synthetic_cartpole_frame(cart_x: int) -> np.ndarray:
    frame = np.full((400, 600, 3), 255, dtype=np.uint8)
    frame[299:302] = 0
    frame[285:305, cart_x - 20 : cart_x + 20] = [129, 132, 203]
    frame[180:286, cart_x - 2 : cart_x + 2] = [202, 152, 101]
    return frame


def test_cartpole_source_and_pixel_adapter_are_deterministic():
    assert verify_installed_cartpole() == GYMNASIUM_CARTPOLE.sha256
    previous = _synthetic_cartpole_frame(290)
    current = _synthetic_cartpole_frame(310)
    first = preprocess_rgb(current, previous_frame=previous)
    second = preprocess_rgb(current, previous_frame=previous)
    assert first.shape == (3, 64, 64)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert np.count_nonzero(first[0]) > 0
    assert np.count_nonzero(first[1]) > 0
    assert np.count_nonzero(first[2]) > 0


def test_terminal_sampler_preserves_led_to_alignment():
    replay = EpisodeReplay()
    for action_offset in (0, 1):
        replay.add(
            Episode(
                obs=np.zeros((13, 3, 16, 16), dtype=np.uint8),
                actions=(
                    np.arange(12, dtype=np.int64) + action_offset
                ) % 2,
                rewards=np.ones(12, dtype=np.float32),
                continues=np.asarray(
                    [1.0] * 11 + [0.0], dtype=np.float32
                ),
            )
        )
    batch = sample_cartpole_sequences(
        replay,
        batch_size=2,
        sequence_length=12,
        terminal_fraction=1.0,
        device=torch.device("cpu"),
        rng=np.random.default_rng(5),
    )
    assert batch.led_to_continues[:, -1].tolist() == [0.0, 0.0]
    assert batch.outcome_valid[:, 0].tolist() == [False, False]
    assert batch.outcome_valid[:, 1:].all()


def test_categorical_bc_head_has_finite_gradients():
    torch.manual_seed(97)
    policy = CartPoleBCPolicy(d_model=16, n_actions=2)
    agent = torch.randn(3, 4, 2, 16)
    logits = policy(agent)
    assert logits.shape == (3, 4, 2)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 2),
        torch.randint(0, 2, (12,)),
    )
    loss.backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )


def test_paired_bootstrap_is_seeded():
    first = paired_bootstrap_interval([1.0, 2.0, 3.0], seed=101, draws=500)
    second = paired_bootstrap_interval([1.0, 2.0, 3.0], seed=101, draws=500)
    assert first == second
    assert first[0] > 0.0
