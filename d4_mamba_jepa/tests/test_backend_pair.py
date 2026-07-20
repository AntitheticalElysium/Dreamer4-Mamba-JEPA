import numpy as np
import pytest
import torch

from d4_mamba_jepa.backend_pair import (
    CUBLAS_WORKSPACE_CONFIG,
    WindowSchedule,
    _gate,
    configure_determinism,
    verify_shared_initialization,
)
from d4_mamba_jepa.data import Episode, EpisodeReplay
from d4_mamba_jepa.model import D4LiteWorld
from d4_mamba_jepa.tests.test_baseline import tiny_config


def _replay() -> EpisodeReplay:
    replay = EpisodeReplay()
    for marker in (11, 22):
        observations = np.full((7, 3, 16, 16), marker, dtype=np.uint8)
        replay.add(
            Episode(
                obs=observations,
                actions=np.arange(6, dtype=np.int64) % 5,
                rewards=np.arange(6, dtype=np.float32),
                continues=np.array([1, 1, 1, 1, 1, 0], np.float32),
            )
        )
    return replay


def test_window_schedule_is_reproducible_hashable_and_preserves_timing():
    replay = _replay()
    first = WindowSchedule.generate(
        replay, updates=3, batch_size=2, sequence_length=4, seed=17
    )
    second = WindowSchedule.generate(
        replay, updates=3, batch_size=2, sequence_length=4, seed=17
    )
    assert np.array_equal(first.entries, second.entries)
    assert first.sha256 == second.sha256
    batch = first.materialize(replay, 0)
    for row, (episode_index, start) in enumerate(first.entries[0]):
        episode = replay.episodes[int(episode_index)]
        start = int(start)
        assert torch.equal(
            batch.led_to_actions[row, 1:],
            torch.from_numpy(episode.actions[start:start + 3]),
        )
        if start:
            assert (
                batch.led_to_actions[row, 0].item()
                == episode.actions[start - 1]
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="official Mamba requires CUDA"
)
def test_backend_pair_has_bit_identical_non_temporal_initialization():
    torch.manual_seed(23)
    transformer = D4LiteWorld(tiny_config(temporal_backend="transformer"))
    torch.manual_seed(23)
    mamba = D4LiteWorld(tiny_config(temporal_backend="mamba2"))
    result = verify_shared_initialization(transformer, mamba)
    assert result["shared_tensors"] > 0
    assert result["transformer_temporal_keys"]
    assert result["mamba_temporal_keys"]


def test_gate_follows_preregistered_conjunction():
    def arm(flow, shuffle, action_advantage, auroc, false_reward):
        return {
            "evaluation": {
                "after": {
                    "uniform": {
                        "raw_losses": {"flow/flow_mse": flow},
                        "paired_action_shuffle": {
                            "shuffled_over_true": shuffle
                        },
                        "one_step_action_conditioning": {
                            "wrong_minus_actual_mse": action_advantage
                        },
                        "generated_k4_reward": {
                            "event_auroc_abs_prediction": auroc,
                            "zero_target_mean_abs_prediction": false_reward,
                        },
                    }
                }
            }
        }

    passing = {
        "T-BASE": arm(1.0, 1.1, 0.1, 0.70, 0.01),
        "M-BASE": arm(1.2, 1.1, 0.1, 0.66, 0.02),
    }
    result = _gate(passing)
    assert result["pass"]
    failing = dict(passing)
    failing["M-BASE"] = arm(1.3, 1.1, 0.1, 0.66, 0.02)
    result = _gate(failing)
    assert not result["pass"]
    assert not result["checks"]["flow_mse_at_most_1_25x_transformer"]


def test_determinism_contract_uses_pinned_official_mamba_helper():
    result = configure_determinism(torch.device("cpu"))
    assert result["torch_deterministic_algorithms"]
    assert result["mamba_deterministic_mode"]
    assert result["cublas_workspace_config"] == CUBLAS_WORKSPACE_CONFIG
    assert result["triton_cache_autotuning"] == "1"
    assert result["source_sha256"] == (
        "cb6e1c30392c11200425c2a23ad9fa3d47f50b556d15e9b0caf79b7d483d6f1d"
    )
