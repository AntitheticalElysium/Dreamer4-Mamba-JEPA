from dataclasses import replace
from pathlib import Path

import pytest
import torch

from d4mj.expert import load_archive
from d4mj.representation import Encoder
from d4mj.train import _cache_digest

ARCHIVE = Path("d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt")


def digest_for(config):
    torch.manual_seed(0)
    return _cache_digest(Encoder(config), config)


def test_cache_digest_covers_the_whole_latent_function(config):
    """Weights are not identity. Two encoders with identical parameters but a
    different frame layout produced the same digest, and one cache would have
    loaded against the other."""
    base = digest_for(config)
    assert base != digest_for(replace(config, resolution=21, patch=7))
    assert base != digest_for(replace(config, depth_encoder=4))
    assert base != digest_for(replace(config, window=8))
    assert base != digest_for(replace(config, d_model_encoder=128))


def test_cache_digest_ignores_the_time_mixer(config):
    """The tokenizer is shared and always attention, so a cache must not acquire a
    different identity from the arm that happened to build it."""
    assert digest_for(replace(config, time_mixer="attention")) == digest_for(
        replace(config, time_mixer="mamba")
    )


def test_cache_digest_tracks_the_weights(config):
    torch.manual_seed(0)
    one = Encoder(config)
    torch.manual_seed(1)
    other = Encoder(config)
    assert _cache_digest(one, config) != _cache_digest(other, config)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archived replay not present")
def test_archive_converts_without_materialising(config):
    """Crop and permute are views, so an episode costs nothing until a window
    indexes it and the 8.6 GB file is never resident."""
    episodes = load_archive(ARCHIVE, config, limit=4)
    assert len(episodes) == 4
    for episode in episodes:
        assert episode.observations.shape[1:] == (
            config.resolution,
            config.resolution,
            config.channels,
        )
        assert episode.observations.dtype == torch.uint8
        assert len(episode.observations) == len(episode) + 1
        assert not bool(episode.terminated.any() and episode.truncated.any())


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archived replay not present")
def test_archive_events_come_from_achievements(config):
    """The relevant sampler needs per-step task events, which the archive already
    carries as per-frame cumulative achievements."""
    raw = torch.load(ARCHIVE, weights_only=False, mmap=True)[0]
    episode = load_archive(ARCHIVE, config, limit=1)[0]
    unlocked = raw["achievements"].sum(-1)
    assert torch.equal(episode.events, unlocked[1:] > unlocked[:-1])
    assert bool(episode.events.any())
