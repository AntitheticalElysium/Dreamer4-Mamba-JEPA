from dataclasses import replace

import torch

from d4mj.data import patchify, unpatchify
from d4mj.representation import Encoder


def test_patchify_round_trips(config):
    frames = torch.randint(
        0, 255, (1, 3, config.resolution, config.resolution, config.channels),
        generator=torch.Generator().manual_seed(0), dtype=torch.uint8,
    )
    patches = patchify(frames, config.patch)
    assert patches.shape == (1, 3, config.n_patches, config.patch_dim)
    restored = unpatchify(patches, config).permute(0, 1, 3, 4, 2)
    assert torch.allclose(restored, frames.float() / 255.0)


def test_patchify_accepts_a_non_contiguous_view(config):
    """The archive stores CHW and is cropped and permuted lazily, so the frames a
    window hands to `patchify` are a view, not a fresh contiguous tensor."""
    chw = torch.randint(
        0, 255, (4, config.channels, 64, 64),
        generator=torch.Generator().manual_seed(0), dtype=torch.uint8,
    )
    view = chw[:, :, : config.resolution, : config.resolution].permute(0, 2, 3, 1)
    assert not view.is_contiguous()
    assert patchify(view[None], config.patch).shape == (1, 4, config.n_patches, config.patch_dim)


def test_receptive_field_is_one_plus_layers_times_window(config):
    """Stacked sliding windows overlap by one position, so L layers reach
    1 + L(W-1) frames, not L*W."""
    layers = config.depth_encoder // config.time_every
    assert config.receptive_field == 1 + layers * (config.window - 1)


def test_nothing_beyond_the_receptive_field_reaches_a_latent(config):
    torch.manual_seed(0)
    encoder = Encoder(config).eval()
    reach = config.receptive_field
    frames = torch.rand(1, reach + 4, config.n_patches, config.patch_dim,
                        generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        scanned, memory, _ = encoder(frames)
        distant = frames.clone()
        distant[:, : frames.shape[1] - reach] = torch.rand_like(distant[:, : frames.shape[1] - reach])
        bounded, _, _ = encoder(distant)
    assert torch.equal(bounded[:, -1], scanned[:, -1])
    assert all(pair[0].shape[2] <= config.window for pair in memory)


def test_chunked_encoding_matches_one_scan(config):
    """`cache_latents` chunks with memory carried; the cheap path must be the
    faithful one or the cache and deployment disagree on the same frame."""
    torch.manual_seed(0)
    encoder = Encoder(config).eval()
    frames = torch.rand(1, 40, config.n_patches, config.patch_dim,
                        generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        whole, _, _ = encoder(frames)
        chunks, memory = [], None
        for start in range(0, frames.shape[1], 16):
            z, memory, _ = encoder(frames[:, start : start + 16], memory, offset=start)
            chunks.append(z)
    assert torch.allclose(torch.cat(chunks, dim=1), whole, atol=1e-4)


def test_masking_is_off_when_probability_is_zero(config):
    """Z* is defined at mask probability 0, and the cache is written there."""
    torch.manual_seed(0)
    encoder = Encoder(config).eval()
    frames = torch.rand(1, 4, config.n_patches, config.patch_dim,
                        generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        _, _, masked = encoder(frames)
    assert not masked.any()
