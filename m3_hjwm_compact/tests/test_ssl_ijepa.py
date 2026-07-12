"""Contracts for the faithful same-frame I-JEPA module (ssl_ijepa.py).

Pinned source: facebookresearch/ijepa @ 52c1ae9. Protocol + labelled deviations:
reviews/2026-07-13-step1-protocol.md.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ModelConfig, RepresentationEncoder  # noqa: E402
from ssl_ijepa import (  # noqa: E402
    IJEPAPretrainer,
    sample_ijepa_masks,
    sincos_2d_pos_embed,
)


def config(**overrides):
    defaults = dict(temporal_backend="gru", predictor="deterministic")
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_masks_are_disjoint_uniform_and_seeded():
    g1 = torch.Generator().manual_seed(7)
    ctx, preds = sample_ijepa_masks(8, 8, g1)
    assert ctx.shape[0] == 8 and len(preds) == 4
    union = torch.zeros(8, 64, dtype=torch.bool)
    for block in preds:
        assert block.shape[0] == 8 and block.shape[1] >= 1
        for row, idx in enumerate(block):
            union[row, idx] = True
    for row, idx in enumerate(ctx):
        assert not union[row, idx].any(), "context overlaps a target block"
        assert len(idx) >= 4, "min_keep violated"
    # target blocks cover 15-20% of 64 cells each, before trimming
    sizes = [block.shape[1] for block in preds]
    assert all(4 <= s <= 20 for s in sizes)
    # seeded determinism
    g2 = torch.Generator().manual_seed(7)
    ctx2, preds2 = sample_ijepa_masks(8, 8, g2)
    assert torch.equal(ctx, ctx2)
    assert all(torch.equal(a, b) for a, b in zip(preds, preds2))


def test_token_dropping_with_all_visible_matches_full_forward():
    cfg = config()
    torch.manual_seed(0)
    encoder = RepresentationEncoder(cfg).eval()
    obs = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
    everything = torch.arange(64).expand(2, 64)
    with torch.no_grad():
        full = encoder(obs)
        dropped = encoder(obs, visible_index=everything)
    assert torch.allclose(full, dropped, atol=1e-5)


def test_target_mask_and_visible_index_are_exclusive():
    cfg = config()
    encoder = RepresentationEncoder(cfg)
    obs = torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8)
    with pytest.raises(ValueError):
        encoder(
            obs,
            target_mask=torch.zeros(1, 8, 8, dtype=torch.bool),
            visible_index=torch.arange(4)[None],
        )


def test_position_queries_change_predictions():
    cfg = config()
    torch.manual_seed(1)
    model = IJEPAPretrainer(cfg).eval()
    obs = torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8)
    ctx_idx = torch.arange(0, 32)[None]
    context = model.online_encoder(obs, visible_index=ctx_idx)
    with torch.no_grad():
        a = model.predictor(context, ctx_idx, torch.tensor([[40, 41]]))
        b = model.predictor(context, ctx_idx, torch.tensor([[50, 51]]))
    assert (a - b).abs().max() > 1e-5, "target-position queries are ignored"


def test_loss_backward_reaches_encoder_but_not_target():
    cfg = config()
    torch.manual_seed(2)
    model = IJEPAPretrainer(cfg)
    obs = torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8)
    loss = model.loss(obs, torch.Generator().manual_seed(3))
    loss.backward()
    stem_grads = [p.grad for p in model.online_encoder.stem.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in stem_grads)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())


def test_loss_decreases_on_fixed_batch():
    cfg = config()
    torch.manual_seed(4)
    model = IJEPAPretrainer(cfg)
    optimizer = torch.optim.AdamW(
        list(model.online_encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-3,
    )
    obs = torch.randint(0, 255, (8, 3, 64, 64), dtype=torch.uint8)
    first, last = None, None
    for step in range(30):
        loss = model.loss(obs, torch.Generator().manual_seed(100 + step))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_target()
        if step == 0:
            first = float(loss)
        last = float(loss)
    assert last < first * 0.9, f"loss did not decrease: {first} -> {last}"


def test_sincos_embedding_distinguishes_rows_and_columns():
    pos = sincos_2d_pos_embed(64, 8)
    assert pos.shape == (64, 64)
    distances = torch.cdist(pos, pos)
    assert distances[distances > 0].min() > 1e-3


def _two_frames_differing_only_under_mask(cfg):
    torch.manual_seed(9)
    grid = cfg.image_size // cfg.patch_size
    active = torch.ones(1, grid, grid, dtype=torch.bool)
    active[0, 2:5, 2:5] = False  # masked block
    a = torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8)
    b = a.clone()
    pixel = active.repeat_interleave(8, 1).repeat_interleave(8, 2)
    noise = torch.randint(0, 255, b.shape, dtype=torch.uint8)
    b[:, :, ~pixel[0]] = noise[:, :, ~pixel[0]]
    return a, b, active


def test_sparse_stem_is_leak_free_and_dense_stem_is_not():
    from ssl_ijepa import sparse_stem_tokens

    cfg = config()
    torch.manual_seed(8)
    encoder = RepresentationEncoder(cfg).eval()
    a, b, active = _two_frames_differing_only_under_mask(cfg)
    with torch.no_grad():
        sparse_a = sparse_stem_tokens(encoder, a, active)
        sparse_b = sparse_stem_tokens(encoder, b, active)
        dense_a = encoder.project(encoder.stem(a.float() / 255.0)).flatten(2).transpose(1, 2)
        dense_b = encoder.project(encoder.stem(b.float() / 255.0)).flatten(2).transpose(1, 2)
    visible = active.flatten(1)[0]
    leak_sparse = (sparse_a[:, visible] - sparse_b[:, visible]).abs().max()
    leak_dense = (dense_a[:, visible] - dense_b[:, visible]).abs().max()
    assert float(leak_sparse) < 1e-5, f"sparse path leaks masked content: {leak_sparse}"
    # sensitivity guard: the dense path MUST leak, or this test tests nothing
    assert float(leak_dense) > 1e-3, "dense-path leak vanished; test insensitive"


def test_sparse_stem_matches_dense_when_everything_visible():
    from ssl_ijepa import sparse_stem_tokens

    cfg = config()
    torch.manual_seed(10)
    encoder = RepresentationEncoder(cfg).eval()
    obs = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
    grid = cfg.image_size // cfg.patch_size
    with torch.no_grad():
        sparse = sparse_stem_tokens(
            encoder, obs, torch.ones(2, grid, grid, dtype=torch.bool)
        )
        dense = encoder.project(encoder.stem(obs.float() / 255.0)).flatten(2).transpose(1, 2)
    assert torch.allclose(sparse, dense, atol=1e-4), "all-visible sparse != dense"


def test_leak_free_pretrainer_trains():
    cfg = config()
    torch.manual_seed(11)
    model = IJEPAPretrainer(cfg, leak_free=True)
    obs = torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8)
    loss = model.loss(obs, torch.Generator().manual_seed(12))
    loss.backward()
    stem_grads = [p.grad for p in model.online_encoder.stem.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in stem_grads)
