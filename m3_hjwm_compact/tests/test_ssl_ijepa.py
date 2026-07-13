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
        dense = dense + encoder.pos_embed  # positions are part of the encoder now
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


def test_sigreg_prefers_gaussian_over_collapsed():
    from ssl_ijepa import SIGReg

    torch.manual_seed(13)
    sigreg = SIGReg()
    gaussian = torch.randn(512, 4, 16)
    collapsed = torch.ones(512, 4, 16) + 0.01 * torch.randn(512, 4, 16)
    low_rank = torch.randn(512, 4, 1).expand(512, 4, 16).contiguous()
    g = float(sigreg(gaussian))
    c = float(sigreg(collapsed))
    r = float(sigreg(low_rank))
    assert g < c / 10, f"gaussian {g} should score far below collapsed {c}"
    assert g < r / 10, f"gaussian {g} should score far below rank-1 {r}"


def test_sigreg_gradient_reaches_encoder():
    cfg = config()
    torch.manual_seed(14)
    model = IJEPAPretrainer(cfg, leak_free=True)
    model.sigreg_weight = 0.02
    obs = torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8)
    loss = model.loss(obs, torch.Generator().manual_seed(15))
    loss.backward()
    stem_grads = [p.grad for p in model.online_encoder.stem.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in stem_grads)


def test_target_blocks_are_rectangles_with_batch_shared_size():
    """2026-07-13 consensus correction: official collator samples ONE block size
    per batch; trimming must never break rectangularity."""
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        _, preds = sample_ijepa_masks(16, 8, g)
        sizes = set()
        for block in preds:
            for row in block:
                rows, cols = row // 8, row % 8
                h = int(rows.max() - rows.min() + 1)
                w = int(cols.max() - cols.min() + 1)
                assert len(row) == h * w, f"non-rectangular mask (seed {seed})"
                sizes.add((h, w))
        assert len(sizes) == 1, f"block sizes not batch-shared: {sizes}"


def test_encoder_adds_positions_before_token_dropping():
    """Official I-JEPA: positions inside the encoder, before apply_masks."""
    cfg = config()
    torch.manual_seed(20)
    encoder = RepresentationEncoder(cfg).eval()
    uniform = torch.full((1, 3, 64, 64), 128, dtype=torch.uint8)
    with torch.no_grad():
        tokens = encoder(uniform)
    local = tokens[0, cfg.registers:]
    spread = local.std(0).mean()
    assert float(spread) > 1e-3, "identical patches are position-indistinguishable"
