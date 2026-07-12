"""Predictor architecture contracts.

Ground truth (pinned sources): I-JEPA's predictor is a ViT with self-attention
across tokens and positional embeddings; V-JEPA-2-AC's predictor prepends
action/state tokens to the frame-token sequence and attends jointly. Crafter's
dominant transition moves content BETWEEN token positions (view shift), which a
per-token MLP cannot express — measured as the failed copy-fidelity bar
(reviews/2026-07-12-validation-run-results.md).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import FuturePredictor, ModelConfig  # noqa: E402


def config(**overrides):
    defaults = dict(temporal_backend="gru", predictor="deterministic")
    defaults.update(overrides)
    return ModelConfig(**defaults)


def streams(cfg):
    grid = cfg.image_size // cfg.patch_size
    return cfg.registers + grid * grid


def test_cross_token_information_flow():
    """Perturbing one context token must be able to change the prediction at
    OTHER token positions (content moves between tokens in Crafter)."""
    cfg = config()
    torch.manual_seed(3)
    predictor = FuturePredictor(cfg).eval()
    s = streams(cfg)
    context = torch.randn(1, s, cfg.token_dim)
    action = torch.zeros(1, dtype=torch.long)
    horizon = torch.ones(1, dtype=torch.long)
    with torch.no_grad():
        base, _ = predictor.all_predictions(context, action, horizon)
        perturbed_ctx = context.clone()
        # A constant shift lies in LayerNorm's null space; perturb with a
        # random direction so pre-LN blocks can actually see it.
        perturbed_ctx[0, 10] += 5.0 * torch.randn(cfg.token_dim)
        perturbed, _ = predictor.all_predictions(perturbed_ctx, action, horizon)
    other = [i for i in range(s) if i != 10]
    delta = (base[:, :, other] - perturbed[:, :, other]).abs().max()
    assert float(delta) > 1e-4, "no cross-token pathway in the predictor"


def test_action_changes_all_token_predictions():
    cfg = config()
    torch.manual_seed(4)
    predictor = FuturePredictor(cfg).eval()
    s = streams(cfg)
    context = torch.randn(2, s, cfg.token_dim)
    horizon = torch.ones(2, dtype=torch.long)
    with torch.no_grad():
        a0, _ = predictor.all_predictions(context, torch.zeros(2, dtype=torch.long), horizon)
        a1, _ = predictor.all_predictions(context, torch.ones(2, dtype=torch.long), horizon)
    per_token_change = (a0 - a1).abs().amax(dim=-1)
    assert float(per_token_change.min()) > 1e-6, "some token ignores the action"


def test_predictor_positions_distinguish_identical_tokens():
    """With identical content at every grid position, predictions must still be
    able to differ by position (requires positional information)."""
    cfg = config()
    torch.manual_seed(5)
    predictor = FuturePredictor(cfg).eval()
    s = streams(cfg)
    context = torch.randn(1, 1, cfg.token_dim).expand(1, s, cfg.token_dim).contiguous()
    with torch.no_grad():
        out, _ = predictor.all_predictions(
            context, torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long)
        )
    spread = out[0, 0].std(0).mean()
    assert float(spread) > 1e-5, "predictions are position-agnostic"


def test_single_stream_contexts_still_work():
    """Pooled/synthetic harness variants call the predictor with one stream."""
    cfg = config(token_dim=16, spatial_heads=4)
    predictor = FuturePredictor(cfg)
    context = torch.randn(3, 1, 16)
    out, logits = predictor.all_predictions(
        context, torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.long)
    )
    assert out.shape == (3, 1, 1, 16)
