import torch

from d4_mamba_jepa.diagnostics import moving_square_batch
from d4_mamba_jepa.model import build_tokenizer
from d4_mamba_jepa.tests.test_baseline import tiny_config
from d4_mamba_jepa.training import (
    tokenizer_full_reconstruction_mse,
    tokenizer_reconstruction_loss,
)


def test_tokenizer_loss_is_upstream_masked_reconstruction_and_backpropagates():
    torch.manual_seed(61)
    cfg = tiny_config(mae_p_min=0.5, mae_p_max=0.5)
    tokenizer = build_tokenizer(cfg, training_mask=True)
    batch = moving_square_batch(cfg, batch_size=2, device="cpu", seed=11)
    loss, metrics = tokenizer_reconstruction_loss(
        tokenizer, batch.observations, patch_size=cfg.patch_size
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert 0.0 < metrics["masked_fraction"].item() < 1.0
    assert any(parameter.grad is not None for parameter in tokenizer.parameters())


def test_full_reconstruction_eval_restores_training_mask_range():
    cfg = tiny_config(mae_p_min=0.2, mae_p_max=0.8)
    tokenizer = build_tokenizer(cfg, training_mask=True)
    batch = moving_square_batch(cfg, batch_size=1, device="cpu", seed=13)
    before = (tokenizer.encoder.mae.p_min, tokenizer.encoder.mae.p_max)
    mse = tokenizer_full_reconstruction_mse(
        tokenizer, batch.observations, patch_size=cfg.patch_size
    )
    after = (tokenizer.encoder.mae.p_min, tokenizer.encoder.mae.p_max)
    assert torch.isfinite(mse)
    assert after == before
