from dataclasses import replace
import copy

import numpy as np
import pytest
import torch

from d4_mamba_jepa.checkpoint import (
    load_checkpoint,
    load_tokenizer_checkpoint,
    restore_optimizer_and_rng,
    save_checkpoint,
    save_tokenizer_checkpoint,
)
from d4_mamba_jepa.model import D4LiteWorld, build_tokenizer
from d4_mamba_jepa.objectives import optimizer_groups
from d4_mamba_jepa.tests.test_baseline import tiny_config
from d4_mamba_jepa.training import WorldLossNormalizer


def test_checkpoint_roundtrip_is_strict_and_hash_pinned(tmp_path):
    torch.manual_seed(67)
    cfg = tiny_config()
    world = D4LiteWorld(cfg)
    normalizer = WorldLossNormalizer()
    optimizer = torch.optim.AdamW(optimizer_groups(world, 1e-4))
    rng = np.random.default_rng(71)
    path = tmp_path / "world.pt"
    digest = save_checkpoint(
        path,
        world=world,
        normalizer=normalizer,
        optimizer=optimizer,
        numpy_rng=rng,
        step=3,
    )
    loaded, loaded_normalizer, payload = load_checkpoint(
        path,
        device=torch.device("cpu"),
        expected_config=cfg,
        expected_sha256=digest,
    )
    assert payload["step"] == 3
    for name, tensor in world.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[name])
    for name, tensor in normalizer.state_dict().items():
        torch.testing.assert_close(tensor, loaded_normalizer.state_dict()[name])
    with pytest.raises(RuntimeError, match="config drift"):
        load_checkpoint(
            path,
            device=torch.device("cpu"),
            expected_config=replace(cfg, n_register=2),
        )


def test_resume_restores_explicit_numpy_and_torch_rng(tmp_path):
    torch.manual_seed(73)
    cfg = tiny_config()
    world = D4LiteWorld(cfg)
    normalizer = WorldLossNormalizer()
    optimizer = torch.optim.AdamW(optimizer_groups(world, 1e-4))
    rng = np.random.default_rng(79)
    path = tmp_path / "world.pt"
    save_checkpoint(
        path,
        world=world,
        normalizer=normalizer,
        optimizer=optimizer,
        numpy_rng=rng,
        step=0,
    )
    _, _, payload = load_checkpoint(path, device=torch.device("cpu"))

    expected_torch_state = payload["rng"]["torch_cpu"].clone()
    expected_numpy_state = copy.deepcopy(payload["rng"]["numpy_generator"])
    torch.rand(10)
    rng.integers(100, size=10)
    restore_optimizer_and_rng(
        payload, optimizer=optimizer, numpy_rng=rng
    )
    assert torch.equal(torch.get_rng_state(), expected_torch_state)
    assert rng.bit_generator.state == expected_numpy_state

    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="numpy_rng is required"):
        restore_optimizer_and_rng(payload, optimizer=optimizer)
    assert torch.equal(torch.get_rng_state(), before)


def test_failed_atomic_save_preserves_existing_checkpoint(tmp_path, monkeypatch):
    cfg = tiny_config()
    world = D4LiteWorld(cfg)
    normalizer = WorldLossNormalizer()
    path = tmp_path / "world.pt"
    original_digest = save_checkpoint(
        path, world=world, normalizer=normalizer, step=0
    )

    def fail_save(*args, **kwargs):
        raise RuntimeError("synthetic save failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="synthetic save failure"):
        save_checkpoint(path, world=world, normalizer=normalizer, step=1)
    from d4_mamba_jepa.checkpoint import file_sha256

    assert file_sha256(path) == original_digest
    assert not list(tmp_path.glob("*.tmp"))


def test_tokenizer_checkpoint_roundtrip(tmp_path):
    cfg = tiny_config()
    tokenizer = build_tokenizer(cfg, training_mask=True)
    path = tmp_path / "tokenizer.pt"
    digest = save_tokenizer_checkpoint(
        path, tokenizer=tokenizer, config=cfg, step=5
    )
    loaded, payload = load_tokenizer_checkpoint(
        path,
        device=torch.device("cpu"),
        expected_config=cfg,
        expected_sha256=digest,
        training_mask=False,
    )
    assert payload["step"] == 5
    for name, tensor in tokenizer.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[name])
    assert loaded.encoder.mae.p_min == 0.0
    assert loaded.encoder.mae.p_max == 0.0


def test_identical_checkpoint_state_has_identical_bytes(tmp_path):
    torch.manual_seed(83)
    world = D4LiteWorld(tiny_config())
    normalizer = WorldLossNormalizer()
    first = save_checkpoint(
        tmp_path / "first.pt",
        world=world,
        normalizer=normalizer,
        step=7,
    )
    second = save_checkpoint(
        tmp_path / "second.pt",
        world=world,
        normalizer=normalizer,
        step=7,
    )
    assert first == second
