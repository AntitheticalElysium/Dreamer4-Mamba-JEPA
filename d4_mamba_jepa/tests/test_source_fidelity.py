"""Regression tests for the source-fidelity fixes from the 2026-07-30 audit.

Each test pins one defect recorded in ``spec/bugs/`` so it cannot silently
return. Test names carry the audit finding id.
"""
from dataclasses import replace
import importlib.util
import sys

import pytest
import torch

from d4_mamba_jepa.config import D4LiteConfig
from d4_mamba_jepa.craftax_runners import (
    VALUE_FORMAT,
    craftax_jepa_config,
    load_value_checkpoint,
    save_value_checkpoint,
)
from d4_mamba_jepa.imagination_actor_critic import ValueHead
from d4_mamba_jepa.model import D4LiteWorld
from d4_mamba_jepa.objectives import optimizer_groups, split_no_weight_decay
from d4_mamba_jepa.source import (
    LEGACY_SOURCE_NAMES,
    PROVENANCE_SCHEMA,
    SourceDriftError,
    source_names_for,
    source_report,
    verify_recorded_sources,
)
from d4_mamba_jepa.tests.test_baseline import tiny_config


requires_mamba = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
    reason="official mamba_ssm and CUDA are required",
)


# --- N1: optimizer_groups crashed on the decoder-free JEPA world -------------
def test_n1_optimizer_groups_accepts_decoder_free_jepa_world():
    cfg = tiny_config(representation_objective="jepa", jepa_jumps=1)
    world = D4LiteWorld(cfg)
    assert world.decoder is None, "the JEPA arm must stay non-generative"
    groups = optimizer_groups(world, 1e-4)
    assert groups and all(group["params"] for group in groups)
    torch.optim.AdamW(groups)


def test_n1_optimizer_groups_still_rejects_a_decoder_in_the_optimizer():
    """The guard must not weaken the CDP check it wraps."""
    cfg = tiny_config(representation_objective="cdp")
    world = D4LiteWorld(cfg)
    assert world.decoder is not None
    for parameter in world.decoder.parameters():
        parameter.requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen reconstruction decoder"):
        optimizer_groups(world, 1e-4)


# --- A10: upstream _no_weight_decay markers were ignored ---------------------
def test_a10_split_no_weight_decay_is_empty_for_the_transformer_arm():
    world = D4LiteWorld(craftax_jepa_config("transformer"))
    trainable = [p for p in world.parameters() if p.requires_grad]
    decay, no_decay = split_no_weight_decay(trainable)
    assert no_decay == []
    assert len(decay) == len(trainable)
    # The T arm's optimizer must be byte-identical to the pre-fix one.
    assert [group["name"] for group in optimizer_groups(world, 1e-4)] == [
        "encoder", "main"
    ]


@requires_mamba
def test_a10_mamba_marked_tensors_land_in_a_zero_weight_decay_group():
    world = D4LiteWorld(craftax_jepa_config("mamba2"))
    marked = {
        name for name, p in world.named_parameters()
        if getattr(p, "_no_weight_decay", False)
    }
    # Exactly dt_bias / A_log / D of each replaced temporal module.
    assert marked == {
        f"dynamics.transformer.layers.{layer}.time.mamba.{leaf}"
        for layer in (1, 3)
        for leaf in ("dt_bias", "A_log", "D")
    }
    groups = optimizer_groups(world, 1e-4)
    exempt = [g for g in groups if g.get("weight_decay") == 0.0]
    assert len(exempt) == 1
    assert len(exempt[0]["params"]) == len(marked)
    marked_ids = {
        id(p) for p in world.parameters()
        if getattr(p, "_no_weight_decay", False)
    }
    assert {id(p) for p in exempt[0]["params"]} == marked_ids
    # and no marked tensor leaked into a decaying group
    for group in groups:
        if group.get("weight_decay") == 0.0:
            continue
        assert not marked_ids.intersection(id(p) for p in group["params"])


# --- P7: the A10 fix must not perturb the T arm's SERIALIZED optimizer -------
def _pre_fix_optimizer(world, lr, enc_lr):
    """The optimizer exactly as built at commit 81d3466, before the A10 fix."""
    if enc_lr is None:
        trainable = [p for p in world.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-2)
    encoder = list(world.encoder.parameters())
    ids = {id(p) for p in encoder}
    other = [
        p for p in world.parameters() if p.requires_grad and id(p) not in ids
    ]
    groups = [{"params": other, "lr": lr, "base_lr": lr}]
    if enc_lr > 0.0:
        groups.append({"params": encoder, "lr": enc_lr, "base_lr": enc_lr})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=1e-2)


def _group_metadata(optimizer):
    return [
        {k: v for k, v in group.items() if k != "params"}
        for group in optimizer.state_dict()["param_groups"]
    ]


@pytest.mark.parametrize("encoder_lr", [None, 6e-6])
def test_p7_transformer_optimizer_state_dict_is_unchanged(monkeypatch, encoder_lr):
    """A10 must be a no-op for the T arm down to the serialized param groups.

    Numerical equivalence is not enough: a stray group key changes
    `optimizer.state_dict()`, which is written into every world checkpoint.
    """
    import d4_mamba_jepa.craftax_runners as runners

    captured = {}
    real_adamw = torch.optim.AdamW

    def spy(params, **kwargs):
        optimizer = real_adamw(params, **kwargs)
        captured["live"] = optimizer
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    cfg = craftax_jepa_config("transformer")
    world = D4LiteWorld(cfg)
    monkeypatch.setattr(runners, "D4LiteWorld", lambda _cfg: world)
    replay = _one_episode_replay()
    runners.train_craftax_jepa_world(
        replay=replay, cfg=cfg, world_steps=0, batch_size=2,
        device=torch.device("cpu"), warmup=1, encoder_learning_rate=encoder_lr,
        log_every=0,
    )
    monkeypatch.setattr(torch.optim, "AdamW", real_adamw)
    expected = _pre_fix_optimizer(world, 1e-4, encoder_lr)
    assert _group_metadata(captured["live"]) == _group_metadata(expected)


def _one_episode_replay():
    import numpy as np

    from d4_mamba_jepa.data import Episode, EpisodeReplay

    replay = EpisodeReplay(capacity_steps=10 ** 6)
    rng = np.random.default_rng(0)
    length = 20
    replay.add(Episode(
        obs=rng.integers(0, 255, (length + 1, 3, 64, 64), dtype=np.uint8),
        actions=rng.integers(0, 17, length).astype(np.int64),
        rewards=np.zeros(length, np.float32),
        continues=np.concatenate(
            [np.ones(length - 1, np.float32), np.zeros(1, np.float32)]
        ),
    ))
    return replay


# --- A20: value checkpoints were non-atomic and unpaired ---------------------
def test_a20_value_checkpoint_roundtrips_with_pairing(tmp_path):
    torch.manual_seed(0)
    value = ValueHead(d_model=16, num_bins=11, log_low=-2.0, log_high=2.0)
    path = tmp_path / "value.pt"
    digest = save_value_checkpoint(path, value, world_checkpoint_sha256="abc")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == VALUE_FORMAT
    assert payload["world_checkpoint_sha256"] == "abc"
    loaded, _ = load_value_checkpoint(
        path, expected_sha256=digest, expected_world_sha256="abc",
        device=torch.device("cpu"),
    )
    for (name, a), (_, b) in zip(
        value.state_dict().items(), loaded.state_dict().items()
    ):
        assert torch.equal(a, b), name


def test_a20_value_checkpoint_rejects_a_foreign_world(tmp_path):
    value = ValueHead(d_model=16, num_bins=11)
    path = tmp_path / "value.pt"
    digest = save_value_checkpoint(path, value, world_checkpoint_sha256="abc")
    with pytest.raises(RuntimeError, match="pairing drift"):
        load_value_checkpoint(
            path, expected_sha256=digest, expected_world_sha256="other",
            device=torch.device("cpu"),
        )


# --- N4: source_report was unconditional and load compared whole reports -----
def test_n4_source_report_covers_code_dependencies_only():
    # No environment is inferred: D4LiteConfig carries no simulator identity.
    assert source_names_for(craftax_jepa_config("transformer")) == (
        "mmbench2_model",
    )
    assert source_names_for(D4LiteConfig(n_actions=2)) == ("mmbench2_model",)
    assert source_names_for(craftax_jepa_config("mamba2")) == (
        "mmbench2_model", "mamba2", "mamba_ssm_tree",
    )
    sigreg = replace(craftax_jepa_config("mamba2"), jepa_anticollapse="sigreg")
    assert "lejepa" in source_names_for(sigreg)
    # A transformer world records no Mamba source, so reloading it must not
    # require an installed byte-matching Mamba.
    report = source_report(craftax_jepa_config("transformer"))
    assert "mamba2" not in report and "gymnasium_cartpole" not in report


def test_n4_environment_sources_are_explicit_not_inferred():
    cfg = craftax_jepa_config("transformer")
    assert "craftax" not in source_report(cfg)
    assert "craftax" in source_report(cfg, environment_sources=("craftax",))
    with pytest.raises(SourceDriftError, match="unknown source names"):
        source_report(cfg, environment_sources=("not_a_simulator",))


def test_n4_no_argument_report_keeps_the_legacy_triple():
    assert set(source_report()) == set(LEGACY_SOURCE_NAMES)


# --- P1: schema-less leniency must not become fail-open provenance ----------
def test_p1_schemaless_block_must_be_exactly_the_legacy_triple():
    legacy = source_report()
    verify_recorded_sources(legacy)  # a genuine pre-versioning payload
    for bad in ({}, {"mmbench2_model": legacy["mmbench2_model"]}):
        with pytest.raises(SourceDriftError, match="must record exactly"):
            verify_recorded_sources(bad)


def test_p1_schema2_block_must_cover_every_required_source():
    cfg = craftax_jepa_config("mamba2")
    required = source_names_for(cfg)
    full = source_report(cfg, environment_sources=("craftax",))
    verify_recorded_sources(full, schema=PROVENANCE_SCHEMA, required=required)
    # dropping any single required source must fail, not pass vacuously
    for name in required:
        truncated = {k: v for k, v in full.items() if k != name}
        with pytest.raises(SourceDriftError, match="omits required sources"):
            verify_recorded_sources(
                truncated, schema=PROVENANCE_SCHEMA, required=required
            )
    with pytest.raises(SourceDriftError, match="omits required sources"):
        verify_recorded_sources({}, schema=PROVENANCE_SCHEMA, required=required)


def test_p1_unknown_schema_is_refused():
    with pytest.raises(SourceDriftError, match="unsupported provenance schema"):
        verify_recorded_sources(source_report(), schema=99)


def test_n4_verify_recorded_sources_rejects_drift_and_unknown_names():
    legacy = source_report()
    tampered = {**legacy}
    tampered["mmbench2_model"] = {**legacy["mmbench2_model"], "sha256": "0" * 64}
    with pytest.raises(SourceDriftError, match="drifted"):
        verify_recorded_sources(tampered)
    with pytest.raises(SourceDriftError, match="unknown recorded source"):
        verify_recorded_sources({"not_a_source": {}})


# --- P3: the digest closure must cover what actually executes ---------------
def test_p3_craftax_digests_cover_the_env_factory_and_env_classes():
    from d4_mamba_jepa.source import CRAFTAX_CLASSIC_DIGESTS

    for name in (
        "craftax_env.py",
        "craftax_classic/envs/craftax_pixels_env.py",
        "craftax_classic/envs/craftax_symbolic_env.py",
    ):
        assert name in CRAFTAX_CLASSIC_DIGESTS


@requires_mamba
def test_p3_mamba_tree_digest_covers_the_operator_modules():
    """mamba2.py alone excludes the kernels it dispatches to."""
    import importlib.util
    import pathlib

    from d4_mamba_jepa.source import verify_installed_mamba_tree

    assert verify_installed_mamba_tree()
    root = pathlib.Path(importlib.util.find_spec("mamba_ssm").origin).parent
    covered = {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
    # the active use_mem_eff_path=False path and its neighbours
    for name in (
        "ops/triton/ssd_combined.py",
        "ops/triton/layernorm_gated.py",
        "ops/triton/selective_state_update.py",
    ):
        assert name in covered


# --- N5: LeJEPA imported far more than it verified ---------------------------
def test_n5_lejepa_loads_in_isolation_without_its_package_init():
    pytest.importorskip("torch")
    from d4_mamba_jepa.source import load_lejepa_sigreg

    slicing_cls, epps_cls = load_lejepa_sigreg()
    # Importing `lejepa.multivariate.slicing` by package path would execute
    # three __init__.py files pulling in 17 further un-pinned modules.
    assert not [name for name in sys.modules if name.startswith("lejepa")]
    assert slicing_cls.__module__.startswith("d4_mamba_jepa._pinned_lejepa")
    assert epps_cls.__module__.startswith("d4_mamba_jepa._pinned_lejepa")
    # and the loaded statistic still works
    test = slicing_cls(
        univariate_test=epps_cls(n_points=17), num_slices=32, reduction="mean"
    )
    torch.manual_seed(0)
    gaussian = torch.randn(512, 16)
    assert float(test(gaussian)) < float(test(gaussian * 5.0))
