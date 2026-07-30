"""Regression tests for the source-fidelity fixes from the 2026-07-30 audit.

Each test pins one defect recorded in ``spec/bugs/`` so it cannot silently
return. Test names carry the audit finding id.
"""
from dataclasses import replace
import importlib.util
import sys

import pytest
import torch

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
def test_n4_source_report_is_config_conditional():
    assert source_names_for(craftax_jepa_config("transformer")) == (
        "mmbench2_model", "craftax",
    )
    assert source_names_for(craftax_jepa_config("mamba2")) == (
        "mmbench2_model", "craftax", "mamba2",
    )
    sigreg = replace(craftax_jepa_config("mamba2"), jepa_anticollapse="sigreg")
    assert "lejepa" in source_names_for(sigreg)
    # A transformer world records no Mamba source, so reloading it must not
    # require an installed byte-matching Mamba.
    report = source_report(craftax_jepa_config("transformer"))
    assert "mamba2" not in report and "gymnasium_cartpole" not in report
    assert "craftax" in report


def test_n4_no_argument_report_keeps_the_legacy_triple():
    assert set(source_report()) == {
        "mmbench2_model", "mamba2", "gymnasium_cartpole",
    }


def test_n4_verify_recorded_sources_accepts_a_legacy_subset():
    """A checkpoint written before Craftax/LeJEPA were reported must still load."""
    legacy = source_report()  # the historical three-source block
    verify_recorded_sources(legacy)
    verify_recorded_sources({"mmbench2_model": legacy["mmbench2_model"]})


def test_n4_verify_recorded_sources_rejects_drift_and_unknown_names():
    legacy = source_report()
    tampered = {
        "mmbench2_model": {**legacy["mmbench2_model"], "sha256": "0" * 64}
    }
    with pytest.raises(SourceDriftError, match="drifted"):
        verify_recorded_sources(tampered)
    with pytest.raises(SourceDriftError, match="unknown recorded source"):
        verify_recorded_sources({"not_a_source": {}})


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
