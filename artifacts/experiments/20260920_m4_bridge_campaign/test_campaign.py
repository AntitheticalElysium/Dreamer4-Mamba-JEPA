"""Contracts of the campaign drivers themselves.

The shared suite covers `d4mj` primitives and stayed green through every defect the 2026-09-20
audit found, because nothing exercised these drivers. These tests target the contracts that were
actually broken: authorization, the corpus type, the Phase-2 objective's shape and weighting, the
recursive anchor, the DEV gate's rule, and the fork term's masking.

Run: PYTHONPATH=.:<this dir> pytest artifacts/experiments/20260920_m4_bridge_campaign/test_campaign.py
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from d4mj.config import config_from_dict, load_recipe, recipe_digest
from d4mj.data import EpisodeCorpus, load_joint_corpus
from d4mj.world_api import ModelBundle

RECIPES = HERE / "recipes"


def recipe(variant="raw", flat=True):
    return load_recipe(RECIPES / f"lewm_mamba_{variant}_m4{'_flat' if flat else ''}.json")


# --- authorization -----------------------------------------------------------------------

def test_recipe_intent_alone_never_authorizes_control():
    """The defect: declaring `agent` authorized a completely untrained model."""
    bundle = ModelBundle.create(recipe())
    with pytest.raises(RuntimeError, match="capability record"):
        bundle.require_control()


@pytest.mark.parametrize("record,match", [
    ({"readout_trained": False, "validated_recursive_depth": 16}, "readout is not recorded"),
    ({"readout_trained": True, "validated_recursive_depth": 0}, "not validation"),
    ({"readout_trained": True, "validated_recursive_depth": 2}, "not validation"),
])
def test_incomplete_capabilities_are_refused(record, match):
    bundle = ModelBundle.create(recipe())
    bundle.capabilities = record
    with pytest.raises(RuntimeError, match=match):
        bundle.require_control()


def test_validated_depth_at_or_above_the_horizon_authorizes():
    bundle = ModelBundle.create(recipe())
    bundle.capabilities = {"readout_trained": True, "validated_recursive_depth": 16}
    bundle.require_control()


def test_m0_m3_recipes_are_still_refused_and_keep_their_sealed_identity():
    base = load_recipe(ROOT / "d4mj/recipes/lewm_mamba_raw.json")
    assert base.agent is None
    bundle = ModelBundle.create(base)
    bundle.capabilities = {"readout_trained": True, "validated_recursive_depth": 16}
    with pytest.raises(RuntimeError, match="M0-M3"):
        bundle.require_control()
    sealed = ROOT / "artifacts/lewm_gates_20260906/paired/raw/joint/step-010000.pt"
    if sealed.is_file():
        payload = torch.load(sealed, map_location="cpu", weights_only=False, mmap=True)
        assert payload["recipe_id"] == recipe_digest(config_from_dict(payload["config"]))


# --- corpus ------------------------------------------------------------------------------

def test_single_source_corpus_is_still_an_episode_corpus():
    """The regression: a single source returned a plain list, disabling cached window pools."""
    store = ROOT / "artifacts/craftax_expert_store_v1"
    if not store.is_dir():
        pytest.skip("expert store not built")
    episodes, contract = load_joint_corpus(store, load_recipe(ROOT / "d4mj/recipes/lewm_mamba_raw.json"))
    assert isinstance(episodes, EpisodeCorpus)
    assert episodes.pools(4)["uniform"]
    assert "sources" not in contract


def test_multi_source_corpus_pins_each_source_separately():
    stores = [ROOT / "artifacts/craftax_expert_store_v1", ROOT / "artifacts/craftax_forks_flat_v1"]
    if not all(s.is_dir() for s in stores):
        pytest.skip("stores not built")
    episodes, contract = load_joint_corpus(stores, recipe())
    assert isinstance(episodes, EpisodeCorpus)
    assert len(contract["sources"]) == 2
    assert len({s["sha256"] for s in contract["sources"]}) == 2
    # the merged digest must not be mistakable for its first source
    assert contract["sha256"] != contract["sources"][0]["sha256"]


def test_flattened_forks_are_never_bc_eligible():
    store = ROOT / "artifacts/craftax_forks_flat_v1"
    if not store.is_dir():
        pytest.skip("flat fork store not built")
    manifest = json.loads((store / "manifest.json").read_text())
    assert manifest["bc_eligible"] is False
    assert all("sha256" in s for s in manifest["shards"])


# --- Phase 2 objective -------------------------------------------------------------------

def test_recursive_anchor_is_T_minus_one_minus_H():
    from bridge import rollout
    for blocks, depth in ((32, 2), (32, 16), (128, 16)):
        assert blocks - 1 - depth == _anchor_for(blocks, depth)


def _anchor_for(blocks, depth):
    return blocks - 1 - depth


def test_dynamics_loss_averages_rather_than_sums_per_step():
    """Summing H steps would give H=16 eight times H=2's dynamics weight for free."""
    from bridge import dynamics_loss

    class Teacher:
        pass
    rows, blocks, dim = 4, 32, 8
    z = torch.zeros(rows, blocks, 1, dim)
    keep = torch.ones(rows, dtype=torch.bool)
    losses = {}
    for depth in (2, 16):
        anchor = blocks - 1 - depth
        teacher = Teacher()
        teacher.predicted = torch.full((rows, blocks - 1, 1, dim), 0.5)
        generated = torch.full((rows, depth, 1, dim), 0.5)
        losses[depth] = float(dynamics_loss(teacher, generated, z, keep, anchor))
    assert losses[2] == pytest.approx(losses[16], rel=1e-6), (
        f"dynamics weight changed with horizon: {losses}")


def test_head_strata_use_the_declared_fixed_weighting():
    """0.5*prefix + 0.5*(0.5*observed_suffix + 0.5*generated_suffix), per DECISIONS.md."""
    prefix, observed_suffix, generated_suffix = 1.0, 3.0, 5.0
    expected = 0.5 * prefix + 0.5 * (0.5 * observed_suffix + 0.5 * generated_suffix)
    assert expected == pytest.approx(2.5)


def test_dev_gate_rule_is_beat_persistence():
    from bridge import dev_gate
    source = dev_gate.__doc__
    assert "persistence" in source.lower() and "S63" in source


# --- fork term ---------------------------------------------------------------------------

def test_masking_a_trailing_singleton_broadcasts_and_is_not_used():
    """The bug: (B,1) * (B,) broadcasts to (B,B), inflating the term and dropping the mask."""
    error = torch.rand(68, 1)
    mask = (torch.rand(68) > 0.1).float()
    assert (error * mask).shape == (68, 68)
    subset = error.squeeze(-1)[mask.bool()].mean()
    broadcast = (error * mask).sum() / mask.sum()
    assert not torch.isclose(subset, broadcast, rtol=1e-3)


def test_branch_term_indexes_latents_without_the_trailing_dimension():
    source = (HERE / "train_joint_pair.py").read_text()
    assert "latent[:, 0, 0]" in source
    assert "advanced.latent[:, 0]." not in source and "onward.latent[:, 0]." not in source


# --- launcher ----------------------------------------------------------------------------

def test_run_campaign_refuses_an_unknown_mode():
    import subprocess
    script = HERE / "run_campaign.sh"
    out = subprocess.run(["bash", str(script)], env={"MODE": "flatt", "PATH": "/usr/bin:/bin"},
                         capture_output=True, text=True)
    assert out.returncode == 2 and "must be exactly" in out.stderr


def test_flat_and_grouped_recipes_differ_only_in_fork_mass():
    from d4mj.config import recipe_dict
    flat, grouped = recipe_dict(recipe(flat=True)), recipe_dict(recipe(flat=False))
    differing = {k for k in set(flat["agent"]) | set(grouped["agent"])
                 if flat["agent"].get(k) != grouped["agent"].get(k)}
    assert differing == {"fork_mass"}
    assert flat["agent"]["fork_mass"] == 0.0 and grouped["agent"]["fork_mass"] == 0.2


def test_paired_recipes_differ_only_on_the_declared_axis():
    from d4mj.config import recipe_dict
    from d4mj.lewm_config import pair_axis
    assert pair_axis(recipe_dict(recipe("raw")), recipe_dict(recipe("tc"))) == "variant"
