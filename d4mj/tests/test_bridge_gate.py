"""The G2/G3 evaluator must produce something `require_bridge_gate` accepts, and nothing else."""

import json

import pytest
import torch

from d4mj.agent import Heads
from d4mj.gates import ComponentGateError, contract_digest, require_bridge_gate
from d4mj.lewm_diagnostics import bridge_gate
from d4mj.tests.test_m4_core import latent_corpus, m4_config
from d4mj.world_api import ModelBundle


def _fixture(tmp_path):
    config = m4_config()
    bundle = ModelBundle.create(config)
    bundle.encoder.freeze()
    for module in bundle.world.predictor_projector.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
    heads = Heads(config).to(config.runtime.device)
    checkpoint = tmp_path / "step-000001.pt"
    checkpoint.write_bytes(b"bridge-checkpoint-fixture")
    cache_contract = {"path": str(tmp_path / "cache"), "manifest_sha256": "a" * 64}
    payload = {"parent": str(tmp_path / "joint.pt"),
               "capabilities": {"readout_trained": True, "trained_recursive_depth": 2}}
    return config, bundle, heads, checkpoint, cache_contract, payload


def _report(tmp_path):
    from d4mj.lewm_config import ScreenConfig
    config, bundle, heads, checkpoint, cache_contract, payload = _fixture(tmp_path)
    report = bridge_gate(bundle, heads, payload, latent_corpus(), cache_contract,
                         ScreenConfig(bootstrap_draws=40), tmp_path / "gate",
                         stage="h2", checkpoint=checkpoint, batches=3)
    return config, checkpoint, cache_contract, report


def test_report_is_structurally_complete(tmp_path):
    _, _, _, report = _report(tmp_path)
    assert report["schema"] == "d4mj_lewm_bridge_gate_v1"
    assert report["stage"] == "h2" and report["decision"] == "continue_h16"
    required = {"source_contract", "recursive_dynamics", "semantic_retention", "action_effects",
                "outcome_calibration", "observed_bc", "paired_uncertainty"}
    assert required == set(report["components"])
    for name, component in report["components"].items():
        assert component["metrics"], f"{name} recorded no measurements"
        assert component["evidence"], f"{name} bound no evidence"
    # report_id must seal every other field
    body = {k: v for k, v in report.items() if k != "report_id"}
    assert report["report_id"] == contract_digest(body)


def test_an_untrained_world_does_not_pass(tmp_path):
    """The whole point: a model that has learned nothing must not authorize 8,000 more updates."""
    config, checkpoint, cache_contract, report = _report(tmp_path)
    failing = [n for n, c in report["components"].items() if c["status"] != "pass"]
    assert failing, "a randomly initialized world passed every gate component"
    with pytest.raises(ComponentGateError):
        require_bridge_gate(report, checkpoint=checkpoint, config=config,
                            cache_contract=cache_contract, stage="h2", minimum_depth=1)


def test_the_boundary_accepts_a_genuinely_passing_report(tmp_path):
    config, checkpoint, cache_contract, report = _report(tmp_path)
    body = {k: v for k, v in report.items() if k != "report_id"}
    for component in body["components"].values():
        component["status"] = "pass"
    body["validated_recursive_depth"] = 2
    sealed = dict(body, report_id=contract_digest(body))
    accepted = require_bridge_gate(sealed, checkpoint=checkpoint, config=config,
                                   cache_contract=cache_contract, stage="h2", minimum_depth=2)
    assert accepted["validated_recursive_depth"] == 2


@pytest.mark.parametrize("tamper", ["report_id", "depth", "evidence"])
def test_a_doctored_report_is_refused(tmp_path, tamper):
    config, checkpoint, cache_contract, report = _report(tmp_path)
    body = {k: v for k, v in report.items() if k != "report_id"}
    for component in body["components"].values():
        component["status"] = "pass"
    body["validated_recursive_depth"] = 2
    sealed = dict(body, report_id=contract_digest(body))
    if tamper == "report_id":
        sealed["validated_recursive_depth"] = 16          # changed after sealing
    elif tamper == "depth":
        body["validated_recursive_depth"] = 1             # honestly sealed, but too shallow
        sealed = dict(body, report_id=contract_digest(body))
    else:
        path = sealed["components"]["source_contract"]["evidence"][0]["path"]
        open(path, "a").write(" ")                        # evidence bytes moved
    with pytest.raises(ComponentGateError):
        require_bridge_gate(sealed, checkpoint=checkpoint, config=config,
                            cache_contract=cache_contract, stage="h2", minimum_depth=2)
