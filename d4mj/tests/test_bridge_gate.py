"""The G2/G3 evaluator must produce something `require_bridge_gate` accepts, and nothing else."""

import json

import pytest
import torch

from d4mj.agent import Heads
from d4mj.gates import ComponentGateError, contract_digest, require_bridge_gate
from d4mj.lewm_diagnostics import bridge_gate
from d4mj.data import Episode, EpisodeCorpus
from d4mj.tests.test_m4_core import latent_corpus, m4_config


def mixed_corpus(splits=("train", "train", "dev", "dev", "dev", "dev", "final", "final")):
    """A corpus whose splits differ, so split handling is exercised rather than assumed."""
    episodes, steps = [], 40
    for index, split in enumerate(splits):
        # Half of each split dies, so the terminal support stratum and the dead class both exist.
        terminated = torch.zeros(steps, dtype=torch.bool)
        if index % 2 == 0:
            terminated[-1] = True
        episodes.append(Episode(
            observations=None, actions_taken=torch.arange(steps) % 17,
            rewards=torch.arange(steps).float(),
            terminated=terminated,
            truncated=torch.zeros(steps, dtype=torch.bool),
            latents=torch.randn(steps + 1, 1, 12,
                                generator=torch.Generator().manual_seed(100 + index)),
            latent_digest="fixture", events=torch.zeros(steps, dtype=torch.bool),
            split=split, episode_id=f"{split}-{index}"))
    return EpisodeCorpus(episodes)
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


def sealed_screen():
    """The tests exercise the real contract: the boundary accepts only this recipe."""
    from d4mj.config import load_recipe
    from pathlib import Path as _Path
    import d4mj
    return load_recipe(_Path(d4mj.__file__).parent / "recipes" / "joint_screen.json")


def _report(tmp_path):
    config, bundle, heads, checkpoint, cache_contract, payload = _fixture(tmp_path)
    report = bridge_gate(bundle, heads, payload, mixed_corpus(), cache_contract,
                         sealed_screen(), tmp_path / "gate",
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


def _passing(report):
    """A report that passes because its own MEASUREMENTS say so."""
    body = {k: v for k, v in report.items() if k != "report_id"}
    for component in body["components"].values():
        component["status"] = "pass"
        component["metrics"]["failed_checks"] = 0
    body["validated_recursive_depth"] = 2
    return dict(body, report_id=contract_digest(body))


def test_the_boundary_accepts_a_genuinely_passing_report(tmp_path):
    config, checkpoint, cache_contract, report = _report(tmp_path)
    accepted = require_bridge_gate(_passing(report), checkpoint=checkpoint, config=config,
                                   cache_contract=cache_contract, stage="h2", minimum_depth=2)
    assert accepted["validated_recursive_depth"] == 2


def test_a_status_string_cannot_override_its_own_measurements(tmp_path):
    config, checkpoint, cache_contract, report = _report(tmp_path)
    body = {k: v for k, v in report.items() if k != "report_id"}
    for component in body["components"].values():
        component["status"] = "pass"          # measurements left showing failure
        component["metrics"]["failed_checks"] = 2
    body["validated_recursive_depth"] = 2
    sealed = dict(body, report_id=contract_digest(body))
    with pytest.raises(ComponentGateError, match="contradicts its own measurements"):
        require_bridge_gate(sealed, checkpoint=checkpoint, config=config,
                            cache_contract=cache_contract, stage="h2", minimum_depth=2)


def test_a_forged_criterion_no_longer_authorizes(tmp_path):
    """Editing status AND criterion together used to pass: the threshold now lives in code."""
    config, checkpoint, cache_contract, report = _report(tmp_path)
    body = {k: v for k, v in report.items() if k != "report_id"}
    for component in body["components"].values():
        component["status"] = "pass"
        component["metrics"]["failed_checks"] = 3          # honest measurement: it failed
        component["criterion"] = {"quantity": "anything", "value": 1.0,
                                  "threshold": 0.0, "direction": "greater"}   # forged
    body["validated_recursive_depth"] = 2
    sealed = dict(body, report_id=contract_digest(body))
    with pytest.raises(ComponentGateError, match="contradicts its own measurements"):
        require_bridge_gate(sealed, checkpoint=checkpoint, config=config,
                            cache_contract=cache_contract, stage="h2", minimum_depth=2)


def test_a_component_without_measurements_is_refused(tmp_path):
    config, checkpoint, cache_contract, report = _report(tmp_path)
    sealed = _passing(report)
    body = {k: v for k, v in sealed.items() if k != "report_id"}
    body["components"]["source_contract"]["metrics"].pop("failed_checks")
    with pytest.raises(ComponentGateError, match="records no integer failed_checks"):
        require_bridge_gate(dict(body, report_id=contract_digest(body)), checkpoint=checkpoint,
                            config=config, cache_contract=cache_contract, stage="h2",
                            minimum_depth=2)


def test_a_custom_evaluation_recipe_cannot_authorize(tmp_path):
    config, checkpoint, cache_contract, report = _report(tmp_path)
    sealed = _passing(report)
    body = {k: v for k, v in sealed.items() if k != "report_id"}
    body["evaluation"] = dict(body["evaluation"], screen_settings_id="0" * 64)
    with pytest.raises(ComponentGateError, match="evaluation recipe"):
        require_bridge_gate(dict(body, report_id=contract_digest(body)), checkpoint=checkpoint,
                            config=config, cache_contract=cache_contract, stage="h2",
                            minimum_depth=2)


@pytest.mark.parametrize("tamper", ["report_id", "depth", "evidence"])
def test_a_doctored_report_is_refused(tmp_path, tamper):
    config, checkpoint, cache_contract, report = _report(tmp_path)
    sealed = _passing(report)
    body = {k: v for k, v in sealed.items() if k != "report_id"}
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


def test_retention_without_coverage_does_not_pass(tmp_path, monkeypatch):
    """`projection_stop` alone fails open: absent intervals must not read as noninferiority."""
    import d4mj.data as data
    import d4mj.lewm_diagnostics as diag
    from d4mj.lewm_config import ScreenConfig

    unresolved = {"probes": {"linear": {"interval": None}, "mlp": {"interval": None}},
                  "projection_stop": False}
    monkeypatch.setattr(data, "screen_windows", lambda *a, **k: {})
    monkeypatch.setattr(diag, "screen_features", lambda *a, **k: {})
    monkeypatch.setattr(diag, "screen_retention", lambda *a, **k: (dict(unresolved), None))
    config, bundle, heads, checkpoint, cache, payload = _fixture(tmp_path)
    component = diag._semantic_retention(bundle, object(), sealed_screen(), tmp_path / "r")
    # `not projection_stop` would have called this a pass
    assert not unresolved["projection_stop"]
    assert component["status"] == "insufficient_coverage"
    assert component["metrics"]["noninferior_to_cls"] is None


def test_no_raw_corpus_does_not_pass(tmp_path):
    from d4mj.lewm_diagnostics import _semantic_retention
    config, bundle, heads, checkpoint, cache, payload = _fixture(tmp_path)
    component = _semantic_retention(bundle, None, sealed_screen(), tmp_path / "r2")
    assert component["status"] == "insufficient_coverage"


def test_the_gate_samples_dev_only_and_never_final(tmp_path):
    """The gate previously sampled the whole cache: its own training data, and FINAL."""
    from d4mj.lewm_diagnostics import _gate_traces
    config, bundle, heads, checkpoint, cache, payload = _fixture(tmp_path)
    corpus = mixed_corpus()
    _, ledger = _gate_traces(bundle, corpus, config, batches=6, seed=3)
    assert ledger["split"] == "dev"
    chosen = {name for row in ledger["selection"]
              for name, _ in row["main"] + row["terminal"]}
    assert chosen, "the ledger recorded no selection"
    assert all(name.startswith("dev-") for name in chosen), sorted(chosen)
    # and the ledger is exact enough to recompute the selection
    assert all({"update", "main", "terminal"} <= set(row) for row in ledger["selection"])


def test_a_corpus_without_dev_is_refused(tmp_path):
    from d4mj.gates import ComponentGateError
    from d4mj.lewm_diagnostics import _gate_traces
    config, bundle, heads, checkpoint, cache, payload = _fixture(tmp_path)
    with pytest.raises(ComponentGateError, match="no DEV episode"):
        _gate_traces(bundle, mixed_corpus(("train", "train", "final")), config, batches=2, seed=3)


@pytest.mark.parametrize("interval,expected", [
    ([-0.20, +0.01], False),   # permits a 0.20 AUC loss
    ([-0.10, +0.02], False),
    ([-0.01, +0.05], True),    # loss bounded inside the margin
])
def test_noninferiority_uses_the_lower_bound(tmp_path, monkeypatch, interval, expected):
    """Testing the UPPER bound only asks 'could it be fine?'; noninferiority bounds the loss."""
    import d4mj.data as data
    import d4mj.lewm_diagnostics as diag
    resolved = {"probes": {"linear": {"interval": list(interval)},
                           "mlp": {"interval": list(interval)}},
                "projection_stop": False}
    monkeypatch.setattr(data, "screen_windows", lambda *a, **k: {})
    monkeypatch.setattr(diag, "screen_features", lambda *a, **k: {})
    monkeypatch.setattr(diag, "screen_retention", lambda *a, **k: (dict(resolved), None))
    config, bundle, heads, checkpoint, cache, payload = _fixture(tmp_path)
    component = diag._semantic_retention(bundle, object(), sealed_screen(), tmp_path / "n")
    assert (component["status"] == "pass") is expected
    assert component["metrics"]["noninferior_to_cls"] is expected


def test_all_action_forks_are_disjoint_from_training_and_within_root(tmp_path):
    """The fork gate decides WITHIN each root over all actions, on seeds never trained on."""
    from d4mj.lewm_diagnostics import fork_population
    config = m4_config()
    forks = fork_population(config, roots=6, seed=1)
    assert forks["roots"] == 6
    assert forks["successors"].shape[1] == 17, "all actions must be present"
    assert forks["reward"].shape[1] == 17 and forks["terminated"].shape[1] == 17
    seeds = set(forks["seed"].tolist())
    assert seeds and min(seeds) >= 15000, "fork seeds must lie outside the training ranges"
    assert not (seeds & set(range(13000, 14512))), "a sealed M03 evaluation seed reached the gate"


def test_within_root_regret_beats_nothing_when_the_score_is_uninformative():
    """A model that ranks actions at random must not beat the action-marginal choice."""
    from d4mj.lewm_diagnostics import _regret
    torch.manual_seed(0)
    truth = torch.rand(400, 17)
    marginal = truth.mean(0, keepdim=True).expand_as(truth)
    noise = torch.rand(400, 17)
    blind = _regret(truth, marginal, maximize=True).mean()
    random_choice = _regret(truth, noise, maximize=True).mean()
    oracle = _regret(truth, truth, maximize=True).mean()
    assert float(oracle) == pytest.approx(0.0, abs=1e-6), "the oracle has no regret"
    # a random ranker is no better than the marginal choice, within noise
    assert float(random_choice) >= float(blind) - 0.05


def test_the_gate_refuses_when_no_fork_population_is_supplied(tmp_path):
    """State-conditioned consequences cannot be established by global baselines alone."""
    from d4mj.lewm_diagnostics import _action_effects
    from d4mj.lewm_diagnostics import _gate_traces
    config, bundle, heads, checkpoint, cache, payload = _fixture(tmp_path)
    traces, _ = _gate_traces(bundle, mixed_corpus(), config, batches=2, seed=5)
    component = _action_effects(bundle, traces, 1, draws=40, seed=7, output=tmp_path / "ae",
                                heads=heads, prior=None, forks=None)
    assert component["status"] != "pass"
    assert component["metrics"]["all_action"]["status"] == "insufficient_coverage"
