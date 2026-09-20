from dataclasses import replace
import json

import pytest
import torch

from d4mj.agent import Heads
from d4mj.checkpoint import read_lewm_actor, read_lewm_bridge
from d4mj.data import (Episode, EpisodeCorpus, sample_bridge_batch,
                       sample_bridge_terminals)
from d4mj.gates import contract_digest, require_actor_gate, require_bridge_gate
from d4mj.lewm_config import AgentSettings
from d4mj.train import bridge_losses, set_phase_mode, train_actor_lewm, train_bridge
from d4mj.world_api import ModelBundle
from d4mj.tests.test_lewm import small_config


def m4_config():
    settings = AgentSettings(
        horizon=2, horizon_eval=10, bootstrap=10, bins=5, mtp_leads=2,
        batch=4, terminal_batch=2, sequence=4, sequence_long=16,
        long_every=4, burn_in=12, true_start_fraction=0.5,
        h2_steps=1, h16_steps=1, recursive_depth=1, recursive_depth_final=2,
        actor_batch=4, actor_screen_steps=1, actor_steps=2,
        warmup=1, checkpoint_every=1, eval_episodes=2,
    )
    return replace(small_config(), agent=settings)


def latent_corpus():
    episodes = []
    for index in range(6):
        steps = 40
        terminated = torch.zeros(steps, dtype=torch.bool); terminated[-1] = True
        events = torch.zeros(steps, dtype=torch.bool); events[10] = True
        episodes.append(Episode(
            observations=None, actions_taken=torch.arange(steps) % 17,
            rewards=torch.arange(steps).float(), terminated=terminated,
            truncated=torch.zeros(steps, dtype=torch.bool),
            latents=torch.randn(steps + 1, 1, 12,
                                generator=torch.Generator().manual_seed(index)),
            latent_digest="fixture", events=events, split="train",
            episode_id=f"episode-{index}",
        ))
    return EpisodeCorpus(episodes)


def test_bridge_sampler_has_real_prefix_and_exact_strata():
    config, corpus = m4_config(), latent_corpus()
    generator = torch.Generator().manual_seed(4)
    batch = sample_bridge_batch(corpus, generator, config, 0)
    assert batch.main.latents.shape == (4, 4, 1, 12)
    assert int(batch.main.relevant.sum()) == 2
    assert int((batch.starts == 0).sum()) == 2
    assert torch.equal(batch.burn_lengths, batch.starts.clamp(max=12))
    assert all(torch.equal(prefix[-1], batch.main.latents[row, 0])
               for row, prefix in enumerate(batch.burn_latents))
    long = sample_bridge_batch(corpus, generator, config, 3)
    assert long.main.latents.shape[1] == 16
    support = sample_bridge_terminals(corpus, generator, config, 0)
    assert support.main.support is not None and bool(support.main.support.all())
    assert bool(support.main.terminated[:, -1].all())


def test_bridge_burnin_reaches_the_loss_and_has_recursive_gradients():
    config, corpus = m4_config(), latent_corpus()
    generator = torch.Generator().manual_seed(7)
    main = sample_bridge_batch(corpus, generator, config, 0)
    terminal = sample_bridge_terminals(corpus, generator, config, 0)
    bundle = ModelBundle.create(config)
    set_phase_mode(bundle, "bridge")
    heads = Heads(config)
    before = {name: value.clone() for name, value in
              bundle.world.predictor_projector.state_dict().items() if "running_" in name}
    losses = bridge_losses(bundle, heads, main, terminal, depth=1)
    assert set(losses) == {"dynamics", "policy", "reward", "continuation"}
    sum(losses.values()).backward()
    assert float(bundle.world.pair_projection.weight.grad.abs().sum()) > 0
    after = {name: value for name, value in
             bundle.world.predictor_projector.state_dict().items() if "running_" in name}
    assert all(torch.equal(before[name], after[name]) for name in before)


def _report(schema, checkpoint, config, cache, stage, decision, depth, components):
    from d4mj.config import recipe_digest
    from d4mj.data import _sha256
    measured = {}
    for name in components:
        evidence = checkpoint.parent / f"{stage}-{name}.json"
        evidence.write_text(json.dumps({"stage": stage, "component": name}) + "\n")
        measured[name] = {"status": "pass", "metrics": {"fixture": 1.0},
                          "evidence": [{"path": str(evidence.resolve()),
                                        "sha256": _sha256(evidence)}]}
    report = {
        "schema": schema, "checkpoint_sha256": _sha256(checkpoint),
        "recipe_id": recipe_digest(config), "cache_id": contract_digest(cache),
        "stage": stage, "decision": decision, "validated_recursive_depth": depth,
        "components": measured,
    }
    report["report_id"] = contract_digest(report)
    return report


def test_phase_gates_bind_checkpoint_and_require_the_full_measurement(tmp_path):
    config, cache = m4_config(), {"schema": "fixture", "sha256": "cache"}
    checkpoint = tmp_path / "state.pt"; checkpoint.write_bytes(b"one")
    bridge_components = {"source_contract", "recursive_dynamics", "semantic_retention",
                         "action_effects", "outcome_calibration", "observed_bc",
                         "paired_uncertainty"}
    h2 = _report("d4mj_lewm_bridge_gate_v1", checkpoint, config, cache, "h2",
                 "continue_h16", 1, bridge_components)
    assert require_bridge_gate(h2, checkpoint=checkpoint, config=config, cache_contract=cache,
                               stage="h2", minimum_depth=1)["validated_recursive_depth"] == 1
    broken = dict(h2); broken["components"] = dict(h2["components"])
    broken["components"].pop("action_effects")
    broken["report_id"] = contract_digest({k: v for k, v in broken.items() if k != "report_id"})
    with pytest.raises(Exception, match="incomplete"):
        require_bridge_gate(broken, checkpoint=checkpoint, config=config, cache_contract=cache,
                            stage="h2", minimum_depth=1)
    checkpoint.write_bytes(b"two")
    with pytest.raises(Exception, match="identity"):
        require_bridge_gate(h2, checkpoint=checkpoint, config=config, cache_contract=cache,
                            stage="h2", minimum_depth=1)

    actor_components = {"model_validity", "critic_direction", "action_distribution",
                        "paired_uncertainty"}
    actor = _report("d4mj_lewm_actor_gate_v1", checkpoint, config, cache,
                    "actor_screen", "continue_actor", 2, actor_components)
    assert require_actor_gate(actor, checkpoint=checkpoint, config=config,
                              cache_contract=cache)["stage"] == "actor_screen"


def test_midstage_resume_carries_the_original_gate_lineage(tmp_path):
    """A gate binds its boundary checkpoint, not a later recovery snapshot.

    Both long stages write intermediate checkpoints.  Resuming one must carry the already
    accepted boundary report forward rather than trying to rebind that report to new bytes.
    """
    base = m4_config()
    config = replace(base, agent=replace(base.agent, h16_steps=2, actor_steps=3))
    corpus = latent_corpus()
    cache = {"schema": "fixture-cache-v1", "sha256": "fixed"}
    parent = {"capabilities": {"joint_complete": True}}
    parent_path = tmp_path / "joint.pt"
    parent_path.write_bytes(b"joint")
    bridge_components = {"source_contract", "recursive_dynamics", "semantic_retention",
                         "action_effects", "outcome_calibration", "observed_bc",
                         "paired_uncertainty"}
    actor_components = {"model_validity", "critic_direction", "action_distribution",
                        "paired_uncertainty"}

    bundle = ModelBundle.create(config)
    bridge_dir = tmp_path / "bridge"
    train_bridge(corpus, bundle, parent, bridge_dir, parent_path=parent_path,
                 cache_contract=cache, stop_after="h2")
    h2_path = bridge_dir / "step-000001.pt"
    h2 = _report("d4mj_lewm_bridge_gate_v1", h2_path, config, cache, "h2",
                 "continue_h16", 1, bridge_components)
    train_bridge(corpus, bundle, parent, bridge_dir, parent_path=parent_path,
                 cache_contract=cache, stop_after="h16", resume=h2_path, gate_report=h2)

    recovered_bridge = tmp_path / "bridge-recovered"
    resumed_bundle = ModelBundle.create(config)
    train_bridge(corpus, resumed_bundle, parent, recovered_bridge, parent_path=parent_path,
                 cache_contract=cache, stop_after="h16",
                 resume=bridge_dir / "step-000002.pt", gate_report=None)
    bridge_path = recovered_bridge / "step-000003.pt"
    bridge = read_lewm_bridge(bridge_path)
    assert bridge["gate_identity"]["report_id"] == h2["report_id"]
    corrupt_bridge = torch.load(bridge_path, map_location="cpu", weights_only=False)
    bn_name = next(name for name in corrupt_bridge["modules"]["world"]
                   if name.startswith("predictor_projector.") and "running_mean" in name)
    corrupt_bridge["modules"]["world"][bn_name] = (
        corrupt_bridge["modules"]["world"][bn_name] + 1
    )
    corrupt_bridge_path = tmp_path / "corrupt-bridge.pt"
    torch.save(corrupt_bridge, corrupt_bridge_path)
    with pytest.raises(ValueError, match="BN identity"):
        read_lewm_bridge(corrupt_bridge_path)

    h16 = _report("d4mj_lewm_bridge_gate_v1", bridge_path, config, cache, "h16",
                  "authorize_actor", 2, bridge_components)
    actor_bundle = ModelBundle.create(config)
    actor_bundle.encoder.load_state_dict(bridge["modules"]["encoder"])
    actor_bundle.world.load_state_dict(bridge["modules"]["world"])
    heads = Heads(config)
    heads.load_state_dict(bridge["modules"]["heads"])
    actor_dir = tmp_path / "actor"
    train_actor_lewm(corpus, actor_bundle, heads, bridge, actor_dir,
                     bridge_path=bridge_path, cache_contract=cache, bridge_gate=h16,
                     stop_after="screen")
    screen_path = actor_dir / "step-000001.pt"
    actor_gate = _report("d4mj_lewm_actor_gate_v1", screen_path, config, cache,
                         "actor_screen", "continue_actor", 2, actor_components)
    train_actor_lewm(corpus, actor_bundle, heads, bridge, actor_dir,
                     bridge_path=bridge_path, cache_contract=cache, bridge_gate=h16,
                     stop_after="budget", resume=screen_path, actor_gate=actor_gate)

    recovered_actor = tmp_path / "actor-recovered"
    actor_bundle = ModelBundle.create(config)
    actor_bundle.encoder.load_state_dict(bridge["modules"]["encoder"])
    actor_bundle.world.load_state_dict(bridge["modules"]["world"])
    heads = Heads(config)
    heads.load_state_dict(bridge["modules"]["heads"])
    train_actor_lewm(corpus, actor_bundle, heads, bridge, recovered_actor,
                     bridge_path=bridge_path, cache_contract=cache, bridge_gate=h16,
                     stop_after="budget", resume=actor_dir / "step-000002.pt",
                     actor_gate=None)
    actor = read_lewm_actor(recovered_actor / "step-000003.pt")
    assert actor["actor_gate_identity"]["report_id"] == actor_gate["report_id"]
    assert actor["capabilities"]["actor_trained"] is True
    corrupt_actor = torch.load(recovered_actor / "step-000003.pt", map_location="cpu",
                               weights_only=False)
    parameter = next(iter(corrupt_actor["modules"]["encoder"]))
    corrupt_actor["modules"]["encoder"][parameter] = (
        corrupt_actor["modules"]["encoder"][parameter] + 1
    )
    corrupt_actor_path = tmp_path / "corrupt-actor.pt"
    torch.save(corrupt_actor, corrupt_actor_path)
    with pytest.raises(ValueError, match="frozen identity"):
        read_lewm_actor(corrupt_actor_path)
