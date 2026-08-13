from dataclasses import replace

import torch

from artifacts.ablate_frozen_continuation_heads import (
    _configure_head,
    terminal_objectives,
)
from artifacts.ablate_phase1b_consequence_gradient import gradient_preflight
from artifacts.diagnose_s35_multimodality import geometry_metrics
from artifacts.evaluate_matched_counterfactual import ARMS, _interaction_auc, arm_config
from artifacts.evaluate_fatality_direction_delta import (
    classify_fatal_delta,
    delta_metrics,
)
from artifacts.localize_flow_phase1b import _conditioned_agent, _summary
from artifacts.measure_encoder_fatality_fidelity import summarize as fidelity_summary
from artifacts.localize_direct_transition_stages import _resumable_linear_probe
from artifacts.phase1b_diagnostic_common import state_digest, stored_state_digest
from artifacts.phase1b_geometry_common import (
    direct_metric_loss,
    geometry_metrics as phase1b_geometry_metrics,
    precision_from_covariance,
    quadratic_error,
    regularized_precision,
    terminal_pair_rows,
)
from artifacts.train_phase1b_geometry_factorial import (
    admit_terminal_batch,
    combine_strata,
)
from artifacts.train_terminal_diversity_scaling import (
    balanced_terminal_schedule,
    stratified_terminal_ranking,
    terminal_metadata,
    terminal_tail_batch,
)
from d4mj.config import Config


def test_s35_geometry_detects_conditional_mean_collapse():
    centers = torch.tensor([[[-1.0]], [[1.0]]])
    weights = torch.tensor([0.5, 0.5])
    death_rate = torch.tensor([0.0, 1.0])
    direct = torch.tensor([[0.0]])
    flow = centers.clone()

    metrics = geometry_metrics(direct, flow, centers, weights, death_rate)

    assert metrics["mode_count"] == 2
    assert metrics["death_stochastic"]
    assert metrics["death_varies_between_observed_modes"]
    assert metrics["direct_to_mean_mse"] == 0.0
    assert metrics["direct_to_nearest_mode_mse"] == 1.0
    assert metrics["direct_mean_advantage_mse"] == 1.0
    assert metrics["direct_mean_closer_than_any_mode"]
    assert metrics["flow_precision_mse"] == 0.0
    assert metrics["flow_coverage_mse"] == 0.0


def test_matched_evaluator_arm_mapping_is_complete():
    base = replace(Config(), device="cpu")
    mapped = {
        arm: (arm_config(base, arm).transition, arm_config(base, arm).time_mixer)
        for arm in ARMS
    }
    assert mapped == {
        "flow-attention": ("flow", "attention"),
        "flow-mamba": ("flow", "mamba"),
        "direct-attention": ("direct", "attention"),
        "direct-mamba": ("direct", "mamba"),
    }


def test_matched_interaction_is_explicitly_undefined_without_both_labels():
    score = torch.tensor([[0.1, 0.9]])
    target = torch.tensor([[False, True]])
    actions = torch.tensor([[0, 1]])
    assert _interaction_auc(score, target, actions) is None


def test_frozen_head_variants_change_only_the_terminal_domain():
    target = torch.tensor([[[1.0], [0.0]]])
    valid = torch.ones_like(target)
    generated = torch.tensor([[[20.0], [-20.0]]])
    observed = -generated

    losses = terminal_objectives(generated, observed, target, valid)

    assert losses["generated_only"] < 1e-7
    assert losses["observed_only"] > 10.0
    assert torch.equal(
        losses["shared_paired"],
        (losses["generated_only"] + losses["observed_only"]) / 2,
    )


def test_frozen_head_terminal_routing_changes_gradients_not_inputs():
    target = torch.tensor([[[1.0], [0.0]]])
    valid = torch.ones_like(target)
    generated = torch.zeros_like(target, requires_grad=True)
    observed = torch.zeros_like(target, requires_grad=True)
    losses = terminal_objectives(generated, observed, target, valid)

    generated_grad, observed_grad = torch.autograd.grad(
        losses["generated_only"],
        (generated, observed),
        allow_unused=True,
        retain_graph=True,
    )
    assert generated_grad is not None and bool((generated_grad != 0).all())
    assert observed_grad is None

    generated_grad, observed_grad = torch.autograd.grad(
        losses["shared_paired"], (generated, observed)
    )
    assert bool((generated_grad != 0).all())
    assert bool((observed_grad != 0).all())


def test_frozen_head_ablation_cannot_train_actor_or_critic():
    from d4mj.agent import Heads

    heads = Heads(replace(Config(), device="cpu"))
    _configure_head(heads)
    trainable = {
        name for name, parameter in heads.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(("model_body.", "reward.", "continuation."))
        for name in trainable
    )


def test_transition_stage_probe_resumes_completed_state_folds(tmp_path):
    config = replace(Config(), device="cpu", n_actions=2)
    group = torch.arange(6).repeat_interleave(2)
    action = torch.tensor([0, 1]).repeat(6)
    target = ((group + action) % 2).float()
    feature = torch.stack(
        (target * 2 - 1, action.float(), group.float()), dim=1
    )
    checkpoint = tmp_path / "probe.pt"
    contract = {"stage": "test", "feature": "latent"}

    first = _resumable_linear_probe(
        feature,
        target,
        action,
        group,
        config,
        seeds=[17],
        steps=2,
        checkpoint=checkpoint,
        contract=contract,
    )
    stored = torch.load(checkpoint, weights_only=False)
    assert stored["complete"] == list(range(6))

    resumed = _resumable_linear_probe(
        feature,
        target,
        action,
        group,
        config,
        seeds=[17],
        steps=2,
        checkpoint=checkpoint,
        contract=contract,
    )
    assert torch.equal(first, resumed)


def test_consequence_gradient_preflight_routes_only_the_allowed_world(config):
    from d4mj.agent import Heads
    from d4mj.transition import World
    from .conftest import latent_batch

    direct = replace(config, transition="direct")
    batch = latent_batch(
        direct, 1, 4, relevant=[False], support=[True]
    )
    batch.terminated[:, -1] = True
    values = gradient_preflight(World(direct), Heads(direct), batch, direct)

    assert values["allowed"] > 0.0
    assert values["stopped"] == 0.0


def test_flow_sample_summary_keeps_first_mean_and_total_variance():
    first, mean, variance = _summary(
        [torch.tensor([1.0, 3.0]), torch.tensor([3.0, 5.0])]
    )
    assert torch.equal(first, torch.tensor([1.0, 3.0]))
    assert torch.equal(mean, torch.tensor([2.0, 4.0]))
    assert variance == 1.0


def test_flow_manual_conditioning_reproduces_the_observe_path():
    from d4mj.data import patchify
    from d4mj.representation import Encoder, pack
    from d4mj.transition import World, observe

    config = replace(
        Config(),
        device="cpu",
        transition="flow",
        n_latents=4,
        d_bottleneck=4,
        packing=2,
        d_model=32,
        depth=2,
        n_heads=2,
        d_model_encoder=32,
        depth_encoder=2,
        n_heads_encoder=2,
        time_every=1,
        n_register=1,
        n_agent=1,
        mlp_ratio=2.0,
        mamba_headdim=8,
        window=2,
        gradient_checkpointing=False,
    )
    encoder, world = Encoder(config).eval(), World(config).eval()
    current = torch.randint(
        0,
        255,
        (1, 1, config.resolution, config.resolution, config.channels),
        dtype=torch.uint8,
    )
    successor = torch.randint_like(current, 0, 255)
    incoming = torch.full((1, 1), config.n_actions, dtype=torch.long)
    state, _ = observe(
        world,
        encoder,
        None,
        incoming,
        patchify(current, config.patch),
        torch.Generator().manual_seed(1),
        config,
    )
    action = torch.tensor([[3]])
    successor_patches = patchify(successor, config.patch)
    encoded, _, _ = encoder(
        successor_patches,
        state.encoder_memory,
        offset=state.world.step,
    )
    latent = pack(encoded, config)
    manual = _conditioned_agent(
        world,
        state.world,
        action,
        latent,
        config.tau_ctx_index,
        torch.Generator().manual_seed(2),
        config,
    )
    observed, agent = observe(
        world,
        encoder,
        state,
        action,
        successor_patches,
        torch.Generator().manual_seed(2),
        config,
    )
    assert torch.equal(observed.world.latent, latent)
    assert torch.equal(agent, manual)


def test_stored_and_live_state_digests_share_one_contract(config):
    from d4mj.transition import World

    world = World(config)
    assert state_digest(world) == stored_state_digest(world.state_dict())


def test_archive_terminal_pairs_exclude_truncation_and_balance_classes():
    from d4mj.data import Episode

    episode = Episode(
        observations=torch.zeros(5, 1),
        actions_taken=torch.tensor([1, 2, 1, 3]),
        rewards=torch.zeros(4),
        terminated=torch.tensor([False, False, False, True]),
        truncated=torch.tensor([False, True, False, False]),
        latents=torch.arange(20, dtype=torch.float).reshape(5, 2, 2),
        latent_digest="test",
    )
    rows, records = terminal_pair_rows([episode], "expert")

    assert rows["target"].tolist() == [0.0, 1.0]
    assert records[0]["transitions"].tolist() == [2, 3]
    assert records[0]["same_action_safe_match"] is False
    assert torch.equal(rows["feature"][1], episode.latents[4])


def test_regularized_precision_is_positive_and_unit_mean_weight():
    samples = torch.randn(1024, 4, generator=torch.Generator().manual_seed(4))
    samples[:, 0] *= 10
    samples[:, 3] *= 0.1
    precision, report = regularized_precision(samples, 0.1)
    eigenvalues = torch.linalg.eigvalsh(precision.double())

    assert float(eigenvalues.min()) > 0
    assert torch.allclose(precision, precision.T, atol=1e-5, rtol=0.0)
    assert abs(float(eigenvalues.mean()) - 1.0) < 1e-5
    assert report["regularized_condition"] < 91.0


def test_precision_reconstructed_from_covariance_matches_sample_path():
    samples = torch.randn(1024, 4, generator=torch.Generator().manual_seed(14))
    first, report = regularized_precision(samples, 0.01)
    second, rebuilt = precision_from_covariance(report["covariance"], 0.01)

    assert torch.allclose(first, second, atol=2e-5, rtol=2e-5)
    assert rebuilt["regularized_condition"] < 10_000


def test_factorial_terminal_stratum_is_explicit_and_equal_per_sequence(config):
    from d4mj.transition import World, transition_loss
    from .conftest import latent_batch

    direct = replace(config, transition="direct")
    world = World(direct)
    reserved = latent_batch(direct, 1, 6, relevant=[False], support=[True])
    ignored = transition_loss(
        world, reserved, torch.Generator().manual_seed(3), direct
    )
    admitted = transition_loss(
        world,
        admit_terminal_batch(reserved),
        torch.Generator().manual_seed(3),
        direct,
    )

    assert ignored == 0
    assert admitted > 0
    observed = combine_strata(torch.tensor(2.0), torch.tensor(10.0), 1 / 5)
    assert observed == torch.tensor(3.6)


def test_fatality_delta_distinguishes_status_quo_and_wrong_direction():
    current = torch.zeros(4, 1, 2)
    target = current.clone()
    target[[1, 3], 0, 0] = 1.0
    label = torch.tensor([0, 1, 0, 1])
    action = torch.zeros(4, dtype=torch.long)
    group = torch.tensor([0, 0, 1, 1])
    means = torch.full((17, 2), 3.0)
    direction = torch.tensor([1.0, 0.0])

    stationary = delta_metrics(
        current,
        target,
        current,
        label,
        action,
        group,
        direction,
        means,
        bootstraps=20,
        seed=4,
    )
    opposite = current.clone()
    opposite[[1, 3], 0, 0] = -1.0
    wrong = delta_metrics(
        current,
        target,
        opposite,
        label,
        action,
        group,
        direction,
        means,
        bootstraps=20,
        seed=4,
    )

    assert stationary["true_delta"]["fatal"]["mean"] == 1.0
    assert stationary["predicted_delta"]["fatal"]["mean"] == 0.0
    assert (
        stationary["fatal_failure_mode"]
        == "predicted_change_not_resolved_from_status_quo"
    )
    assert wrong["fatal_failure_mode"] == "wrong_direction"
    assert classify_fatal_delta(1.0, 0.2, [0.8, 1.2], [0.1, 0.3]) == "tracks_true_direction"


def test_terminal_diversity_subsets_are_nested_and_schedules_balanced(config):
    from d4mj.data import Episode

    direct = replace(config, transition="direct")
    episodes = []
    for index in range(12):
        steps = direct.sequence_long + 4
        terminated = torch.zeros(steps, dtype=torch.bool)
        terminated[-1] = True
        actions = torch.arange(steps) % direct.n_actions
        actions[-1] = index % 4
        episodes.append(
            Episode(
                observations=torch.zeros(steps + 1, 1),
                actions_taken=actions,
                rewards=torch.zeros(steps),
                terminated=terminated,
                truncated=torch.zeros(steps, dtype=torch.bool),
                latents=torch.randn(
                    steps + 1,
                    direct.n_spatial,
                    direct.d_spatial,
                    generator=torch.Generator().manual_seed(index),
                ),
                latent_digest="test",
                bc_eligible=index % 3 == 0,
            )
        )

    metadata = terminal_metadata(episodes, direct)
    ranking = stratified_terminal_ranking(metadata, seed=12)
    schedule = balanced_terminal_schedule(ranking[:5], draws=103, seed=13)
    counts = torch.bincount(schedule, minlength=len(episodes))
    selected_counts = counts[counts > 0]

    assert len(metadata) == len(ranking) == 12
    assert set(ranking[:5]).issubset(ranking[:9])
    assert int(selected_counts.max() - selected_counts.min()) == 1
    batch = terminal_tail_batch(episodes[ranking[0]], direct, step=99, total=100)
    assert batch.latents.shape[1] == direct.sequence_long
    assert bool(batch.support.all())
    assert bool(batch.terminated[:, -1].all())


def test_quadratic_error_identity_is_ordinary_mean_squared_error():
    predicted = torch.randn(2, 3, 2, 4)
    target = torch.randn_like(predicted)
    observed = quadratic_error(predicted, target, torch.eye(8))
    expected = (predicted - target).pow(2).mean(dim=(2, 3))
    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)


def test_direct_metric_loss_identity_matches_production_objective(config):
    from d4mj.transition import World, transition_loss
    from .conftest import latent_batch

    direct = replace(config, transition="direct")
    world = World(direct)
    batch = latent_batch(direct, 2, 6)
    first = transition_loss(
        world, batch, torch.Generator().manual_seed(7), direct
    )
    second = direct_metric_loss(
        world,
        batch,
        torch.Generator().manual_seed(7),
        direct,
        torch.eye(direct.n_spatial * direct.d_spatial),
    )
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_phase1b_geometry_separates_directional_and_orthogonal_error():
    target = torch.tensor(
        [[[[-1.0, 0.0]]], [[[-0.8, 1.0]]], [[[0.8, 0.0]]], [[[1.0, 1.0]]]]
    )
    predicted = target.clone()
    predicted[..., 0] = 0.0
    data = {
        "predicted": predicted,
        "target": target,
        "label": torch.tensor([0, 0, 1, 1]),
        "action": torch.zeros(4, dtype=torch.long),
        "group": torch.arange(4),
    }
    metrics = phase1b_geometry_metrics(
        data,
        torch.tensor([1.0, 0.0]),
        torch.zeros(2, 2),
        torch.eye(2),
        bootstrap_samples=20,
        bootstrap_seed=9,
    )

    assert metrics["target_separation"]["auc"] == 1.0
    assert metrics["predicted_separation"]["auc"] == 0.5
    assert metrics["direction_mse"] > 0
    assert metrics["orthogonal_mse_per_dimension"] == 0.0


def test_encoder_fidelity_compression_is_a_ratio_of_ratios():
    """Compression must be scale-free in both spaces and matched per action, so a
    uniform latent rescaling shows up and an action-mix difference does not."""
    def bucket(pixel, latent):
        n = len(pixel)
        return {"pixel": pixel, "latent": latent, "along": [1.0] * n}

    death = {0: bucket([8.0], [4.0]), 1: bucket([2.0], [1.0])}
    ordinary = {
        0: bucket([2.0] * 40, [1.0] * 40),   # death is 4x in both spaces
        1: bucket([1.0] * 40, [1.0] * 40),   # death is 2x in pixels, 1x in latents
    }
    summary = fidelity_summary(death, ordinary)
    assert summary["per_action"]["0"]["compression"] == 1.0
    assert summary["per_action"]["1"]["compression"] == 0.5
    assert summary["actions_compared"] == 2

    halved = {a: bucket(v["pixel"], [x / 2 for x in v["latent"]]) for a, v in death.items()}
    rescaled = fidelity_summary(halved, ordinary)
    assert rescaled["weighted_compression"] == summary["weighted_compression"] / 2


def test_encoder_fidelity_skips_actions_without_enough_ordinary_support():
    def bucket(pixel, latent):
        return {"pixel": pixel, "latent": latent, "along": [1.0] * len(pixel)}

    death = {0: bucket([8.0], [4.0]), 1: bucket([8.0], [4.0])}
    ordinary = {0: bucket([2.0] * 40, [1.0] * 40), 1: bucket([2.0] * 3, [1.0] * 3)}
    summary = fidelity_summary(death, ordinary)
    assert summary["actions_compared"] == 1
    assert "1" not in summary["per_action"]
