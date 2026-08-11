from dataclasses import replace

import torch

from artifacts.ablate_frozen_continuation_heads import (
    _configure_head,
    terminal_objectives,
)
from artifacts.diagnose_s35_multimodality import geometry_metrics
from artifacts.evaluate_matched_counterfactual import ARMS, _interaction_auc, arm_config
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
