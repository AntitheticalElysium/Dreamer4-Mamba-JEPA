from dataclasses import replace

import torch

from d4mj.agent import Heads
from d4mj.counterfactual import OutcomeForks, actor_safety_metrics, outcome_metrics


def forks_for(config, informative: bool) -> OutcomeForks:
    states, actions = 3, config.n_actions
    reward = torch.zeros(states, actions)
    death = torch.zeros(states, actions, dtype=torch.bool)
    for state in range(states):
        reward[state, state + 1] = 1.0
        death[state, state + 4] = True

    if informative:
        model_reward = reward.clone()
        model_death = death.float() * 0.98 + (~death).float() * 0.02
    else:
        model_reward = torch.zeros_like(reward)
        model_death = torch.full_like(reward, death.float().mean())
    return OutcomeForks(
        agent=torch.zeros(states, config.n_agent, config.d_model),
        true_reward=reward,
        true_death=death,
        model_reward=model_reward,
        model_death=model_death,
        seed=torch.arange(states),
        step=torch.zeros(states, dtype=torch.long),
    )


def test_simulator_does_not_reserve_the_training_accelerator():
    from d4mj import env

    assert env.jax.default_backend() == "cpu"


def test_outcome_gate_requires_information_beyond_marginals(config):
    config = replace(config, outcome_gate_min_opportunities=3)
    policy = Heads(config)
    assert outcome_metrics(forks_for(config, True), policy, config)["passed"]
    assert not outcome_metrics(forks_for(config, False), policy, config)["passed"]


def test_static_action_prior_does_not_pass_as_counterfactual_information(config):
    forks = forks_for(config, True)
    static_reward = forks.true_reward.mean(0, keepdim=True).expand_as(forks.true_reward)
    static_death = forks.true_death.float().mean(0, keepdim=True).expand_as(forks.model_death)
    static = OutcomeForks(
        agent=forks.agent,
        true_reward=forks.true_reward,
        true_death=forks.true_death,
        model_reward=static_reward,
        model_death=static_death,
        seed=forks.seed,
        step=forks.step,
    )
    assert not outcome_metrics(static, Heads(config), config)["passed"]


def test_observed_successor_metrics_localise_generation_failure(config):
    config = replace(config, outcome_gate_min_opportunities=3)
    broken = forks_for(config, False)
    observed = forks_for(config, True)
    localised = replace(
        broken,
        observed_reward=observed.model_reward,
        observed_death=observed.model_death,
    )
    metrics = outcome_metrics(localised, Heads(config), config)
    assert not metrics["passed"]
    assert metrics["observed_reward_choice_regret"] == 0.0
    assert metrics["observed_terminal_bce"] < metrics["terminal_bce"]
    assert metrics["observed_terminal_auc"] == 1.0


def test_actor_gate_rejects_increased_counterfactual_death():
    before = {"true_death_under_policy": 0.10, "true_reward_under_policy": 0.2}
    safer = {"true_death_under_policy": 0.08, "true_reward_under_policy": 0.3}
    riskier = {"true_death_under_policy": 0.11, "true_reward_under_policy": 0.4}
    assert actor_safety_metrics(before, safer)["passed"]
    assert not actor_safety_metrics(before, riskier)["passed"]
