from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .agent import Heads
from .config import Config
from .data import patchify
from .env import reset, step as env_step
from .representation import Encoder
from .state import WorldState
from .transition import World, advance, observe


@dataclass(frozen=True)
class OutcomeForks:
    agent: Tensor
    true_reward: Tensor
    true_death: Tensor
    model_reward: Tensor
    model_death: Tensor
    seed: Tensor
    step: Tensor
    observed_reward: Tensor | None = None
    observed_death: Tensor | None = None


@torch.no_grad()
def collect_outcome_forks(
    world: World, encoder: Encoder, prior: Heads, config: Config
) -> OutcomeForks:
    """Execute every action from matched real states on DEV seeds."""
    device = config.device
    world, encoder, prior = world.to(device).eval(), encoder.to(device).eval(), prior.to(device).eval()
    agents, rewards, deaths = [], [], []
    model_rewards, model_deaths, observed_rewards, observed_deaths = [], [], [], []
    seeds, steps = [], []

    def add(seed: int, index: int, env_state, state, agent) -> None:
        true_reward, true_death, observations = [], [], []
        for action in range(config.n_actions):
            observation, _, reward, terminated, _ = env_step(
                env_state, action, seed + index + 1
            )
            observations.append(observation)
            true_reward.append(reward)
            true_death.append(terminated)
        predicted_reward, predicted_death = _predict_actions(
            world, prior, state.world, seed, index, config
        )
        observed_reward, observed_death = _predict_observed_actions(
            world, encoder, prior, state, observations, seed, index, config
        )
        agents.append(agent[0, -1].cpu())
        rewards.append(torch.tensor(true_reward))
        deaths.append(torch.tensor(true_death))
        model_rewards.append(predicted_reward.cpu())
        model_deaths.append(predicted_death.cpu())
        observed_rewards.append(observed_reward.cpu())
        observed_deaths.append(observed_death.cpu())
        seeds.append(seed)
        steps.append(index)

    scheduled = set(config.outcome_gate_steps)
    for seed in config.outcome_gate_seeds:
        observation, env_state = reset(seed)
        state = None
        incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=device)
        world_rng = torch.Generator(device=device).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=device).manual_seed(seed + 2**20)
        added = set()

        for index in range(config.outcome_gate_limit):
            patches = patchify(observation[None, None], config.patch).to(device)
            state, agent = observe(world, encoder, state, incoming, patches, world_rng, config)
            if index in scheduled:
                add(seed, index, env_state, state, agent)
                added.add(index)

            logits = prior(agent)["policy"][:, -1, 0]
            action = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
            previous = env_state
            observation, env_state, _, terminated, truncated = env_step(
                env_state, action, seed + index + 1
            )
            if (terminated or truncated) and index not in added:
                add(seed, index, previous, state, agent)
            incoming.fill_(action)
            if terminated or truncated:
                break

    if not agents:
        raise RuntimeError("counterfactual gate collected no states")
    return OutcomeForks(
        agent=torch.stack(agents),
        true_reward=torch.stack(rewards).float(),
        true_death=torch.stack(deaths).bool(),
        model_reward=torch.stack(model_rewards),
        model_death=torch.stack(model_deaths),
        seed=torch.tensor(seeds),
        step=torch.tensor(steps),
        observed_reward=torch.stack(observed_rewards),
        observed_death=torch.stack(observed_deaths),
    )


@torch.no_grad()
def outcome_metrics(forks: OutcomeForks, policy: Heads, config: Config) -> dict[str, float | int | bool]:
    agent = forks.agent.to(config.device)[:, None]
    probabilities = policy.to(config.device).eval()(agent)["policy"][:, 0, 0].softmax(-1).cpu()
    true_reward, true_death = forks.true_reward, forks.true_death.float()
    model_reward, model_death = forks.model_reward, forks.model_death
    reward_varies = true_reward.amax(1) > true_reward.amin(1)
    death_varies = true_death.amax(1) > true_death.amin(1)

    reward_regret = _choice_regret(true_reward[reward_varies], model_reward[reward_varies])
    marginal_regret = _marginal_regret(true_reward[reward_varies])
    death_truth = true_death[death_varies].flatten()
    death_score = model_death[death_varies].flatten()
    terminal_bce = _binary_loss(death_score, death_truth)
    death_marginal = _action_mean(true_death[death_varies]).flatten()
    marginal_bce = _binary_loss(death_marginal, death_truth)
    terminal_auc = _auc(death_score, death_truth.bool())

    report: dict[str, float | int | bool] = {
        "states": len(true_reward),
        "reward_opportunity_states": int(reward_varies.sum()),
        "terminal_opportunity_states": int(death_varies.sum()),
        "reward_choice_regret": reward_regret,
        "reward_marginal_regret": marginal_regret,
        "terminal_bce": terminal_bce,
        "terminal_marginal_bce": marginal_bce,
        "terminal_auc": terminal_auc,
        "true_reward_under_policy": float((probabilities * true_reward).sum(1).mean()),
        "model_reward_under_policy": float((probabilities * model_reward).sum(1).mean()),
        "true_death_under_policy": float((probabilities * true_death).sum(1).mean()),
        "model_death_under_policy": float((probabilities * model_death).sum(1).mean()),
    }
    if forks.observed_reward is not None and forks.observed_death is not None:
        observed_death = forks.observed_death.float()
        report.update(
            {
                "observed_reward_choice_regret": _choice_regret(
                    true_reward[reward_varies], forks.observed_reward[reward_varies]
                ),
                "observed_terminal_bce": _binary_loss(
                    observed_death[death_varies].flatten(), death_truth
                ),
                "observed_terminal_auc": _auc(
                    observed_death[death_varies].flatten(), death_truth.bool()
                ),
            }
        )
    report["passed"] = bool(
        report["reward_opportunity_states"] >= config.outcome_gate_min_opportunities
        and report["terminal_opportunity_states"] >= config.outcome_gate_min_opportunities
        and reward_regret < marginal_regret
        and terminal_bce < marginal_bce
        and terminal_auc > 0.5
    )
    return report


def actor_safety_metrics(before: dict, after: dict) -> dict[str, float | bool]:
    death_change = float(after["true_death_under_policy"] - before["true_death_under_policy"])
    reward_change = float(after["true_reward_under_policy"] - before["true_reward_under_policy"])
    return {
        "true_death_change": death_change,
        "true_reward_change": reward_change,
        "passed": death_change <= 0.0,
    }


def _predict_actions(
    world: World, heads: Heads, state: WorldState, seed: int, index: int, config: Config
) -> tuple[Tensor, Tensor]:
    rewards, deaths = [], []
    samples = config.outcome_gate_flow_samples if config.transition == "flow" else 1
    for action in range(config.n_actions):
        action_rewards, action_deaths = [], []
        chosen = torch.tensor([[action]], device=config.device)
        for sample in range(samples):
            rng = torch.Generator(device=config.device).manual_seed(
                config.seed + 2**23 + seed * 4099 + index * 17 + sample
            )
            _, agent = advance(world, state, chosen, rng, config)
            readout = heads(agent)
            action_rewards.append(_expect(readout["reward"][:, -1, 0], heads.centers)[0])
            action_deaths.append(1.0 - readout["continuation"][:, -1, 0].sigmoid()[0])
        rewards.append(torch.stack(action_rewards).mean())
        deaths.append(torch.stack(action_deaths).mean())
    return torch.stack(rewards), torch.stack(deaths)


def _predict_observed_actions(
    world: World,
    encoder: Encoder,
    heads: Heads,
    state,
    observations: list[Tensor],
    seed: int,
    index: int,
    config: Config,
) -> tuple[Tensor, Tensor]:
    """Read each simulator successor through the deployed observation path."""
    rewards, deaths = [], []
    samples = config.outcome_gate_flow_samples if config.transition == "flow" else 1
    for action, observation in enumerate(observations):
        action_rewards, action_deaths = [], []
        incoming = torch.tensor([[action]], device=config.device)
        patches = patchify(observation[None, None], config.patch).to(config.device)
        for sample in range(samples):
            rng = torch.Generator(device=config.device).manual_seed(
                config.seed + 2**24 + seed * 4099 + index * 17 + sample
            )
            _, agent = observe(world, encoder, state, incoming, patches, rng, config)
            readout = heads(agent)
            action_rewards.append(_expect(readout["reward"][:, -1, 0], heads.centers)[0])
            action_deaths.append(1.0 - readout["continuation"][:, -1, 0].sigmoid()[0])
        rewards.append(torch.stack(action_rewards).mean())
        deaths.append(torch.stack(action_deaths).mean())
    return torch.stack(rewards), torch.stack(deaths)


def _expect(logits: Tensor, centers: Tensor) -> Tensor:
    mean = (logits.softmax(-1) * centers).sum(-1)
    return mean.sign() * torch.expm1(mean.abs())


def _choice_regret(truth: Tensor, score: Tensor) -> float:
    if not len(truth):
        return float("inf")
    chosen = score.argmax(1, keepdim=True)
    return float((truth.amax(1) - truth.gather(1, chosen).squeeze(1)).mean())


def _marginal_regret(truth: Tensor) -> float:
    if not len(truth):
        return float("inf")
    return _choice_regret(truth, _action_mean(truth))


def _action_mean(truth: Tensor) -> Tensor:
    """Best state-blind action-frequency baseline on the gate states."""
    return truth.mean(0, keepdim=True).expand_as(truth)


def _binary_loss(probability: Tensor, target: Tensor) -> float:
    if not len(target):
        return float("inf")
    return float(F.binary_cross_entropy(probability.clamp(1e-6, 1 - 1e-6), target))


def _auc(score: Tensor, target: Tensor) -> float:
    positive, negative = score[target], score[~target]
    if not len(positive) or not len(negative):
        return 0.5
    comparisons = positive[:, None] - negative[None]
    return float((comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean())
