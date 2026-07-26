"""Self-contained JAX PPO-RNN expert trainer for Craftax-Classic.

Purpose: train a competent policy whose rolled-out trajectories become the
OFFLINE expert dataset the world model / BC actually need (the forage set only
reaches ``collect_stone``; the deep half of the 22 achievements needs a real
policy). PPO only *generates* the corpus -- the thesis stays offline.

SOURCE (vendored, adapted, pinned):
  MichaelTMatthews/Craftax_Baselines @ 7ce36fa05b84a2c9e758012f1e6da402e1e3a891
  files ``ppo_rnn.py`` + ``wrappers.py`` (MIT). Originally from Chris Lu's
  purejaxrl (https://github.com/luchris429/purejaxrl).

Local changes vs the reference (kept in ONE file, no extra deps):
  - target ``Craftax-Classic-Symbolic-v1`` (22 achievements, matches our 17-action
    64x64 pixel env) instead of full Craftax;
  - dropped wandb / orbax / logz / distrax / chex: a tiny native-JAX categorical
    replaces distrax, ``flax.struct`` replaces chex, and the trained params are
    saved with ``flax.serialization`` (msgpack) instead of orbax;
  - single ``train_expert`` entry + ``load_expert`` / ``expert_action_fn`` so the
    generation pipeline can roll the policy out; a light progress print.

SPEED: the fast path is jit + ``lax.scan`` over the whole run with vectorized
``OptimisticResetVecEnvWrapper`` envs -- this is what makes Craftax PPO fast on a
GPU. NOTE: this machine currently has a CPU-only jaxlib, so it runs on CPU (much
slower). Installing a CUDA jaxlib (``jax[cuda12]``, whose bundled runtime works
with the cu13 driver) is the single biggest speed lever; nothing else changes.
"""
from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax import struct
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import BlockType

REFERENCE = (
    "Craftax_Baselines@7ce36fa05b84a2c9e758012f1e6da402e1e3a891:ppo_rnn.py"
    "+wrappers.py (purejaxrl origin)"
)
ENV_NAME = "Craftax-Classic-Symbolic-v1"


# ---------------------------------------------------------------------------
# Native-JAX categorical (replaces distrax.Categorical). Handles any leading dims.
# ---------------------------------------------------------------------------
def cat_sample(logits, key):
    return jax.random.categorical(key, logits, axis=-1)


def cat_log_prob(logits, actions):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, actions[..., None], axis=-1)[..., 0]


def cat_entropy(logits):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(logp) * logp, axis=-1)


# ---------------------------------------------------------------------------
# Env wrappers (vendored from Craftax_Baselines/wrappers.py; chex removed).
# ---------------------------------------------------------------------------
class GymnaxWrapper:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Track episode returns/lengths; expose ``returned_episode`` in info."""

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        return obs, LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_return = state.episode_returns + reward
        new_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_return * (1 - done),
            episode_lengths=new_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class DeathPenaltyWrapper(GymnaxWrapper):
    """Subtract a penalty on death/lava (CrafterDojo's survival shaping).

    Without it the expert plateaus at short episodes; the penalty pushes it to
    survive long enough to reach the deep crafting chain (the paper's competent
    experts survive ~9000 steps).
    """

    def __init__(self, env, death_penalty: float):
        super().__init__(env)
        self.death_penalty = float(death_penalty)

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        is_lava = (
            env_state.map[env_state.player_position[0], env_state.player_position[1]]
            == BlockType.LAVA.value
        )
        is_dead = env_state.player_health <= 0
        reward = reward - self.death_penalty * (is_lava | is_dead)
        return obs, env_state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """Efficient batched 'optimistic' resets (the fast Craftax vectorization)."""

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)
        assert num_envs % reset_ratio == 0, "reset_ratio must divide num_envs"
        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        self.num_resets = num_envs // reset_ratio
        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        return self.reset_fn(jax.random.split(_rng, self.num_envs), params)

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self.step_fn(
            jax.random.split(_rng, self.num_envs), state, action, params
        )
        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self.reset_fn(
            jax.random.split(_rng, self.num_resets), params
        )
        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)
        being_reset_random = jax.random.choice(
            _rng, jnp.arange(self.num_envs), shape=(self.num_resets,),
            p=done, replace=False,
        )
        being_reset_deterministic = jnp.argsort(done)[-self.num_resets:]
        being_reset = jax.lax.select(
            done.astype(jnp.int32).sum() < self.num_resets,
            being_reset_deterministic, being_reset_random,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))
        obs_re = obs_re[reset_indexes]
        state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            return state, jax.lax.select(done, obs_re, obs_st)

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)
        return obs, state, reward, done, info


# ---------------------------------------------------------------------------
# Recurrent actor-critic (vendored; returns logits instead of a distrax dist).
# ---------------------------------------------------------------------------
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan, variable_broadcast="params", in_axes=0, out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: int
    layer_size: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.relu(nn.Dense(
            self.layer_size, kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0))(obs))
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        a = nn.relu(nn.Dense(self.layer_size, kernel_init=orthogonal(2),
                             bias_init=constant(0.0))(embedding))
        a = nn.relu(nn.Dense(self.layer_size, kernel_init=orthogonal(2),
                             bias_init=constant(0.0))(a))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(a)

        c = nn.relu(nn.Dense(self.layer_size, kernel_init=orthogonal(2),
                             bias_init=constant(0.0))(embedding))
        c = nn.relu(nn.Dense(self.layer_size, kernel_init=orthogonal(2),
                             bias_init=constant(0.0))(c))
        value = nn.Dense(1, kernel_init=orthogonal(1.0),
                         bias_init=constant(0.0))(c)
        return hidden, logits, jnp.squeeze(value, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any


def default_config() -> dict:
    return {
        "ENV_NAME": ENV_NAME,
        "NUM_ENVS": 256,
        "NUM_STEPS": 64,
        "TOTAL_TIMESTEPS": int(3e8),
        "LR": 2e-4,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.8,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "ANNEAL_LR": True,
        "LAYER_SIZE": 512,
        "OPTIMISTIC_RESET_RATIO": 16,
        "DEATH_PENALTY": 10,  # CrafterDojo survival shaping (train_ppo.sh)
        "SEED": 0,
        "PROGRESS_EVERY": 10,
    }


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    if config.get("DEATH_PENALTY", 0):
        env = DeathPenaltyWrapper(env, config["DEATH_PENALTY"])
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(
        env, num_envs=config["NUM_ENVS"],
        reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) \
            / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = ActorCriticRNN(env.action_space(env_params).n, config["LAYER_SIZE"])
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["LAYER_SIZE"])
        params = network.init(_rng, init_hstate, init_x)
        lr = linear_schedule if config["ANNEAL_LR"] else config["LR"]
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["LAYER_SIZE"])

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, ustep = runner_state
                rng, _rng = jax.random.split(rng)
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, logits, value = network.apply(train_state.params, hstate, ac_in)
                action = cat_sample(logits, _rng)
                log_prob = cat_log_prob(logits, action)
                value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
                transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                return (train_state, env_state, obsv, done, hstate, rng, ustep), transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            train_state, env_state, last_obs, last_done, hstate, rng, ustep = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    return (gae, value, done), gae
                _, advantages = jax.lax.scan(
                    _get_advantages, (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, gae, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, logits, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done))
                        log_prob = cat_log_prob(logits, traj_batch.action)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae,
                        ).mean()
                        entropy = cat_entropy(logits).mean()
                        total = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, gae, targets)
                    return train_state.apply_gradients(grads=grads), total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)
                shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])),
                        1, 0),
                    shuffled)
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, init_hstate, traj_batch, advantages, targets, rng), total_loss

            init_hstate = initial_hstate[None, :]
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state, rng = update_state[0], update_state[-1]

            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / jnp.maximum(1.0, traj_batch.info["returned_episode"].sum()),
                {"returned_episode_returns": traj_batch.info["returned_episode_returns"],
                 "returned_episode_lengths": traj_batch.info["returned_episode_lengths"]})

            def _progress(metric, ustep):
                if int(ustep) % config["PROGRESS_EVERY"] == 0:
                    print(f"[ppo] update {int(ustep)}/{config['NUM_UPDATES']} "
                          f"ep_return={float(metric['returned_episode_returns']):.3f} "
                          f"ep_len={float(metric['returned_episode_lengths']):.0f}", flush=True)
            jax.debug.callback(_progress, metric, ustep)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, ustep + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv,
                        jnp.zeros((config["NUM_ENVS"]), dtype=bool), init_hstate, _rng, 0)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metric": metric}

    return train


def train_expert(config: dict | None = None, *, output_path: str | Path) -> dict:
    """Train the PPO expert and save its params (msgpack). Returns a summary."""
    cfg = default_config()
    if config:
        cfg.update({k.upper(): v for k, v in config.items()})
    train = jax.jit(make_train(cfg))
    t0 = time.time()
    out = jax.block_until_ready(train(jax.random.PRNGKey(cfg["SEED"])))
    seconds = time.time() - t0
    params = out["runner_state"][0].params
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(serialization.to_bytes(params))
    final = jax.tree.map(lambda x: float(x[-1]), out["metric"])
    return {
        "reference": REFERENCE,
        "config": {k: v for k, v in cfg.items() if k.isupper()},
        "params_path": str(out_path),
        "seconds": seconds,
        "sps": cfg["TOTAL_TIMESTEPS"] / seconds,
        "final_episode_return": final["returned_episode_returns"],
        "final_episode_length": final["returned_episode_lengths"],
        "backend": jax.default_backend(),
    }


def load_expert(params_path: str | Path, *, obs_dim: int, action_dim: int,
                layer_size: int = 512):
    """Rebuild the network + load params. Returns (network, params, init_carry)."""
    network = ActorCriticRNN(action_dim, layer_size)
    dummy_h = ScannedRNN.initialize_carry(1, layer_size)
    dummy_x = (jnp.zeros((1, 1, obs_dim)), jnp.zeros((1, 1)))
    template = network.init(jax.random.PRNGKey(0), dummy_h, dummy_x)
    params = serialization.from_bytes(template, Path(params_path).read_bytes())
    return network, params, dummy_h


def expert_action_fn(network, params, *, layer_size: int = 512, greedy: bool = False):
    """Return a stateful ``act(hidden, obs, done, key) -> (hidden, action)`` for
    rolling the expert out (one env). ``obs`` is the symbolic observation."""
    @jax.jit
    def act(hidden, obs, done, key):
        ac_in = (obs[np.newaxis, np.newaxis, :], done[np.newaxis, np.newaxis])
        hidden, logits, _ = network.apply(params, hidden, ac_in)
        logits = logits[0, 0]
        action = jnp.argmax(logits) if greedy else jax.random.categorical(key, logits)
        return hidden, action
    return act


def _cli() -> None:
    p = argparse.ArgumentParser(description="Train a Craftax-Classic PPO expert.")
    p.add_argument("--out", required=True)
    p.add_argument("--total-timesteps", type=lambda x: int(float(x)), default=int(3e8))
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--layer-size", type=int, default=512)
    p.add_argument("--death-penalty", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    import json
    summary = train_expert(
        {"TOTAL_TIMESTEPS": args.total_timesteps, "NUM_ENVS": args.num_envs,
         "LAYER_SIZE": args.layer_size, "DEATH_PENALTY": args.death_penalty,
         "SEED": args.seed},
        output_path=args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()


__all__ = [
    "REFERENCE", "ENV_NAME", "default_config", "make_train",
    "train_expert", "load_expert", "expert_action_fn",
    "ActorCriticRNN", "ScannedRNN",
]
