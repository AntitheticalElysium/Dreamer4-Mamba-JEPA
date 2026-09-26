"""D9. A diagnostic store of true futures, with the simulator's own state beside every frame.

Seeds 900,000+ (a range no judgement block uses), walked by the collector's BC policy -- the policy that
produced the support corpus the worlds trained on -- with observe.walk's loop (same policy sampling, same
per-step environment key). Roots: every 20th step from step 8, while 16 more steps exist. Per root:

  context   frames t-3..t and actions t-3..t-1 (the evaluator protocol: 4 observed frames)
  onestep   all 17 actions x K=4 keys from the root state: successor frames + visible/hidden facts + death
  future    the policy's own next 16 actions a_t..a_{t+15}, rolled from the root state under the walk's
            own keys (the factual trajectory, sample 0) and K=4 other keys (samples 1-4): frames +
            visible/hidden facts + death at every step. Samples 1-4 give the conditional spread of the
            future given the FULL root state and the action sequence (the aleatoric floor).
"""
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts/eda"))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
from collect_broad_forks import frames_of, load_policy  # noqa: E402
from d4mj.config import Config  # noqa: E402
from d4mj.data import patchify  # noqa: E402
from d4mj.env import _env, reset, step as env_step  # noqa: E402
from d4mj.transition import observe  # noqa: E402
from observe import state_features  # noqa: E402

OUT = ROOT / "artifacts/eda/diagnosis_futures_v1"
N, K, H, EVERY, FIRST, MAX_STEPS, TARGET = 17, 4, 16, 20, 8, 320, 1000
SEED0 = 900_000


def make_rollers():
    env, params = _env()

    def one(state, key, action):
        obs, nxt, _, _, _ = env.step(key, state, action, params)
        from craftax.craftax_classic.constants import BlockType
        dead = (nxt.map[nxt.player_position[0], nxt.player_position[1]] == BlockType.LAVA.value) | (nxt.player_health <= 0)
        return obs, nxt, dead

    fork = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, None, 0)), in_axes=(None, 0, None)))   # [K, 17]

    def roll(state, actions, key):
        def body(s, xs):
            a, k = xs
            obs, nxt, dead = one(s, k, a)
            return nxt, (obs, nxt, dead)
        return jax.lax.scan(body, state, (actions, jax.random.split(key, H)))[1]
    return fork, jax.jit(jax.vmap(roll, in_axes=(None, None, 0)))


def facts(states, index):
    s = jax.tree_util.tree_map(lambda x: x[index], states)
    visible, hidden, _ = state_features(s)
    return visible, hidden


def walk(seed, policy, config, fork, roll):
    encoder, world, heads = policy
    observation, env_state = reset(seed)
    state = None
    incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=config.device)
    world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
    policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
    frames, states, actions = [], [], []
    for index in range(MAX_STEPS):
        frames.append(observation.clone())
        states.append(env_state)
        patches = patchify(observation[None, None], config.patch).to(config.device)
        state, agent = observe(world, encoder, state, incoming, patches, world_rng, config)
        chosen = int(torch.multinomial(heads(agent)["policy"][:, -1, 0].softmax(-1), 1, generator=policy_rng))
        actions.append(chosen)
        observation, env_state, _, terminated, truncated = env_step(env_state, chosen, seed + index + 1)
        incoming.fill_(chosen)
        if terminated or truncated:
            frames.append(observation.clone()); states.append(env_state)
            break
    else:
        frames.append(observation.clone()); states.append(env_state)
    records = []
    for t in range(FIRST, len(actions) - H + 1, EVERY):
        root = states[t]
        vis0, hid0, _ = state_features(root)
        keys = jax.random.split(jax.random.PRNGKey(3 * 10**7 + seed * 1000 + t), 2 * K)
        obs1, nxt1, dead1 = fork(root, keys[:K], jnp.arange(N))
        seq = jnp.asarray(actions[t:t + H])
        obsH, nxtH, deadH = roll(root, seq, keys[K:])
        one_vis, one_hid = zip(*[facts(nxt1, (k, a)) for k in range(K) for a in range(N)])
        fut_vis = [[state_features(states[t + j + 1])[0] for j in range(H)]]
        fut_hid = [[state_features(states[t + j + 1])[1] for j in range(H)]]
        for k in range(K):
            v, h = zip(*[facts(nxtH, (k, j)) for j in range(H)])
            fut_vis.append(list(v)); fut_hid.append(list(h))
        fut_frames = torch.stack([torch.stack(frames[t + 1:t + 1 + H])] + [frames_of(obsH[k]) for k in range(K)])
        factual_dead = torch.tensor([float(states[t + j + 1].player_health) <= 0 for j in range(H)])
        records.append({
            "seed": seed, "step": t,
            "context": torch.stack(frames[t - 3:t + 1]), "context_actions": torch.tensor(actions[t - 3:t]),
            "root_visible": torch.from_numpy(vis0).half(), "root_hidden": torch.from_numpy(hid0).half(),
            "onestep_frames": frames_of(obs1).view(K, N, *frames[0].shape),
            "onestep_visible": torch.from_numpy(np.stack(one_vis)).half().view(K, N, -1),
            "onestep_hidden": torch.from_numpy(np.stack(one_hid)).half().view(K, N, -1),
            "onestep_dead": torch.from_numpy(np.asarray(dead1)),
            "future_actions": torch.tensor(actions[t:t + H]),
            "future_frames": fut_frames,
            "future_visible": torch.from_numpy(np.stack([np.stack(v) for v in fut_vis])).half(),
            "future_hidden": torch.from_numpy(np.stack([np.stack(h) for h in fut_hid])).half(),
            "future_dead": torch.cat([factual_dead[None], torch.from_numpy(np.asarray(deadH))]),
        })
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base = replace(Config(), n_latents=32)
    config = replace(base, transition="direct", time_mixer="attention")
    policy = load_policy(base)
    fork, roll = make_rollers()
    total = sum(int(p.stem.split("-r")[1]) for p in OUT.glob("seed-*.pt"))
    seed = SEED0
    with torch.no_grad():
        while total < TARGET:
            target = next(OUT.glob(f"seed-{seed:06d}-r*.pt"), None)
            if target is None:
                records = walk(seed, policy, config, fork, roll)
                path = OUT / f"seed-{seed:06d}-r{len(records):04d}.pt"
                torch.save(records, path.with_suffix(".tmp"))
                path.with_suffix(".tmp").replace(path)
                total += len(records)
                print(json.dumps({"seed": seed, "roots": len(records), "total": total,
                                  "seconds": round(time.time() - started, 1)}), flush=True)
            seed += 1


if __name__ == "__main__":
    main()
