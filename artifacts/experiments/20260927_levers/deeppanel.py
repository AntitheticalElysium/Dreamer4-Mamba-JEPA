"""E10 stage 1. Does action-dependent death survive to depth 16? Label statistics for a deep decision panel.

The judgement blocks carry P(death) at H1 and H2 only (observe.py: the second step is the stored NOOP). An H16 panel
needs, per root, each first action's P(dead by step k), k = 1..16, and a continuation that is IDENTICAL across the 17
branches so any difference between branches is the first action's (collect_multistep_forks.py's rule). Two
continuations are measured here before any panel is built:
  noop      15 NOOP steps (the H2 label's continuation, extended)
  recorded  the BC policy's own next 15 actions from the real episode after the root (open loop; NOOP past the
            episode's end): realistic movement, still identical across branches
Walk: observe.py's collector loop exactly (frozen BC policy, damaging-choice retention with a 32-frame history), fresh
seeds. Per retained root and continuation: 17 first actions x K key sequences (independent of the walk's keys), each a
16-step open-loop rollout in the real simulator (auto_reset off, so death is cumulative); P[a, k] = P(dead by k | a).
Reported per k: opportunity rate (P varies over actions), mean within-root spread (max - min), the FIT-free prior
(always the action with the lowest mean P) and oracle expected safe, and how often the k = 16 argmin differs from the
k = 1 argmin (a first-action decision that only a deep horizon gets right).
Usage: deeppanel.py --seed-start 69000 --seeds 10 --keys 32 --out <dir>  (writes seed-*.pt and summary.json)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "artifacts/eda"))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
H, NOOP = 16, 0


def make_deep():
    """(state, keys [K,H,2], first [17], continuation [H-1]) -> dead-by-k [17, K, H], vmapped over actions and keys."""
    from craftax.craftax_classic.constants import BlockType
    from d4mj.env import _env
    env, params = _env()

    def rollout(state, keys, first, continuation):
        actions = jnp.concatenate([first[None], continuation])

        def body(carry, x):
            s, dead = carry
            key, a = x
            _, nxt, _, _, _ = env.step(key, s, a, params)
            d = (nxt.map[nxt.player_position[0], nxt.player_position[1]] == BlockType.LAVA.value) | (nxt.player_health <= 0)
            dead = dead | d
            return (nxt, dead), dead
        _, dead = jax.lax.scan(body, (state, jnp.bool_(False)), (keys, actions))
        return dead
    per_key = jax.vmap(rollout, in_axes=(None, 0, None, None))
    return jax.jit(jax.vmap(per_key, in_axes=(None, None, 0, None)))


def walk(seed, policy, config, fork, deep, keys):
    """observe.walk's loop and retention; deep labels for every retained root after the episode ends."""
    from collect_broad_forks import HISTORY
    from d4mj.data import patchify
    from d4mj.env import reset, step as env_step
    from d4mj.transition import observe
    encoder, world, heads = policy
    observation, env_state = reset(seed)
    state = None
    incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=config.device)
    world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
    policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
    frames, led, roots = 0, [config.n_actions], []
    for index in range(400):
        frames += 1
        patches = patchify(observation[None, None], config.patch).to(config.device)
        state, agent = observe(world, encoder, state, incoming, patches, world_rng, config)
        logits = heads(agent)["policy"][:, -1, 0]
        chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
        if frames >= HISTORY:
            _, _, _, dead1, _, health1, _ = fork(env_state, jax.random.PRNGKey(seed + index + 1), jnp.arange(17))
            damaging = ((np.asarray(health1) - float(env_state.player_health)) <= -1) | np.asarray(dead1)
            if damaging.any() and not damaging.all():
                roots.append((index, env_state))
        observation, env_state, _, terminated, truncated = env_step(env_state, chosen, seed + index + 1)
        led.append(chosen)
        incoming.fill_(chosen)
        if terminated or truncated:
            break
    out = []
    for index, s in roots:
        split = jax.random.split(jax.random.PRNGKey(2**31 - 1 - seed * 1000 - index), keys * H).reshape(keys, H, 2)
        recorded = (led[index + 2:index + 2 + H - 1] + [NOOP] * H)[:H - 1]
        labels = {}
        for name, cont in (("noop", [NOOP] * (H - 1)), ("recorded", recorded)):
            dead = np.asarray(deep(s, split, jnp.arange(17), jnp.array(cont)))          # [17, K, H]
            labels[name] = torch.from_numpy(dead.mean(1).astype(np.float32))              # [17, H]
        out.append({"seed": seed, "step": index, "bc_action": led[index + 1], "recorded": torch.tensor(recorded),
                    "p_dead_by": labels})
    return out


def summarize(rows):
    res = {"roots": len(rows)}
    for name in ("noop", "recorded"):
        p = torch.stack([r["p_dead_by"][name] for r in rows])                          # [R, 17, H]
        per_k = []
        for k in range(H):
            pk = p[:, :, k]
            opp = pk.amax(1) > pk.amin(1)
            prior = int(pk[opp].mean(0).argmin()) if opp.any() else 0
            best1 = p[:, :, 0].argmin(1)
            per_k.append({"k": k + 1, "opportunity": float(opp.float().mean()),
                          "spread": float((pk.amax(1) - pk.amin(1))[opp].mean()) if opp.any() else 0.0,
                          "mean_p": float(pk.mean()), "prior_action": prior,
                          "prior_safe": float(1 - pk[opp, prior].mean()) if opp.any() else None,
                          "oracle_safe": float(1 - pk[opp].amin(1).mean()) if opp.any() else None,
                          "h1_choice_safe": float(1 - pk[opp].gather(1, best1[opp, None]).mean()) if opp.any() else None,
                          "argmin_differs_from_k1": float((pk.argmin(1) != best1)[opp].float().mean()) if opp.any() else None})
        res[name] = per_k
    return res


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--keys", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from dataclasses import replace
    from collect_broad_forks import load_policy, make_fork
    from d4mj.config import Config
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    base = replace(Config(), n_latents=32)
    config = replace(base, transition="direct", time_mixer="attention")
    policy = load_policy(base)
    fork, deep = make_fork(), make_deep()
    with torch.no_grad():
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            if list(args.out.glob(f"seed-{seed:06d}-r*.pt")):
                continue
            rows = walk(seed, policy, config, fork, deep, args.keys)
            target = args.out / f"seed-{seed:06d}-r{len(rows):04d}.pt"
            torch.save(rows, target.with_suffix(".tmp"))
            target.with_suffix(".tmp").replace(target)
            log(seed=seed, roots=len(rows))
    rows = [r for f in sorted(args.out.glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in ("noop", "recorded"):
        log(continuation=name, **{f"k{r['k']}": (round(r["opportunity"], 3), round(r["spread"], 3), r["argmin_differs_from_k1"] and round(r["argmin_differs_from_k1"], 3))
                                  for r in summary[name] if r["k"] in (1, 2, 4, 8, 16)})


if __name__ == "__main__":
    main()
