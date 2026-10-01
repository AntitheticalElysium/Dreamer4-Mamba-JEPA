"""E10. A deep decision panel: does action-dependent death survive to depth 16, and data to judge worlds on it.

The judgement blocks carry P(death) at H1 and H2 only (observe.py: the second step is the stored NOOP). An H16 panel
needs, per root, each first action's P(dead by step k), k = 1..16, and a continuation that is IDENTICAL across the 17
branches so any difference between branches is the first action's (collect_multistep_forks.py's rule). Continuations:
  noop      15 NOOP steps (the H2 label's continuation, extended)
  recorded  the BC policy's own next 15 actions from the real episode after the root (open loop; NOOP past the
            episode's end): realistic movement, still identical across branches
Walk: observe.py's collector loop exactly (frozen BC policy, damaging-choice retention with a 32-frame history). Per
retained root and continuation: 17 first actions x K key sequences (independent of the walk's keys), each a 16-step
open-loop rollout in the real simulator (auto_reset off, so death is cumulative); P[a, k] = P(dead by k | a).
Modes:
  smoke    (stage 1) fresh seeds, labels only; summary per k: opportunity rate (P varies over actions), mean
           within-root spread, prior / oracle expected safe, how often the k = 16 argmin differs from k = 1's
  replay   (stage 2, fit/dev) the readout-ladder partition's FIT-train or FIT-dev seeds; every stored fork-store root
           (32-frame history) is reproduced and CHECKED (frames within 1, one-step deaths identical, as observe.py
           replay), then labelled
  collect  (stage 2, judgement) fresh seeds, the collector's retention rule
Stage-2 rows add, for the recorded continuation and key sequence 0: the real frames at depths 1, 2, 4, 8, 16 for all 17
branches and that draw's dead-by-k; the root's last 8 frames and actions; the visible state (strata).
Usage: deeppanel.py --mode smoke --seed-start 69000 --seeds 10 --out <dir>
       deeppanel.py --mode replay --split fit|dev --out <dir>
       deeppanel.py --mode collect --seed-start 62000 --seeds 400 --out <dir>
"""
import argparse
import collections
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
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "artifacts/eda")); sys.path.insert(0, str(LADDER))
H, NOOP, DEPTHS = 16, 0, (1, 2, 4, 8, 16)


def make_deep(frames=False):
    """(state, keys [K,H,2], first [17], continuation [H-1]) -> dead-by-k [17, K, H] (and observations
    [17, K, H, 63, 63, 3] when `frames`), vmapped over actions and keys."""
    from craftax.craftax_classic.constants import BlockType
    from d4mj.env import _env
    env, params = _env()

    def rollout(state, keys, first, continuation):
        actions = jnp.concatenate([first[None], continuation])

        def body(carry, x):
            s, dead = carry
            key, a = x
            obs, nxt, _, _, _ = env.step(key, s, a, params)
            d = (nxt.map[nxt.player_position[0], nxt.player_position[1]] == BlockType.LAVA.value) | (nxt.player_health <= 0)
            dead = dead | d
            return (nxt, dead), ((dead, obs) if frames else dead)
        _, out = jax.lax.scan(body, (state, jnp.bool_(False)), (keys, actions))
        return out
    per_key = jax.vmap(rollout, in_axes=(None, 0, None, None))
    return jax.jit(jax.vmap(per_key, in_axes=(None, None, 0, None)))


def walk(seed, policy, config, fork, deep, keys, stored=None, detail=None):
    """observe.walk's loop and retention (or, with `stored`, the stored roots, checked); deep labels for every
    retained root after the episode ends. `detail` = the frames variant of make_deep for stage-2 rows."""
    from collect_broad_forks import HISTORY, frames_of
    from d4mj.data import patchify
    from d4mj.env import reset, step as env_step
    from d4mj.transition import observe
    from observe import state_features
    encoder, world, heads = policy
    observation, env_state = reset(seed)
    state = None
    incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=config.device)
    world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
    policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
    frames, led, roots = collections.deque(maxlen=HISTORY), [config.n_actions], []
    last = max(stored) if stored is not None else 400 - 1
    for index in range(last + 1):
        frames.append(observation.clone())
        patches = patchify(observation[None, None], config.patch).to(config.device)
        state, agent = observe(world, encoder, state, incoming, patches, world_rng, config)
        logits = heads(agent)["policy"][:, -1, 0]
        chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
        if len(frames) >= HISTORY:
            _, _, _, dead1, _, health1, _ = fork(env_state, jax.random.PRNGKey(seed + index + 1), jnp.arange(17))
            if stored is None:
                damaging = ((np.asarray(health1) - float(env_state.player_health)) <= -1) | np.asarray(dead1)
                keep = bool(damaging.any() and not damaging.all())
            else:
                keep = index in stored
                if keep:
                    row = stored[index]
                    drift = int((torch.stack(list(frames)).int() - row["frames"].int()).abs().max())
                    bad = int((np.asarray(dead1) != row["terminated"].numpy()).sum())
                    if drift > 1 or bad:
                        raise SystemExit(f"replay does not reproduce stored root {seed}:{index} (drift {drift}, deaths {bad})")
            if keep:
                extra = None if detail is None else {
                    "frames": torch.stack(list(frames)[-8:]), "led_to_action": torch.tensor(led[-8:]),
                    "visible": torch.from_numpy(state_features(env_state)[0])}
                roots.append((index, env_state, extra))
        observation, env_state, _, terminated, truncated = env_step(env_state, chosen, seed + index + 1)
        led.append(chosen)
        incoming.fill_(chosen)
        if terminated or truncated:
            break
    if stored is not None and len(roots) != len(stored):
        raise SystemExit(f"replay of {seed} reached {len(roots)} of {len(stored)} stored roots")
    out = []
    for index, s, extra in roots:
        split = jax.random.split(jax.random.PRNGKey(2**31 - 1 - seed * 1000 - index), keys * H).reshape(keys, H, 2)
        recorded = (led[index + 2:index + 2 + H - 1] + [NOOP] * H)[:H - 1]
        labels = {}
        for name, cont in (("noop", [NOOP] * (H - 1)), ("recorded", recorded)):
            dead = np.asarray(deep(s, split, jnp.arange(17), jnp.array(cont)))          # [17, K, H]
            labels[name] = torch.from_numpy(dead.mean(1).astype(np.float32))              # [17, H]
        row = {"seed": seed, "step": index, "bc_action": led[index + 1], "recorded": torch.tensor(recorded),
               "p_dead_by": labels}
        if extra is not None:
            dead0, obs0 = detail(s, split[:1], jnp.arange(17), jnp.array(recorded))       # [17,1,H], [17,1,H,63,63,3]
            if not np.array_equal(np.asarray(dead0)[:, 0], dead[:, 0]):                  # `dead` = recorded, all K keys
                raise SystemExit("the frames rollout disagrees with the label rollout's key sequence 0")
            obs0 = np.asarray(obs0)[:, 0][:, [d - 1 for d in DEPTHS]]                   # [17, 5, 63, 63, 3]
            row |= extra | {"depth_frames": frames_of(obs0), "depth_dead": torch.from_numpy(np.asarray(dead0)[:, 0])}
        out.append(row)
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
    parser.add_argument("--mode", choices=("smoke", "replay", "collect"), default="smoke")
    parser.add_argument("--split", choices=("fit", "dev"))
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--keys", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from dataclasses import replace
    from collect_broad_forks import HISTORY, load_policy, make_fork
    from d4mj.config import Config
    import observe  # noqa: F401  before confirm, or observe's `from replay import` finds artifacts/eda/replay.py
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    base = replace(Config(), n_latents=32)
    config = replace(base, transition="direct", time_mixer="attention")
    policy = load_policy(base)
    fork, deep = make_fork(), make_deep()
    detail = None if args.mode == "smoke" else make_deep(frames=True)
    if args.mode == "replay":
        from confirm import seeds_for
        from d4mj.lewm_diagnostics import FORK_STORE
        partition = json.loads((LADDER / "evidence/root_partition.json").read_text())
        fit_seeds, _ = seeds_for(partition, FORK_STORE)
        order = fit_seeds if args.split == "fit" else sorted(partition["fit_dev"]["seeds"])
        paths = {int(p.stem.split("-")[1]): p for p in FORK_STORE.glob("seed-*.pt")}
    else:
        order = list(range(args.seed_start, args.seed_start + args.seeds))
    with torch.no_grad():
        for seed in order:
            if list(args.out.glob(f"seed-{seed:06d}-r*.pt")):
                continue
            stored = None
            if args.mode == "replay":
                stored = {int(r["step"]): r for r in torch.load(paths[seed], weights_only=False)
                          if len(r["frames"]) >= HISTORY}
            rows = walk(seed, policy, config, fork, deep, args.keys, stored, detail) if stored != {} else []
            target = args.out / f"seed-{seed:06d}-r{len(rows):04d}.pt"
            torch.save(rows, target.with_suffix(".tmp"))
            target.with_suffix(".tmp").replace(target)
            log(seed=seed, roots=len(rows))
    rows = [{"p_dead_by": r["p_dead_by"]} for f in sorted(args.out.glob("seed-*.pt"))      # labels only: the
            for r in torch.load(f, weights_only=False)]                                # frames do not fit in RAM
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in ("noop", "recorded"):
        log(continuation=name, **{f"k{r['k']}": (round(r["opportunity"], 3), round(r["spread"], 3), r["argmin_differs_from_k1"] and round(r["argmin_differs_from_k1"], 3))
                                  for r in summary[name] if r["k"] in (1, 2, 4, 8, 16)})


if __name__ == "__main__":
    main()
