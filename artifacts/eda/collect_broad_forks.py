"""One-pass broad hazard-choice collection, with a second step for surviving branches.

The existing broad corpus took five passes -- collect, rescore health, replay histories,
replay successors, encode -- because the first pass kept only outcomes. This keeps the
raw frames the first time it sees a root, so the later replays are unnecessary, and adds
the one thing the corpus lacks: a second successor for every branch that survives its
first action. That is exactly the second generated state `_direct_loss` trains, so it is
a target the production objective can already consume.

Selection reproduces the existing corpus exactly -- damage-CHOICE, not terminal tails:

    damaging[a] = (health_delta[a] <= -1) or terminated[a]
    keep iff damaging.any() and not damaging.all()

Most retained roots have no immediately lethal action. That is the point: the terminal-
tail corpus was 87.8% traps and taught an action-marginal shortcut, and the broad corpus
is 10% fatal and does not.

Everything the original did is preserved because the BC trajectory must be identical:
32-slot checkpoints, `world_rng = seed + 2**21`, `policy_rng = seed + 2**20`, branch key
`seed + index + 1`, led-to-action starting at the BOS token. The second step uses
`seed + index + 2`, matching `collect_multistep_forks`.

Raw and tokenizer-independent, so any Direct/Flow/Mamba variant can re-encode it.

Every root collected here is training data. The evaluation roots live on seeds 14000-14511
and are already sealed, so applying the usual whole-seed 80/10/10 to this corpus as well
would discard a fifth of it for no gain.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

# must precede `import jax`: d4mj.env sets this too, but only when imported first, and
# jax pins its backend at first import. Craftax on the training GPU is refused there.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent

from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import _env, reset, step as env_step
from d4mj.representation import Encoder
from d4mj.transition import World, observe

N_ACTIONS = 17
NOOP = 0
HISTORY = 32
PHASE1A = ROOT / "artifacts/stage_a_terminalfix/phase1a.pt"
PHASE2 = ROOT / "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt"


def load_policy(config):
    """The frozen BC policy that produced the existing corpus.

    Two things make this awkward and both are load-bearing. These checkpoints are 32-slot
    and predate the 64-slot default, so the config must be pinned rather than taken from
    `Config()`. And S85 promoted the action-token mixer into `World`, so the stored
    `World` no longer matches the class -- but the rollout only calls the backbone and the
    policy head, never `world.predict`, which is the only consumer of the missing weights.
    So the world loads non-strictly and the missing keys are asserted to be exactly the
    predict head. Anything else missing would be a silently wrong policy, and a silently
    wrong policy means a different trajectory and a different corpus.
    """
    direct = replace(config, transition="direct", time_mixer="attention")
    encoder = Encoder(config).to(config.device)
    world = World(direct).to(config.device)
    heads = Heads(direct).to(config.device)
    load(PHASE1A, config, part0=encoder)

    payload = torch.load(PHASE2, weights_only=False)
    assert payload["config"]["n_latents"] == config.n_latents, "phase2 slot count differs"
    report = world.load_state_dict(payload["modules"]["part0"], strict=False)
    allowed = ("direct_mixer.", "direct_norm.", "readout.")
    unexpected = [k for k in report.missing_keys if not k.startswith(allowed)]
    assert not unexpected, f"backbone weights missing, policy would be wrong: {unexpected}"
    heads.load_state_dict(payload["modules"]["part1"])
    # `d4mj.checkpoint.load` does this and the original collection went through it, so
    # skipping it leaves the global stream in a different place than the run we must
    # reproduce
    torch.set_rng_state(payload["rng"])

    for module in (encoder, world, heads):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return encoder, world, heads


def make_fork():
    """All 17 successors from one state under one key: frames, health, death, reward,
    achievements, truncation -- and the successor states, so the second step continues
    from them without re-stepping."""
    env, params = _env()

    def one(state, key, action):
        observation, nxt, reward, done, _ = env.step(key, state, action, params)
        from craftax.craftax_classic.constants import BlockType

        lava = nxt.map[nxt.player_position[0], nxt.player_position[1]] == BlockType.LAVA.value
        dead = lava | (nxt.player_health <= 0)
        return (observation, nxt, reward, dead, done & ~dead,
                nxt.player_health, nxt.achievements.sum())

    return jax.jit(jax.vmap(one, in_axes=(None, None, 0)))


def frames_of(observation) -> torch.Tensor:
    return torch.from_numpy(np.asarray(observation) * 255.0).round().to(torch.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=15_000)
    parser.add_argument("--seeds", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=400, help="states examined per seed")
    parser.add_argument("--out", type=Path, default=HERE / "broad_forks_v2")
    parser.add_argument("--target", type=int, default=0, help="stop once this many roots")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # `Config.transition` defaults to "flow", and passing that to `observe` runs
    # `commit_inputs`'s flow branch -- noising the latent and drawing from world_rng --
    # which silently produces a different BC trajectory. The encoder is loaded under the
    # flow config it was written with; everything else uses direct, as the original did.
    base = replace(Config(), n_latents=32)
    config = replace(base, transition="direct", time_mixer="attention")
    encoder, world, heads = load_policy(base)
    fork = make_fork()

    # One file per seed, written atomically, and the file is its own resume marker --
    # a buffered shard plus a separate marker can mark a seed done and then lose its
    # rows. The root count lives in the name so resuming does not have to load 11 GiB
    # to find out how much it already has.
    done = {int(f.stem.split("-")[1]): int(f.stem.split("-r")[1]) for f in
            args.out.glob("seed-*-r*.pt")}
    examined, retained, started = 0, sum(done.values()), time.time()
    if done:
        print(f"resuming: {len(done)} seeds already collected, {retained:,} roots",
              flush=True)

    with torch.no_grad():
        for order, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds)):
            if seed in done:
                continue
            observation, env_state = reset(seed)
            state = None
            incoming = torch.full((1, 1), config.n_actions, dtype=torch.long,
                                  device=config.device)
            world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
            policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
            frames, led, rows = [], [config.n_actions], []

            for index in range(args.limit):
                frames.append(observation.clone())
                patches = patchify(observation[None, None], config.patch).to(config.device)
                state, agent = observe(world, encoder, state, incoming, patches,
                                       world_rng, config)
                logits = heads(agent)["policy"][:, -1, 0]
                chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))

                key = jax.random.PRNGKey(seed + index + 1)
                obs1, state1, reward1, dead1, trunc1, health1, ach1 = fork(
                    env_state, key, jnp.arange(N_ACTIONS))
                delta = np.asarray(health1) - float(env_state.player_health)
                dead = np.asarray(dead1)
                damaging = (delta <= -1) | dead
                examined += 1

                if len(frames) >= HISTORY and damaging.any() and not damaging.all():
                    alive = ~dead & ~np.asarray(trunc1)
                    second = np.zeros((N_ACTIONS,) + frames[0].shape, dtype=np.uint8)
                    s_reward = np.zeros(N_ACTIONS, np.float32)
                    s_delta = np.zeros(N_ACTIONS, np.float32)
                    s_dead = np.zeros(N_ACTIONS, bool)
                    s_trunc = np.zeros(N_ACTIONS, bool)
                    key2 = jax.random.PRNGKey(seed + index + 2)
                    for a in np.where(alive)[0]:
                        branch = jax.tree_util.tree_map(lambda x, i=int(a): x[i], state1)
                        o2, st2, r2, d2, t2, h2, _ = fork(branch, key2, jnp.array([NOOP]))
                        second[a] = frames_of(o2[0]).numpy()
                        s_reward[a] = float(r2[0])
                        s_delta[a] = float(h2[0]) - float(health1[a])
                        s_dead[a] = bool(d2[0])
                        s_trunc[a] = bool(t2[0])
                    rows.append({
                        "seed": seed, "step": index,
                        "frames": torch.stack(frames[-HISTORY:]),
                        "led_to_action": torch.tensor(led[-HISTORY:], dtype=torch.long),
                        "bc_action": chosen,
                        "successors": frames_of(obs1),
                        "reward": torch.from_numpy(np.asarray(reward1, np.float32)),
                        "health_delta": torch.from_numpy(delta.astype(np.float32)),
                        "achievement_delta": torch.from_numpy(
                            (np.asarray(ach1) - int(env_state.achievements.sum())).astype(np.int16)),
                        "terminated": torch.from_numpy(dead),
                        "truncated": torch.from_numpy(np.asarray(trunc1)),
                        # second step exists only where the first left the branch alive;
                        # `second_valid` is false elsewhere and those frames are zeros,
                        # never a repeated terminal frame
                        "second_valid": torch.from_numpy(alive),
                        "second": torch.from_numpy(second),
                        "second_reward": torch.from_numpy(s_reward),
                        "second_health_delta": torch.from_numpy(s_delta),
                        "second_terminated": torch.from_numpy(s_dead),
                        "second_truncated": torch.from_numpy(s_trunc),
                    })


                observation, env_state, _, terminated, truncated = env_step(
                    env_state, chosen, seed + index + 1)
                # The real successor must be the chosen branch -- same state, same key.
                # The jitted vmapped render and the plain one can disagree by one level on
                # a pixel sitting exactly on a rounding boundary (measured: 1 of 11,907 at
                # 15000:143), so the check tolerates that and nothing more. A wrong action,
                # key or state would differ across most of the frame.
                drift = (observation.cpu().int() - frames_of(obs1[chosen]).cpu().int()).abs()
                assert int(drift.max()) <= 1, (
                    f"trajectory step disagrees with its own fork at {seed}:{index}: "
                    f"{int((drift > 0).sum())} pixels, max {int(drift.max())}")
                led.append(chosen)
                incoming.fill_(chosen)
                if terminated or truncated:
                    break

            target_file = args.out / f"seed-{seed:06d}-r{len(rows):04d}.pt"
            temporary = target_file.with_suffix(".tmp")
            torch.save(rows, temporary)
            temporary.replace(target_file)
            retained += len(rows)
            rows = []
            if (order + 1) % 10 == 0:
                rate = (order + 1) / (time.time() - started)
                print(f"  seed {order+1}/{args.seeds}  examined {examined:,}  "
                      f"stored {retained:,}  [{time.time()-started:.0f}s, "
                      f"{(args.seeds-order-1)/rate:.0f}s left]", flush=True)
            if args.target and retained >= args.target:
                print(f"reached target {args.target:,} roots", flush=True)
                break

    print(f"examined {examined:,} states, {retained:,} roots stored in "
          f"{time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
