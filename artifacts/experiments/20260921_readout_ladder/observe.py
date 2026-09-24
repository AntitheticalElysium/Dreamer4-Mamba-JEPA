"""Observability data: the full simulator state beside every fork root, split into what is drawn.

The replay pilot (`REPLAY.md`) showed one-step death is fixed by the full state and action, and the
ranking probes (`RANKPROBE.md`) found no representation of the root observation that beats the
action prior. The open question is whether the deciding state can be recovered from what the agent
SEES. This writes the data to answer it, in two modes sharing one walk:

  replay   re-walk the 700 FIT seeds; at each stored root, prove the replay reproduces it exactly
           (as `replay.py` does) and save the state features and repeated-key probabilities.
           Development data: the pixel history is already in the fork store.
  collect  walk FRESH seeds from 50,000 -- a range nothing in this codebase has used -- with the
           collector's exact loop and retention rule (`collect_broad_forks.py`: damaging-choice,
           32-frame history), writing rows in the fork store's own schema plus the state features,
           until the predeclared stop rule is met. Judgement data, untouched by anything.

State split, read off the renderer (`craftax_classic/renderer.py`) rather than assumed:

  visible  the 7x9 egocentric tile view (17 block types); zombies, cows, skeletons on screen; arrows
           by flight direction (the texture is flipped/transposed per direction); player facing and
           sleep sprite; night darkness (light level); the HUD digits for health, food, drink,
           energy; inventory counts.
  hidden   never drawn: each on-screen zombie's and skeleton's attack cooldown and every mob's health,
           and the player's recover / hunger / thirst / fatigue accumulators.
  history  32 steps of the visible HUD stats, sleep, light, and zombie / skeleton / arrow counts in
           the 3x3 cells around the player. Attacks need adjacency and reset a cooldown to 5
           (`game_logic.py:817-837`), and a hit shows as a health drop beside a zombie, so this is
           the visible trace from which a cooldown could in principle be inferred.

Every root also gets P(death1 | s, a) and P(death2 | s, a) over K independent key pairs, the second
step being the stored NOOP, so outcomes can be scored as expected risk rather than one draw.

Stop rule for `collect`, fixed here before any fresh seed is walked: seeds in order from 50,000,
stopping at the first seed boundary where the one-step opportunity roots -- roots whose P(death1)
varies across the 17 actions -- reach 500, or at 1,000 seeds, whichever comes first.
"""

import argparse
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
HERE = Path(__file__).parent
EDA = ROOT / "artifacts/eda"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EDA))

from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.lewm_diagnostics import FORK_STORE
from d4mj.transition import observe

from collect_broad_forks import HISTORY, N_ACTIONS, frames_of, load_policy, make_fork  # noqa: E402
from confirm import seeds_for  # noqa: E402
from replay import make_second, outcome  # noqa: E402

ROWS, COLS = 7, 9                    # craftax_classic OBS_DIM
BLOCKS = 17
FRESH = EDA / "observe_fresh_v1"
FIT_STATE = EDA / "observe_fit_v1"


def _local(position, player):
    local = np.asarray(position) - np.asarray(player) + np.array([ROWS // 2, COLS // 2])
    return local, (local >= 0).all(-1) & (local < [ROWS, COLS]).all(-1)


def state_features(s):
    """(visible, hidden, history_step) as float32 vectors, from one EnvState."""
    player = np.asarray(s.player_position)
    padded = np.pad(np.asarray(s.map), ROWS + 2, constant_values=1)       # OUT_OF_BOUNDS
    top = player - np.array([ROWS // 2, COLS // 2]) + ROWS + 2
    view = padded[top[0]:top[0] + ROWS, top[1]:top[1] + COLS]
    tiles = np.eye(BLOCKS, dtype=np.float32)[view]                         # [7, 9, 17]
    mobs = np.zeros((ROWS, COLS, 7), np.float32)                           # zombie cow skeleton, arrows x4
    hidden_grid = np.zeros((ROWS, COLS, 5), np.float32)                    # z cd, z hp, s cd, s hp, cow hp
    for channel, group, extra in ((0, s.zombies, (0, 1)), (1, s.cows, (None, 4)), (2, s.skeletons, (2, 3))):
        local, on = _local(group.position, player)
        on = on & np.asarray(group.mask)
        for i in np.where(on)[0]:
            r, c = local[i]
            mobs[r, c, channel] += 1
            if extra[0] is not None:
                hidden_grid[r, c, extra[0]] = float(np.asarray(group.attack_cooldown)[i]) / 5.0
            hidden_grid[r, c, extra[1]] = float(np.asarray(group.health)[i]) / 5.0
    local, on = _local(s.arrows.position, player)
    on = on & np.asarray(s.arrows.mask)
    directions = np.asarray(s.arrow_directions)
    for i in np.where(on)[0]:
        d = tuple(int(x) for x in directions[i])
        channel = {(-1, 0): 3, (1, 0): 4, (0, -1): 5, (0, 1): 6}.get(d, 3)
        mobs[local[i][0], local[i][1], channel] += 1
    hud = np.array([float(s.player_health), float(s.player_food), float(s.player_drink),
                    float(s.player_energy)], np.float32) / 9.0
    facing = np.eye(4, dtype=np.float32)[int(s.player_direction) - 1]
    inventory = np.array([float(v) for v in jax.tree_util.tree_leaves(s.inventory)], np.float32) / 9.0
    player_bits = np.array([float(s.is_sleeping), float(s.light_level)], np.float32)
    visible = np.concatenate([tiles.ravel(), mobs.ravel(), hud, facing, player_bits, inventory])
    hidden = np.concatenate([hidden_grid.ravel(),
                             np.array([float(s.player_recover), float(s.player_hunger),
                                       float(s.player_thirst), float(s.player_fatigue)], np.float32)])
    near = mobs[ROWS // 2 - 1:ROWS // 2 + 2, COLS // 2 - 1:COLS // 2 + 2]
    near = np.stack([near[..., 0], near[..., 2], near[..., 3:].sum(-1)], -1).ravel()
    history_step = np.concatenate([hud, player_bits, near])
    return visible, hidden, history_step


def repeated(fork, second, env_state, seed, index, keys):
    """P(death1) and P(death2) over `keys` independent key pairs, never the original key."""
    split = jax.random.split(jax.random.PRNGKey(2**30 + seed * 1000 + index), 2 * keys)
    one, two = np.zeros(N_ACTIONS), np.zeros(N_ACTIONS)
    for k in range(keys):
        a, b, _, _ = outcome(fork, second, env_state, split[k], split[keys + k])
        one += a
        two += b
    return (one / keys).astype(np.float32), (two / keys).astype(np.float32)


def walk(seed, policy, config, fork, second, keys, stored=None):
    """One episode with the collector's loop. `stored` (replay) maps step -> stored row to verify;
    without it (collect), roots are retained by the collector's own damaging-choice rule."""
    encoder, world, heads = policy
    observation, env_state = reset(seed)
    state = None
    incoming = torch.full((1, 1), config.n_actions, dtype=torch.long, device=config.device)
    world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
    policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
    frames, led, history, out = [], [config.n_actions], [], []
    if stored is not None and not stored:
        return []
    last = max(stored) if stored is not None else 400 - 1
    for index in range(last + 1):
        frames.append(observation.clone())
        visible, hidden, step_features = state_features(env_state)
        history.append(step_features)
        patches = patchify(observation[None, None], config.patch).to(config.device)
        state, agent = observe(world, encoder, state, incoming, patches, world_rng, config)
        logits = heads(agent)["policy"][:, -1, 0]
        chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
        key1, key2 = jax.random.PRNGKey(seed + index + 1), jax.random.PRNGKey(seed + index + 2)
        keep = index in stored if stored is not None else False
        if stored is None and len(frames) >= HISTORY:
            obs1, _, reward1, dead1, trunc1, health1, ach1 = fork(env_state, key1, jnp.arange(N_ACTIONS))
            delta = np.asarray(health1) - float(env_state.player_health)
            damaging = (delta <= -1) | np.asarray(dead1)
            keep = bool(damaging.any() and not damaging.all())
        if keep:
            d1, d2, alive, raw2 = outcome(fork, second, env_state, key1, key2)
            p1, p2 = repeated(fork, second, env_state, seed, index, keys)
            hist = np.stack(history[-HISTORY:])
            record = {"seed": seed, "step": index, "visible": torch.from_numpy(visible),
                      "hidden": torch.from_numpy(hidden), "history": torch.from_numpy(hist),
                      "p_death1": torch.from_numpy(p1), "p_death2": torch.from_numpy(p2),
                      "death1": torch.from_numpy(d1), "death2": torch.from_numpy(d2)}
            if stored is not None:
                row = stored[index]
                drift = int((torch.stack(frames[-HISTORY:]).int() - row["frames"].int()).abs().max())
                bad = int((d1 != row["terminated"].numpy()).sum()) + int(
                    (alive != row["second_valid"].numpy()).sum()) + int(
                    (raw2[alive] != row["second_terminated"].numpy()[alive]).sum())
                if drift > 1 or bad:
                    raise SystemExit(f"replay does not reproduce stored root {seed}:{index} "
                                     f"(drift {drift}, outcome mismatches {bad})")
            else:
                # The fork store's own row schema, so every existing reader works on fresh roots.
                second_frames = np.zeros((N_ACTIONS,) + tuple(frames[0].shape), dtype=np.uint8)
                s_reward, s_delta = np.zeros(N_ACTIONS, np.float32), np.zeros(N_ACTIONS, np.float32)
                s_trunc = np.zeros(N_ACTIONS, bool)
                _, state1, _, _, _, h1, _ = fork(env_state, key1, jnp.arange(N_ACTIONS))
                for a in np.where(alive)[0]:
                    branch = jax.tree_util.tree_map(lambda x, i=int(a): x[i], state1)
                    o2, _, r2, _, t2, h2, _ = fork(branch, key2, jnp.array([0]))
                    second_frames[a] = frames_of(o2[0]).numpy()
                    s_reward[a], s_delta[a], s_trunc[a] = float(r2[0]), float(h2[0]) - float(h1[a]), bool(t2[0])
                record.update({
                    "frames": torch.stack(frames[-HISTORY:]),
                    "led_to_action": torch.tensor(led[-HISTORY:], dtype=torch.long),
                    "bc_action": chosen, "successors": frames_of(obs1),
                    "reward": torch.from_numpy(np.asarray(reward1, np.float32)),
                    "health_delta": torch.from_numpy(delta.astype(np.float32)),
                    "achievement_delta": torch.from_numpy(
                        (np.asarray(ach1) - int(env_state.achievements.sum())).astype(np.int16)),
                    "terminated": torch.from_numpy(d1), "truncated": torch.from_numpy(np.asarray(trunc1)),
                    "second_valid": torch.from_numpy(alive), "second": torch.from_numpy(second_frames),
                    "second_reward": torch.from_numpy(s_reward),
                    "second_health_delta": torch.from_numpy(s_delta),
                    "second_terminated": torch.from_numpy(raw2 & alive),
                    "second_truncated": torch.from_numpy(s_trunc)})
            out.append(record)
        observation, env_state, _, terminated, truncated = env_step(env_state, chosen, seed + index + 1)
        led.append(chosen)
        incoming.fill_(chosen)
        if terminated or truncated:
            break
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "collect"), required=True)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--keys", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=50_000)
    parser.add_argument("--max-seeds", type=int, default=1000)
    parser.add_argument("--target-opportunity", type=int, default=500)
    parser.add_argument("--limit-seeds", type=int, default=0, help="smoke only")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    out = args.out or (FIT_STATE if args.mode == "replay" else FRESH)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)

    base = replace(Config(), n_latents=32)
    config = replace(base, transition="direct", time_mixer="attention")
    policy = load_policy(base)
    fork, second = make_fork(), make_second()
    done = {int(f.stem.split("-")[1]) for f in out.glob("seed-*-r*.pt")}

    if args.mode == "replay":
        fit_seeds, _ = seeds_for(json.loads(args.partition.read_text()), FORK_STORE)
        paths = {int(p.stem.split("-")[1]): p for p in FORK_STORE.glob("seed-*.pt")}
        order = sorted(fit_seeds)[:args.limit_seeds or None]
    else:
        order = list(range(args.seed_start, args.seed_start + args.max_seeds))[:args.limit_seeds or None]
    # Resuming must count what is already written, or the stop rule would overshoot.
    opportunity = sum(int(r["p_death1"].max() > r["p_death1"].min())
                      for f in out.glob("seed-*-r*.pt") for r in torch.load(f, weights_only=False))
    with torch.no_grad():
        for number, seed in enumerate(order):
            if seed in done:
                continue
            stored = None
            if args.mode == "replay":
                stored = {int(r["step"]): r for r in torch.load(paths[seed], weights_only=False)
                          if len(r["frames"]) >= HISTORY}
            records = walk(seed, policy, config, fork, second, args.keys, stored)
            if args.mode == "replay" and len(records) != len(stored):
                raise SystemExit(f"replay of {seed} produced {len(records)} of {len(stored)} roots")
            target = out / f"seed-{seed:06d}-r{len(records):04d}.pt"
            torch.save(records, target.with_suffix(".tmp"))
            target.with_suffix(".tmp").replace(target)
            opportunity += sum(int(r["p_death1"].max() > r["p_death1"].min()) for r in records)
            log(seed=seed, done=number + 1, roots=len(records), opportunity=opportunity)
            if args.mode == "collect" and opportunity >= args.target_opportunity:
                log(stage="stop_rule_met", seeds=number + 1, opportunity=opportunity)
                break
    total = sum(1 for _ in out.glob("seed-*-r*.pt"))
    log(status="observe_complete", mode=args.mode, seed_files=total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
