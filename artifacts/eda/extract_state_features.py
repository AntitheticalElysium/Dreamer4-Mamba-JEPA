"""Simulator-state features at every hazard root, split into visible and hidden.

The split is by what the renderer actually draws. Craftax-Classic renders a 9x7
player-centred window plus a status bar, so: local tiles, mob sprites inside the
window, the four vital digits, the sleep overlay and light level are visible;
absolute position, the map beyond the window, the four continuous counters, mob
health and -- the one this test exists for -- `Mobs.attack_cooldown` are not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from artifacts.localize_counterfactual import load_models
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.transition import observe

OUT = HERE / "state_features"
OUT.mkdir(exist_ok=True)
N_BLOCK = 17
NEAR = 3          # nearest mobs described per type
TYPES = ("zombies", "skeletons", "cows", "arrows")


def one_hot(value: int, size: int) -> list[float]:
    out = [0.0] * size
    if 0 <= value < size:
        out[value] = 1.0
    return out


def features(state) -> tuple[list[float], list[float]]:
    """(visible, hidden). Every entry is placed in exactly one of the two."""
    position = np.asarray(state.player_position)
    board = np.asarray(state.map)

    visible: list[float] = [
        float(state.player_health), float(state.player_food),
        float(state.player_drink), float(state.player_energy),
        float(state.is_sleeping), float(state.light_level),
    ]
    visible += one_hot(int(state.player_direction), 5)
    inventory = state.inventory
    visible += [float(getattr(inventory, name)) for name in
                ("wood", "stone", "coal", "iron", "diamond", "sapling",
                 "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
                 "wood_sword", "stone_sword", "iron_sword")]
    # the 9x7 rendered window of tiles, one-hot
    for dx in range(-4, 5):
        for dy in range(-3, 4):
            x, y = position[0] + dx, position[1] + dy
            tile = int(board[x, y]) if 0 <= x < board.shape[0] and 0 <= y < board.shape[1] else 1
            visible += one_hot(tile, N_BLOCK)

    hidden: list[float] = [
        float(state.player_recover), float(state.player_hunger),
        float(state.player_thirst), float(state.player_fatigue),
        float(state.timestep), float(position[0]), float(position[1]),
    ]
    for name in TYPES:
        mobs = getattr(state, name)
        mask = np.asarray(mobs.mask)
        pos = np.asarray(mobs.position)
        cooldown = np.asarray(mobs.attack_cooldown)
        health = np.asarray(mobs.health)
        offsets = pos - position
        inside = mask & (np.abs(offsets[:, 0]) <= 4) & (np.abs(offsets[:, 1]) <= 3)
        order = np.argsort(np.where(inside, np.abs(offsets).sum(-1), 999))[:NEAR]
        for index in order:
            live = bool(inside[index])
            # geometry of a mob drawn inside the window is visible
            visible += [float(live), float(offsets[index, 0]) * live,
                        float(offsets[index, 1]) * live]
            # its attack phase and health are never drawn
            hidden += [float(cooldown[index]) * live, float(health[index]) * live]
        for _ in range(NEAR - len(order)):
            visible += [0.0, 0.0, 0.0]
            hidden += [0.0, 0.0]
    return visible, hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=512)
    parser.add_argument("--seed-start", type=int, default=14_000)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    encoder, world, heads = load_models(
        ROOT / "artifacts/stage_a_terminalfix/phase1a.pt",
        ROOT / "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt",
        base, config)

    wanted = {}
    for path in sorted((HERE / "branched_damage").glob("seed-*.pt")):
        payload = torch.load(path, weights_only=False)
        seed = int(payload["seed"])
        for row in payload["rows"]:
            label = ((row["health"].numpy() <= -1) | row["dead"].numpy())
            if label.any() and not label.all():
                wanted.setdefault(seed, {})[int(row["step"])] = label.astype(np.float32)
    print(f"{sum(len(v) for v in wanted.values())} hazard roots across "
          f"{len(wanted)} seeds", flush=True)

    started, rows = time.time(), []
    with torch.no_grad():
        for order, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds)):
            if seed not in wanted:
                continue
            observation, env_state = reset(seed)
            state = None
            incoming = torch.full((1, 1), config.n_actions, dtype=torch.long,
                                  device=config.device)
            world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
            policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
            target = max(wanted[seed])
            for index in range(min(target + 1, args.limit)):
                patches = patchify(observation[None, None], config.patch).to(config.device)
                state, agent = observe(world, encoder, state, incoming, patches,
                                       world_rng, config)
                logits = heads(agent)["policy"][:, -1, 0]
                chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
                if index in wanted[seed]:
                    visible, hidden = features(env_state)
                    rows.append({
                        "seed": seed, "step": index,
                        "visible": torch.tensor(visible, dtype=torch.float32),
                        "hidden": torch.tensor(hidden, dtype=torch.float32),
                        "latent": state.world.latent[0, -1].flatten().cpu(),
                        "label": torch.from_numpy(wanted[seed][index]),
                    })
                observation, env_state, _, terminated, truncated = env_step(
                    env_state, chosen, seed + index + 1)
                incoming.fill_(chosen)
                if terminated or truncated:
                    break
            if (order + 1) % 50 == 0:
                rate = (order + 1) / (time.time() - started)
                print(f"  seed {order+1}/{args.seeds} roots {len(rows)} "
                      f"[{time.time()-started:.0f}s, {(args.seeds-order-1)/rate:.0f}s left]",
                      flush=True)
    torch.save(rows, OUT / "features.pt")
    (OUT / "manifest.json").write_text(json.dumps({
        "roots": len(rows),
        "visible_dim": len(rows[0]["visible"]), "hidden_dim": len(rows[0]["hidden"]),
        "hidden_contents": "player_recover/hunger/thirst/fatigue, timestep, absolute "
                           "position, per-mob attack_cooldown and health",
    }, indent=2))
    print(f"wrote {len(rows)} roots, visible dim {len(rows[0]['visible'])}, "
          f"hidden dim {len(rows[0]['hidden'])}")


if __name__ == "__main__":
    main()
