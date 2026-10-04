"""D1. What the one-step death decision actually depends on, read off the game code.

craftax_classic/game_logic.py: the player moves first (move_player: blocked by SOLID_BLOCKS, mobs, bounds;
lava is enterable and kills), then each zombie attacks iff Manhattan distance == 1 AND its hidden
attack_cooldown <= 0 (damage 2 awake, 7 asleep; cooldown reset to 5, else decremented). A sleeping player's
action is replaced by NOOP. So the outcome of (state, action) is a deterministic function of: the 4 neighbour
tiles (walkable?), mob positions, health, sleep flag, lava -- all visible -- and the zombie cooldown (hidden).

This scores rules built only from those facts against the simulator's 32-key P(death1) on the four blocks
already opened (55k-58k). Nothing is fitted. Rules:
  visible_rule   minimise (enters lava, #zombies at distance 1 after the move); ties averaged
  cooldown_rule  the same, counting only zombies whose hidden cooldown <= 0 (the full-state rule)
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
from frozen_ladder import strata  # noqa: E402

BLOCKS = {"55k": "observe_fresh_v6", "56k": "observe_fresh_v7", "57k": "observe_fresh_v8", "58k": "observe_fresh_v9"}
SOLID = {1, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16}          # OUT_OF_BOUNDS + constants.SOLID_BLOCKS
LAVA, N = 14, 17
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
C = (3, 4)


def load(store):
    rows = [r for f in sorted((ROOT / "artifacts/eda" / store).glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
    return rows


def unpack(v):
    tiles = v[:1071].reshape(7, 9, 17).argmax(-1)
    mobs = v[1071:1512].reshape(7, 9, 7)
    return tiles, mobs, float(v[1512]) * 9, bool(v[1520] > 0.5)


def rule(row, use_cooldown):
    v, hidden = row["visible"].float(), row["hidden"].float()
    tiles, mobs, health, sleeping = unpack(v)
    cooldown = hidden[:7 * 9 * 5].reshape(7, 9, 5)[..., 0] * 5
    zombies = [(r, c) for r in range(7) for c in range(9) if mobs[r, c, 0] > 0
               and (not use_cooldown or cooldown[r, c] <= 0)]
    occupied = {(r, c) for r in range(7) for c in range(9) if mobs[r, c, :3].sum() > 0}
    risk = torch.zeros(N)
    for a in range(N):
        pos = C
        if not sleeping and a in MOVES:
            t = (C[0] + MOVES[a][0], C[1] + MOVES[a][1])
            if int(tiles[t]) not in SOLID and t not in occupied:
                pos = t
        lava = int(tiles[pos]) == LAVA
        adjacent = sum(abs(r - pos[0]) + abs(c - pos[1]) == 1 for r, c in zombies)
        risk[a] = 100 * lava + adjacent
    return risk


def tie_safe(risk, p):
    best = risk == risk.min()
    return float((1 - p[best]).mean())


def main():
    out = {}
    for block, store in BLOCKS.items():
        rows = load(store)
        p = torch.stack([r["p_death1"].float() for r in rows])
        vis = torch.stack([r["visible"].float() for r in rows])
        opp = p.amax(1) > p.amin(1)
        zom = opp & strata(vis)["zombie_adjacent"]
        res = {}
        for name, cd in (("visible_rule", False), ("cooldown_rule", True)):
            s = torch.tensor([tie_safe(rule(r, cd), r["p_death1"].float()) for r in rows])
            res[name] = {"overall": float(s[opp].mean()), "zombie": float(s[zom].mean())}
        ceiling = 1 - p.amin(1)
        down = 1 - p[:, 4]
        res["ceiling"] = {"overall": float(ceiling[opp].mean()), "zombie": float(ceiling[zom].mean())}
        res["DOWN"] = {"overall": float(down[opp].mean()), "zombie": float(down[zom].mean())}
        res["n"] = {"opportunity": int(opp.sum()), "zombie": int(zom.sum())}
        out[block] = res
        print(block, json.dumps(res))
    (Path(__file__).parent / "decision.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
