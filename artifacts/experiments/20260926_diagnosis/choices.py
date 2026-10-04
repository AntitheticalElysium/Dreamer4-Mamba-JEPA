"""D3. What each world's own head chooses on zombie roots, and why -- from the D2 dump, nothing trained.

The decision structure (D1): on a zombie-adjacent root, any move into a free, non-lava neighbour tile is
safe (the zombie can only hit at distance 1 after the move); staying put -- a non-move action, or a move
into a solid tile / mob -- is what gets hit. So each chosen action is one of:
  move_ok     a move the visible rule says succeeds (target walkable, no mob, not lava)
  move_blocked a move into a solid tile or a mob (the player stays)
  stay        any non-move action (NOOP, DO, SLEEP, PLACE_*, MAKE_*)
Per world, on the zombie opportunity roots of 55k-58k: the share of each category, and expected safe within
each. Then the head's preference between categories: per root, mean P(dead) over move_ok actions minus mean
over stay actions ("stay_minus_move", > 0 = the head knows staying is worse), on generated and on true
successors. Then the action-marginal decomposition: S_marginal = mean over roots of sum_a pi(a) (1 - p(a)),
pi = the world's own chosen-action histogram, i.e. what its choices are worth with the state ignored.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
from decision import C, LAVA, MOVES, SOLID, unpack  # noqa: E402
from frozen_ladder import strata  # noqa: E402

DUMP = ROOT / "artifacts/eda/diagnosis_dump_v1"
WORLDS = ("H2", "Z", "ZW", "U", "W", "W_s2")
N = 17


def move_table(visible):
    """[n, 17] category: 0 move_ok, 1 move_blocked, 2 stay; and [n, 17] zombies adjacent after the action."""
    cat = torch.full((len(visible), N), 2, dtype=torch.long)
    adj = torch.zeros(len(visible), N)
    for i, v in enumerate(visible):
        tiles, mobs, _, sleeping = unpack(v)
        occupied = {(r, c) for r in range(7) for c in range(9) if mobs[r, c, :3].sum() > 0}
        zombies = [(r, c) for r in range(7) for c in range(9) if mobs[r, c, 0] > 0]
        for a in range(N):
            pos = C
            if a in MOVES and not sleeping:
                t = (C[0] + MOVES[a][0], C[1] + MOVES[a][1])
                ok = int(tiles[t]) not in SOLID and t not in occupied and int(tiles[t]) != LAVA
                cat[i, a] = 0 if ok else 1
                if int(tiles[t]) not in SOLID and t not in occupied:
                    pos = t
            adj[i, a] = sum(abs(r - pos[0]) + abs(c - pos[1]) == 1 for r, c in zombies)
    return cat, adj


def main():
    blocks = sorted({p.name.split("_")[0] for p in DUMP.glob("*_meta.pt")})
    blocks = [b for b in blocks if all((DUMP / f"{b}_{w}.pt").exists() for w in WORLDS)]
    result = {"blocks": blocks, "worlds": {}}
    metas = {b: torch.load(DUMP / f"{b}_meta.pt") for b in blocks}
    tables = {b: move_table(metas[b]["visible"]) for b in blocks}
    for w in WORLDS:
        per = {}
        for b in blocks:
            m, d = metas[b], torch.load(DUMP / f"{b}_{w}.pt")
            p = m["p_death1"]
            zom = strata(m["visible"])["zombie_adjacent"]
            cat, _ = tables[b]
            row = {}
            for key in ("p_dead", "p_dead_real"):
                score = d[key]
                chosen = score.argmin(1)
                safe = 1 - p.gather(1, chosen[:, None]).squeeze(1)
                ccat = cat.gather(1, chosen[:, None]).squeeze(1)
                pi = torch.bincount(chosen[zom], minlength=N).float()
                pi = pi / pi.sum()
                marginal = float(((1 - p[zom]) * pi).sum(1).mean())
                ok, stay = (cat == 0).float(), (cat == 2).float()
                has = zom & (ok.sum(1) > 0)
                gap = (score * stay).sum(1) / stay.sum(1).clamp_min(1) - (score * ok).sum(1) / ok.sum(1).clamp_min(1)
                row[key] = {
                    "safe_zombie": float(safe[zom].mean()),
                    "share": {name: float((ccat[zom] == k).float().mean()) for k, name in enumerate(("move_ok", "move_blocked", "stay"))},
                    "safe_given": {name: float(safe[zom & (ccat == k)].mean()) if bool((zom & (ccat == k)).any()) else None
                                   for k, name in enumerate(("move_ok", "move_blocked", "stay"))},
                    "state_ignored_marginal": marginal,
                    "stay_minus_move_pdead": float(gap[has].mean()),
                    "stay_worse_share": float((gap[has] > 0).float().mean()),
                    "top_actions": {int(a): round(float(pi[a]), 3) for a in pi.argsort(descending=True)[:4]},
                }
            per[b] = row
        result["worlds"][w] = per
        print(w, json.dumps({b: {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                                     if kk in ("safe_zombie", "share", "state_ignored_marginal", "stay_minus_move_pdead")}
                                 for k, v in r.items() if k == "p_dead"} for b, r in per.items()}), flush=True)
    (HERE / "choices.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
