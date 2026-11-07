"""Danger rule EDA (labels only): is a zombie hit a deterministic function of the state and action (game_logic.py
814-833: a zombie at distance 1 from the player's post-move position with attack cooldown <= 0 attacks; damage 2 awake /
7 asleep; cooldown -> 5, else -1 every step), and can readiness be recovered from the visible recent past?
Usage: python -m artifacts.experiments.20261107_f4_followups.danger_rule SPLIT"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = HERE.parents[1] / "eda/s0"


def main(split):
    L = np.load(LABELS / f"s0_{split}_labels.npz")
    T, s_off = L["ep_T"], np.r_[0, np.cumsum(L["ep_T"] + 1)]
    zp, zm, zc, pos, body, sleep = L["s_zombies_p"], L["s_zombies_m"], L["s_zombies_cd"], L["s_pos"], L["s_body"], L["s_sleep"]
    rows = []
    for e in range(len(T)):
        lo = s_off[e]
        last_hit = -10 ** 6
        for t in range(T[e]):
            g = lo + t
            d = np.abs(zp[g].astype(int) - pos[g + 1].astype(int)).sum(-1)       # pre-move zombies vs post-move player
            near = zm[g] & (d == 1)
            dmg = int(body[g, 0] - body[g + 1, 0])
            ready_near = near & (zc[g] <= 0)
            rule = bool(ready_near.any())
            attacked = bool((ready_near & (zc[g + 1] == 5)).any())         # its cooldown reset: it attacked
            killed = bool((ready_near & ~zm[g + 1]).any())                  # removed before its update (player's DO)
            if near.any():
                fresh = not (zm[g - 1] & (np.abs(zp[g - 1].astype(int) - pos[g].astype(int)).sum(-1) == 1)).any() if t > 0 else True
                rows.append((rule, dmg, bool(sleep[g]), int(near.sum()), t - last_hit, fresh, float(L["ep_eps"][e]), attacked, killed))
            if (zm[g] & (zc[g + 1] == 5) & (zc[g] <= 0)).any():
                last_hit = t
    a = np.array(rows, dtype=float)
    rule, dmg, asleep, n_near, since, fresh, attacked, killed = a[:, 0] > 0, a[:, 1], a[:, 2] > 0, a[:, 3], a[:, 4], a[:, 5] > 0, a[:, 7] > 0, a[:, 8] > 0
    zdmg = np.where(asleep, 7, 2)
    hit = dmg >= zdmg                                                    # this step's drop covers a zombie hit
    res = {"split": split, "eligible rows (a zombie at distance 1 from the post-move player)": int(len(a)),
           "hits (recorded drop >= zombie damage)": int(hit.sum()),
           "P(hit | rule) / P(hit | not rule)": [float(hit[rule].mean()), float(hit[~rule].mean())],
           "rule vs recorded, by sleep": {k: [float(hit[m & rule].mean()), float(hit[m & ~rule].mean()), int((m & rule).sum()), int((m & ~rule).sum())]
                                          for k, m in (("awake", ~asleep), ("asleep", asleep))},
           "share of eligible rows with > 1 zombie at distance 1": float((n_near > 1).mean()),
           "P(attack | rule) / P(attack | not rule) (attack = cooldown reset to 5)": [float(attacked[rule].mean()), float(attacked[~rule].mean())],
           "rule rows without an attack: share killed first by the player": float(killed[rule & ~attacked].mean()) if (rule & ~attacked).any() else None,
           "attacks per 1000 transitions": 1000 * float(attacked.sum()) / float(sum(T))}
    # visible-history observer: ready unless a hit happened in the last 5 steps (cooldown 5, one clock for all zombies)
    obs_ready = since > 5
    res["visible observer (no hit in the last 5 steps) vs the true rule"] = {
        "agreement": float((obs_ready == rule).mean()), "P(attack | observer ready) / P(attack | observer cooling)": [float(attacked[obs_ready].mean()), float(attacked[~obs_ready].mean())],
        "share of eligible rows the observer calls ready": float(obs_ready.mean())}
    res["fresh arrivals (no zombie at distance 1 one step earlier)"] = {"share": float(fresh.mean()), "P(hit | fresh)": float(hit[fresh].mean())}
    (HERE / f"readings/danger/danger_rule_{split}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
