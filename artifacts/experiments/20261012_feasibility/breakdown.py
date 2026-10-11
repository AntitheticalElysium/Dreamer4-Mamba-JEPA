"""10-12 feasibility, breakdown (written while q1-q3 ran; POST HOC, descriptive): where do the one-step innovation (q2) and the
decision gains (q3) live? Root features from the replayed root state: visible mobs (zombies / cows / skeletons / arrows inside
the 7 x 9 view), any zombie within Manhattan distance 3 of the player, night (light < 0.5), sleeping, health.
Reports: q2 visible entropy by visible-mob count and by action group (move / noop / DO / other); q3 S16 - mode and S1 - mode on
opportunity roots by whether a zombie is near.
Usage: breakdown.py  -> artifacts/eda/feasibility_v1/breakdown.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feasibility as FE  # noqa: E402


def features(r):
    s = FE.root_state(r)
    p = np.asarray(s.player_position)

    def visible(mob):
        pos, mask = np.asarray(mob.position), np.asarray(mob.mask)
        d = pos - p
        return int((mask & (np.abs(d[:, 0]) <= 3) & (np.abs(d[:, 1]) <= 4)).sum())
    z = np.asarray(s.zombies.position); zm = np.asarray(s.zombies.mask)
    near = bool((zm & (np.abs(z - p).sum(1) <= 3)).any())
    return {"vis_zombies": visible(s.zombies), "vis_cows": visible(s.cows), "vis_skeletons": visible(s.skeletons),
            "vis_arrows": visible(s.arrows), "zombie_near": near, "night": float(s.light_level) < 0.5,
            "sleeping": bool(s.is_sleeping), "health": int(s.player_health)}


def main():
    FE.RP.env_and_render()
    roots = FE.load_roots(0)
    feats = [features(r) for r in roots]
    out = {}
    q2p, q3p = FE.OUT / "q2_rows.pt", FE.OUT / "q3_rows.pt"
    if q2p.exists():
        rows = torch.load(q2p, weights_only=False)
        nm = np.array([sum(feats[x["root"]][k] for k in ("vis_zombies", "vis_cows", "vis_skeletons", "vis_arrows")) for x in rows])
        H = np.array([x["vis_H_mm"] for x in rows]); m = np.array([x["vis_m"] for x in rows])
        out["q2_by_visible_mobs"] = {str(k): {"root_actions": int((nm == k).sum()), "H_mm_mean": round(float(H[nm == k].mean()), 3),
                                              "H_mm_p90": round(float(np.quantile(H[nm == k], 0.9)), 3),
                                              "deterministic_share": round(float((m[nm == k] == 1).mean()), 3)}
                                     for k in sorted(set(nm.tolist()))}
        grp = np.array(["move" if 1 <= x["action"] <= 4 else "noop" if x["action"] == 0 else "do" if x["action"] == 5 else "other" for x in rows])
        out["q2_by_action"] = {g: round(float(H[grp == g].mean()), 3) for g in ("noop", "move", "do", "other")}
        out["q2_bits_per_visible_mob"] = round(float(H[nm > 0].sum() / nm[nm > 0].sum()), 3)
    if q3p.exists():
        rows = torch.load(q3p, weights_only=False)
        res = json.loads((FE.OUT / "q3.json").read_text()) if (FE.OUT / "q3.json").exists() else {}
        near = np.array([feats[x["root"]]["zombie_near"] for x in rows])
        night = np.array([feats[x["root"]]["night"] for x in rows])
        opp = np.array([x["dead_sel"].mean(1).max() > x["dead_sel"].mean(1).min() for x in rows])
        out["q3_opportunity_by_zombie_near"] = {"near": int((opp & near).sum()), "not_near": int((opp & ~near).sum())}
        out["q3_opportunity_by_night"] = {"night": int((opp & night).sum()), "day": int((opp & ~night).sum())}
        out["root_features_summary"] = {k: round(float(np.mean([f[k] for f in feats])), 3) for k in feats[0]}
    (FE.OUT / "breakdown.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
