"""Danger leads (user 11-07: "do we have any leads? discuss first"), CPU, saved futures only (no generation): exp 2's
zombie population for F3 (16 EDM steps; prefix-20) and F4 (K = 4; K = 8), split by what the world's own inputs can tell.
- Fresh arrival vs already beside: a zombie adjacent at state t - 1 but none adjacent at t - 2 cannot have hit within
  the window, so from the inputs P(hit | ready) is high (H16 measurement: 0.97 fresh vs 0.48 already beside, 4 frames).
- Asleep and ready (truth: hit and woken 1.00): the joint of the drawn wake and the drawn health change.
Writes readings/danger_leads.json."""
import importlib, json
from pathlib import Path
import numpy as np

CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RAW = {"f3_exp2": EXP / "20261106_determinant_curves/readings/raw.npz", "f3_t2": EXP / "20261106_decision_levels/readings/t2/raw.npz",
       "f4_k4": EXP / "20261106_p1f4/readings/consequences/curves/raw.npz", "f4_k8": HERE / "readings/k8/curves/raw.npz"}
r3 = lambda x: round(float(x), 3)


def main():
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    T = CV.transitions(L)
    g, s_off = T["g"], np.r_[0, np.cumsum(L["ep_T"] + 1)]
    prev = np.maximum(g - 1, 0)
    first = np.isin(g, s_off)                                            # t - 1 is the episode's first state
    pos_p = L["s_pos"][prev].astype(int)
    adj_prev = (L["s_zombies_m"][prev] & (np.abs(L["s_zombies_p"][prev].astype(int) - pos_p[:, None]).sum(-1) == 1)).any(1) & ~first
    out = {}
    for name, p in RAW.items():
        r = np.load(p, allow_pickle=True)
        j, st = r["zombie/j"], r["zombie/stratum"].astype(str)
        h0, h1 = T["body"][j, 0], T["true_body"][j, 0]
        gen_h, gen_awake = r["zombie/gen_hud63"], r["zombie/gen_asleep"] < 0.5
        dmg, tdmg = gen_h < h0, h1 < h0
        res = {}
        for key, m in (("awake_ready_fresh", np.char.startswith(st, "awake_ready") & ~adj_prev[j]),
                       ("awake_ready_beside", np.char.startswith(st, "awake_ready") & adj_prev[j]),
                       ("awake_cooling", np.char.startswith(st, "awake_cooling")),
                       ("asleep_ready_fresh", np.char.startswith(st, "asleep_ready") & ~adj_prev[j]),
                       ("asleep_ready_beside", np.char.startswith(st, "asleep_ready") & adj_prev[j]),
                       ("asleep_cooling", np.char.startswith(st, "asleep_cooling"))):
            res[key] = {"futures": int(m.sum()), "damage_gen": r3(dmg[m].mean()), "damage_true": r3(tdmg[m].mean())}
            if key.startswith("asleep_ready"):
                res[key].update({"woken_and_damaged": r3((gen_awake & dmg)[m].mean()), "woken_not_damaged": r3((gen_awake & ~dmg)[m].mean()),
                                 "asleep_and_damaged": r3((~gen_awake & dmg)[m].mean()),
                                 "drawn_delta_when_damaged": {int(d): int(c) for d, c in zip(*np.unique((gen_h - h0)[m & dmg], return_counts=True))},
                                 "true_delta": {int(d): int(c) for d, c in zip(*np.unique((h1 - h0)[m], return_counts=True))}})
            if key.startswith("awake_ready"):
                res[key]["drawn_delta_when_damaged"] = {int(d): int(c) for d, c in zip(*np.unique((gen_h - h0)[m & dmg], return_counts=True))}
        out[name] = res
        print(name, json.dumps({k: {q: v[q] for q in ("futures", "damage_gen", "damage_true")} for k, v in res.items()}), flush=True)
    (HERE / "readings/danger_leads.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
