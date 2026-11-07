"""P1-F4 against F3 on the same instruments (RESULTS.md): wake per stratum and woken beside a zombie (curves raw), each
(item, count) transition (curves raw), the actor interface per class, own-output recursion counts. F3's references:
exp 2 (16 EDM steps), decision levels T2 (prefix-20), F3's readout (interface 3 / 16 steps, recursion 3 / 16 steps).
Writes readings/comparison.json. CPU, reads saved outputs only."""
import importlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
CURVES = {"f3_exp2": EXP / "20261106_determinant_curves/readings/raw.npz", "f3_t2": EXP / "20261106_decision_levels/readings/t2/raw.npz",
          "f4": HERE / "readings/consequences/curves/raw.npz"}
INTERFACE = {"f3": EXP / "20261105_p1f3/readings/interface_steps.json", "f4": HERE / "readings/interface.json"}
RECURSION = {"f3_3": EXP / "20261105_p1f3/readings/recursion_steps3/raw.npz", "f3_16": EXP / "20261105_p1f3/readings/recursion_steps16/raw.npz",
             "f4_4": HERE / "readings/recursion_steps4/raw.npz"}
r3 = lambda x: round(float(x), 3)


def curves(r):
    s, ga, ra = r["zombie/stratum"].astype(str), r["zombie/gen_asleep"], r["zombie/rec_asleep"]
    woken = {st: {"gen": r3(1 - ga[np.char.startswith(s, st)].mean()), "true": r3(1 - ra[np.char.startswith(s, st)].mean())}
             for st in ("asleep_ready", "asleep_cooling")}
    sa = r["asleep/stratum"].astype(str)
    wake = {st: r3(1 - r["asleep/gen_asleep"][sa == st].mean()) for st in sorted(set(sa))}
    st = r["mining/stratum"].astype(str)
    kind, n = np.array([x.split("_")[0] for x in st]), np.array([int(x.split("_")[1]) for x in st])
    g = np.array([r[f"mining/gen_hud{CV.MINING[k][2]}"][i] for i, k in enumerate(kind)])
    trans = {f"{k}_{x}": {"next": r3((g[m] == x + 1).mean()), "copy": r3((g[m] == x).mean()), "futures": int(m.sum())}
             for k in CV.MINING for x in range(9) for m in [(kind == k) & (n == x)] if m.sum() >= 8}
    return {"woken_beside_zombie": woken, "woken_asleep_by_stratum": wake, "transitions": trans}


def interface():
    a, b = (json.loads(p.read_text())["checkpoints"] for p in INTERFACE.values())
    cks = {**{k: v["classes"] for k, v in a.items()}, **{k: v["classes"] for k, v in b.items()}}
    pick = lambda c, w: {q: r3(c[w]["achievement_unlock" if w == "recorded" and q == "unlock_p" else q]["est"]) for q in ("unlock_p", "neg_p", "term_p")}
    return {cl: {"recorded": pick(cks["f4_4steps"][cl], "recorded"), **{k: pick(v[cl], "gen") for k, v in cks.items() if cl in v}}
            for cl in cks["f4_4steps"]}


def recursion(r):
    steps = (0, 3, 7, 15)
    return {"truth": [r3(r["truth_count"].sum(-1)[:, s].mean()) for s in steps],
            **{m: [r3(r[f"{m}__count"].sum(-1)[:, s].mean()) for s in steps] for m in ("teacher", "free")},
            "free_terminated_by_16": r3(1 - r["free__active"][:, -1].mean()),
            "reward_per_step": {"truth": r3(r["truth_reward"].mean()), "teacher": r3(r["teacher__reward_sample"].mean()),
                                "free": r3(r["free__reward_sample"].mean())}}


def main():
    out = {"curves": {k: curves(np.load(p, allow_pickle=True)) for k, p in CURVES.items()}, "interface": interface(),
           "recursion_steps_1_4_8_16": {k: recursion(np.load(p)) for k, p in RECURSION.items()}}
    (HERE / "readings/comparison.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["recursion_steps_1_4_8_16"]), json.dumps(out["curves"]["f4"]["woken_beside_zombie"]))


if __name__ == "__main__":
    main()
