"""P1-F3's declared verdicts (PLAN.md "Declared readings"; Codex review 11-05, finding 2), computed once from the
readout's outputs against F2's pinned references. Differences between worlds use unpaired episode-bootstrap SEs
(conservative); "x beyond 3 SE" is the strict form (the estimate's 3-SE bound clears x). No new measurement.

Usage: python compare.py [--self-test]   (self-test: F2's own references in place of F3's outputs; every difference
must be 0, the lever verdict "no measurable lever", every guard pass)"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CT = ROOT / "artifacts/experiments/20261105_consequence_trace/readings"
REF = {"trace": CT / "f2/results.json", "stone": CT / "stone/f2/results.json", "interface": CT / "evening/interface_f2_steps.json",
       "recursion16": CT / "recursion/f2_steps16/raw.npz", "s1": ROOT / "artifacts/experiments/20261103_p1f2/run/s1/results.json"}
F3 = {"trace": HERE / "readings/trace/results.json", "stone": HERE / "readings/trace/results.json",
      "interface": HERE / "readings/interface_steps.json", "recursion16": HERE / "readings/recursion_steps16/raw.npz",
      "s1": HERE / "run/s1/results.json"}
KINDS = ("zombies", "cows", "skeletons", "arrows")


def se_diff(a, b):
    return b["est"] - a["est"], float(np.hypot(a["se"], b["se"]))


def field20(paths, kind):
    d = json.loads(Path(paths["stone" if kind == "stone" else "trace"]).read_text())
    return d["classes"][kind]["parts"]["hud"]["field"]["20.0"]["mean"]


def gen(paths, label):
    d = json.loads(Path(paths["interface"]).read_text())["checkpoints"]
    return d[next(k for k in d if k.endswith(label))]["classes"]


def count_gate(npz):
    """S2.1 count check (TEST_SPEC): per kind and mode, mean visible count over the 16 steps vs the reader on the real
    percepts; fires if the ratio is off by > 30% AND the difference is > 3 SE (episode clusters)."""
    from saccade.eval import gates as G
    z = np.load(npz); ep = z["episode"]; m = np.ones(len(ep), bool); out = {}
    for i, k in enumerate(KINDS):
        real = z["teacher__real_reader_count"][:, :, i].mean(1)
        for mode in ("teacher", "free"):
            g = z[f"{mode}__count"][:, :, i].mean(1); d = G.estimate(g - real, m, ep)
            ratio = float(g.mean() / real.mean()) if real.mean() else float("nan")
            out[f"{k}/{mode}"] = {"ratio": ratio, "diff": d, "fires": bool(abs(ratio - 1) > 0.3 and abs(d["est"]) > 3 * d["se"])}
    return out


def s1_presence(paths):
    t = json.loads(Path(paths["s1"]).read_text())["s1_1"]["trained"]
    return {f"{reg}/{q}": float(np.mean([s[reg][q]["est"] for s in t])) for reg in ("day_r1_2", "day_r1_4", "night_r1_2", "night_r1_4")
            for q in ("presence_recall", "presence_precision")}


def main():
    sys.path.insert(0, str(ROOT))
    test = "--self-test" in sys.argv
    f3 = REF if test else F3
    res = {"self_test": test, "references": {k: str(v.relative_to(ROOT)) for k, v in REF.items()}}
    # Primary: the field at sigma 20 (conditioning only)
    prim = {k: dict(zip(("diff", "se"), se_diff(field20(REF, k), field20(f3, k))), f2=field20(REF, k)["est"], f3=field20(f3, k)["est"])
            for k in ("coal", "iron", "wood", "stone", "drink")}
    beyond = lambda k: prim[k]["diff"] - 3 * prim[k]["se"] > 0
    c, i = prim["coal"]["f3"], prim["iron"]["f3"]
    lever = ("CONFIRMED" if c >= 0.25 and i >= 0.15 and beyond("coal") and beyond("iron") else
             "no measurable lever" if c <= 0.15 and i <= 0.08 else "partial")
    w, s = prim["wood"]["f3"], prim["stone"]["f3"]
    plateau = "signal-limited" if w >= 0.80 and s >= 0.80 else "not signal-limited" if w <= 0.72 and s <= 0.72 else "unresolved"
    res["primary"] = {"field_sigma20_hud": prim, "lever": lever, "plateau": plateau}
    # Actor-facing at 16 steps (native heads on generated futures)
    a, b = gen(REF, "_16steps"), gen(f3, "_16steps")
    act = {}
    for name, cls, q in (("first_coal", "ach_COLLECT_COAL", "unlock_p"), ("first_iron", "ach_COLLECT_IRON", "unlock_p"),
                         ("risky_death", "risky", "term_p"), ("adjacent_damage", "adjacent", "neg_p"), ("risky_damage", "risky", "neg_p")):
        d, e = se_diff(a[cls]["gen"][q], b[cls]["gen"][q])
        act[name] = {"f2": a[cls]["gen"][q]["est"], "f3": b[cls]["gen"][q]["est"], "diff": d, "se": e, "improved": bool(d - 3 * e > 0.05)}
    sap = b["sapling_context"]["gen"]["unlock_p"]["est"] - b["sapling_context"]["recorded"]["achievement_unlock"]["est"]
    act["sapling_calibrated"] = {"f3_minus_recorded": sap, "ok": bool(abs(sap) <= 0.02)}
    res["actor_16_steps"] = act
    # Guards
    g = {}
    tr = json.loads(Path(f3["trace"]).read_text())["classes"]
    biases = {f"{k}/{p}": x["field"]["0.05"]["bias"]["est"] for k, cl in tr.items() if "parts" in cl for p, x in cl["parts"].items()}
    g["low_noise_bias"] = {"max_abs": float(max(map(abs, biases.values()))), "ok": bool(max(map(abs, biases.values())) < 0.01)}
    p2, p3 = s1_presence(REF), s1_presence(f3)
    worst = min(p3[k] - p2[k] for k in p2)
    g["s1_presence"] = {"worst_f3_minus_f2": worst, "ok": bool(worst >= -0.02)}
    t2 = json.loads(Path(REF["s1"]).read_text())["s1_4"]["tsp_nats_per_pixel"]["all"]["est"]
    t3 = json.loads(Path(f3["s1"]).read_text())["s1_4"]["tsp_nats_per_pixel"]["all"]["est"]
    g["s1_tsp"] = {"f2": t2, "f3": t3, "ok": bool(t3 <= 1.05 * t2)}
    maj = {}
    for cls in ("ach_COLLECT_WOOD", "ach_COLLECT_STONE", "ach_PLACE_TABLE", "ach_PLACE_STONE", "ach_COLLECT_DRINK",
                "ach_MAKE_WOOD_PICKAXE", "ach_MAKE_WOOD_SWORD", "ach_MAKE_STONE_PICKAXE", "ach_MAKE_STONE_SWORD"):
        d, e = se_diff(a[cls]["gen"]["unlock_p"], b[cls]["gen"]["unlock_p"])
        maj[cls] = {"diff": d, "se": e, "breach": bool(d + 3 * e < -0.05)}
    g["common_classes_16"] = {"classes": maj, "ok": not any(v["breach"] for v in maj.values())}
    d, e = se_diff(a["random"]["gen"]["unlock_p"], b["random"]["gen"]["unlock_p"])
    g["false_rewards_16"] = {"f2": a["random"]["gen"]["unlock_p"]["est"], "f3": b["random"]["gen"]["unlock_p"]["est"],
                             "diff": d, "se": e, "ok": not bool(d - 3 * e > 0.01)}
    cg = count_gate(f3["recursion16"]); n = sum(v["fires"] for v in cg.values())
    g["own_history_count_gate_16"] = {"fires": n, "checks": cg, "ok": bool(n <= 2)}
    res["guards"] = g
    out = HERE / ("readings/comparison_self_test.json" if test else "readings/comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
    print("lever:", lever, "| plateau:", plateau)
    print("field@20:", {k: f"{v['f2']:.3f}->{v['f3']:.3f} ({v['diff']:+.3f}+-{v['se']:.3f})" for k, v in prim.items()})
    print("actor@16:", {k: (f"{v['f2']:.3f}->{v['f3']:.3f} imp={v['improved']}" if "f2" in v else v) for k, v in act.items()})
    print("guards:", {k: v["ok"] for k, v in g.items()}, "| count-gate fires", n)
    if test:
        assert all(v["diff"] == 0 for v in prim.values()) and all(v["diff"] == 0 for k, v in act.items() if "diff" in v), "self-test"
        assert lever == "no measurable lever" and all(v["ok"] for v in g.values()), "self-test guards"
        print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
