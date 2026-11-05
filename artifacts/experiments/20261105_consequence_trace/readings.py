"""PLAN.md's declared readings R0-R8 applied mechanically to one trace results.json (written 11-05 while F2's trace
ran, checked on the GPU smoke only; no F2 / F2.1 number seen). "x beyond 3 SE" = est + 3 SE < x (strict side) for the
negative thresholds of R5 / R4. Wood has no B1 population: its R2 / R5 pattern is shown, the interpretation needing
reader (b) is "not localised" by the plan's own rule.
Amendment from the GPU smoke, before any F2 / F2.1 number: for a class the conditioning determines, the exact
denoiser's class-average f at the first call is f_recorded (law of total expectation), which the two-outcome value (1)
matches only where f_recorded ~ 1 (HUD parts; the smoke's iron map cell records 0.76). R2's declared rho stays
primary; rho_vs_recorded = first call / f_recorded is printed beside it, and a disagreement is reported, not resolved.

Usage: python readings.py RESULTS_JSON  -> prints the readings, writes readings.json beside it."""
import json, sys
from pathlib import Path

READER = {"coal": "coal", "iron": "iron", "drink": "drink", "risky": "attack", "adjacent": "attack"}
MINING_DRINK = ("coal", "iron", "wood", "drink")


def below(x, thr, k=3):
    return x["est"] + k * x["se"] < thr


def arm(r, name):
    ap = r.get(name, {}).get("ap") if isinstance(r.get(name), dict) else None
    return None if ap is None else ("accessible" if ap >= 0.9 else "not accessible" if ap <= 0.7 else "partial"), ap


def r1(readers, pop):
    r = readers.get(pop, {})
    if "skipped" in r or not r:
        return {"population": pop, "skipped": r.get("skipped", "absent")}
    out = {"population": pop, "rows": r["rows"], "rate": r["rate"], "rule_ap": r["d_rule_inputs"]["ap"],
           "qualified": r["d_rule_inputs"]["ap"] >= 0.95}
    for k in ("a_full_conditioning", "b_belief_action_fixation", "c_raw_frame_action", "e_embedding_action",
              "f_coarse_history_action"):
        out[k] = arm(r, k)
    s = lambda k: out[k][0]
    out["H1_consistent"] = s("c_raw_frame_action") == "accessible" and s("e_embedding_action") == "not accessible"
    out["H2_consistent"] = s("e_embedding_action") == "accessible" and s("b_belief_action_fixation") == "not accessible"
    out["measurable"] = max(out["b_belief_action_fixation"][1], out["c_raw_frame_action"][1]) >= 0.95
    return out


def part_readings(c, p, first, kind):
    x = c["parts"][p]
    levels = sorted((float(s), v) for s, v in x["field"].items())
    sd2 = c["geometry"][p]["sigma_d2"]
    under = [s for s, v in levels if below(v["bias"], -0.05)]
    at_first = min(levels, key=lambda sv: abs(sv[0] - first))[1]["bias"]
    rho = x["first_call"]["est"] / x["exact_two_outcome_first_call"]["est"] if x["exact_two_outcome_first_call"]["est"] else None
    smp = x["sampler"]
    return {
        "R0": {"bias_0.05": levels[0][1]["bias"]["est"], "bias_0.05_ok": abs(levels[0][1]["bias"]["est"]) < 0.01,
               "rerun": x["rerun_control"]["est"], "rerun_ok": abs(x["rerun_control"]["est"]) < 0.01},
        "R2": {"first_call": x["first_call"], "exact": x["exact_two_outcome_first_call"]["est"], "rho": rho,
               "verdict": None if rho is None else "exact-like" if rho >= 0.8 else "lacks" if rho <= 0.5 else "between",
               "rho_vs_recorded": x["first_call"]["est"] / x["f_recorded"]["est"] if c["p_class"] == 1.0 else None},
        "R5": {"sigma_d2": sd2, "lowest_underweighted_level": under[0] if under else None,
               "underweighted_at_or_above_sigma_d2": any(s >= sd2 for s in under),
               "calibrated_below_sigma_d2": not any(s < sd2 for s in under),
               "bias_at_first_level": at_first, "calibrated_at_first_level": abs(at_first["est"]) < 0.05,
               "bias_by_level": {round(s, 4): round(v["bias"]["est"], 4) for s, v in levels}},
        "R3_descriptive": x["real_plus_noise_minus_pure_at_first_level"],
        "R4": {"sampler": smp, "verdict": "loses (N2)" if below(smp, -0.1) else
               "deficit is the first estimate's" if abs(smp["est"]) <= 0.05 else "between"},
        "R6_descriptive": {"noop_first_call": x["noop"]["first_call"], "action_contrast": x["noop"]["action_contrast"]},
        "R7_descriptive": x["context_sigma_5_stress"]["contrast_change"],
        "production": x["production"], "f_recorded": x["f_recorded"],
    }


def interpret(r, b):
    """PLAN.md "Interpretation, declared now"; b = reader (b)'s R1 state or None (no population)."""
    R2, R5, R4 = r["R2"]["verdict"], r["R5"], r["R4"]["verdict"]
    out = []
    if b == "accessible" and R5["underweighted_at_or_above_sigma_d2"] and R5["calibrated_below_sigma_d2"] and R2 == "lacks":
        out.append("N1-consistent")
    if R5["calibrated_at_first_level"] and R2 == "lacks":
        out.append("start-distribution")
    if R2 == "exact-like" and R4 == "loses (N2)":
        out.append("N2")
    if R2 == "exact-like" and R4 == "deficit is the first estimate's":
        out.append("carried to the samples (R2 >= 0.8, R4 within 0.05): a readout mismatch only if the samples' "
                   "readers show low rates; check before reading anything else")
    return out or ["not localised"]


def main():
    path = Path(sys.argv[1])
    d = json.loads(path.read_text())
    first, readers = d["first_level"], d.get("readers", {})
    res = {"checkpoint": d["checkpoint"], "checkpoint_sha256": d["checkpoint_sha256"], "episodes": d["episodes"],
           "first_level": first, "R1": {pop: r1(readers, pop) for pop in readers}, "classes": {}}
    safe = d["classes"].get("safe")
    if safe:
        x = safe["parts"]["hud"]
        vals = [v["mean"]["est"] for s, v in x["field"].items() if float(s) <= first + 1e-9]
        vals += [x["first_call"]["est"], x["production"]["est"]]
        res["R0_safe"] = {"max_abs_f": max(map(abs, vals)), "ok": max(map(abs, vals)) < 0.05}
    for kind, c in d["classes"].items():
        if kind in ("safe", "random"):
            continue
        rk = {"contexts": c["contexts"], "rows_valid": c["rows_valid"], "p_class": c["p_class"], "parts": {}}
        if kind in MINING_DRINK:
            rk["R0_recorded"] = c["recorded_reached_end"]["est"]
            rk["R0_recorded_ok"] = c["recorded_reached_end"]["est"] >= 0.99
        pop = READER.get(kind)
        b = res["R1"].get(pop, {}).get("b_belief_action_fixation", (None,))[0] if pop else None
        gate = res["R1"].get(pop, {}).get("measurable") if kind in ("coal", "iron", "drink") else None
        rk["reader_b"], rk["measurable_gate"] = b, gate
        for p in c["parts"]:
            r = part_readings(c, p, first, kind)
            r["interpretation"] = interpret(r, b)
            if gate is False:
                r["interpretation"] = ["descriptive only (R0 measurability gate failed)"] + r["interpretation"]
            rk["parts"][p] = r
        res["classes"][kind] = rk
    res["R8_descriptive"] = {k: c["objective"] for k, c in d["classes"].items()}
    (path.parent / "readings.json").write_text(json.dumps(res, indent=1))
    print("R0 safe", res.get("R0_safe"))
    for pop, r in res["R1"].items():
        print("R1", pop, {k: v for k, v in r.items() if k not in ("population",)})
    for kind, rk in res["classes"].items():
        for p, r in rk["parts"].items():
            f = lambda x: f"{x['est']:.3f}+-{x['se']:.3f}"
            print(f"{kind:9s} {p:4s} rec {f(r['f_recorded'])} | R2 first {f(r['R2']['first_call'])} rho "
                  f"{r['R2']['rho']:.2f} {r['R2']['verdict']} (vs rec {r['R2']['rho_vs_recorded'] or float('nan'):.2f}) | R5 sd2 {r['R5']['sigma_d2']:.2f} lowest-under "
                  f"{r['R5']['lowest_underweighted_level']} cal<sd2 {r['R5']['calibrated_below_sigma_d2']} first-bias "
                  f"{r['R5']['bias_at_first_level']['est']:.3f} | R4 {f(r['R4']['sampler'])} {r['R4']['verdict']} | "
                  f"prod {f(r['production'])} | R0 {r['R0']['bias_0.05_ok']},{r['R0']['rerun_ok']},"
                  f"{rk.get('R0_recorded_ok')} gate {rk['measurable_gate']} b {rk['reader_b']} -> {r['interpretation']}")


if __name__ == "__main__":
    main()
