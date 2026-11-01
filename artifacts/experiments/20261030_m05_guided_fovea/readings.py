"""The declared readings of PLAN.md applied to evaluate.py's results.json: classification only, no gates (user, 10-30:
the control arm is decided by hand after dissecting the M0.5 arm). For a comparison B - A (guided_1500 - base:
descriptive, continued joint training and placement are not separated; guided_1500 - random_1500: attribution):
- PREDICTION (glimpse primary, periphery secondary), per class group, over the input-defined population (UNKNOWN
  generations are not verified successes): the verified-success difference with its 95% interval and MEI = 0.25 x
  (A's recorded rate - A's verified success): POSITIVE (est >= MEI, lower > 0), DETECTED (lower > 0, est < MEI),
  NEGATIVE (upper < 0), BOUNDED NULL (holds 0, upper < MEI), INCONCLUSIVE (holds 0, upper >= MEI). Flag UNKNOWN-HEAVY
  (declared before any M0.5 reading): either checkpoint's unknown share > 0.10, or the unknown difference's interval
  excludes 0; the verified comparison may then be biased and the bounds and the among-readable diagnostic must be read
  with it. Groups: drink / wood appearance, table (fixed outputs), drink / wood increment; sapling and damage are
  stochastic (calibration: generated beside recorded rate). Failed attempts: verified false change MORE / LESS / NO
  CHANGE. Beside each: the instrument on the rows' real tokens against the observed value, and glimpse-periphery
  coherence.
- PERCEPTION (monitored): BETTER / WORSE / NO CHANGE by the NLL difference's interval.
- ACTION ROUTING (diagnostic, not memory content): MORE / LESS / NO CHANGE; B's own routing beside it.
- REGRESSION PANEL: reward NLL, continuation BCE, TSP NLL (per-frame estimates; not paired), movement's shift (paired),
  entity count gaps (generated - real), belief probe (departed AP by stratum, HUD exact), reader qualification lost.
Usage: python -m artifacts.experiments.20261030_m05_guided_fovea.readings RESULTS_JSON A B
"""
import json, sys
from pathlib import Path

PRED = {"drink": ("appearance", "increment"), "wood": ("appearance", "increment"), "table": ("all",)}
CAL = ("sapling", "adjacent", "risky")
UNKNOWN_MAX = 0.10


def cls_pred(d, mei):
    if d is None or mei is None:
        return "UNMEASURED"
    lo, hi = d["ci95"]
    if lo > 0:
        return "POSITIVE" if d["est"] >= mei else "DETECTED"
    if hi < 0:
        return "NEGATIVE"
    return "BOUNDED NULL" if hi < mei else "INCONCLUSIVE"


def sign(d, better_if_negative=False):
    if d is None:
        return "UNMEASURED"
    lo, hi = d["ci95"]
    if better_if_negative:
        return "BETTER" if hi < 0 else "WORSE" if lo > 0 else "NO CHANGE"
    return "MORE" if lo > 0 else "LESS" if hi < 0 else "NO CHANGE"


v = lambda x: None if x is None else x["est"]


def main():
    path, a, b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    res = json.loads(path.read_text())
    A, B, D = res["checkpoints"][a], res["checkpoints"][b], res["differences"][f"{b}-{a}"]
    out = {"a": a, "b": b, "prediction": {}, "calibration": {}, "failed_attempts": {}, "perception": {},
           "action_routing": {}, "regression": {}}
    for route in ("guided", "periphery"):
        out["prediction"][route] = {}
        for kind, gs in PRED.items():
            for g in gs:
                sa = A["prediction"][route].get(kind, {}).get(g) or {}
                sb = B["prediction"][route].get(kind, {}).get(g) or {}
                dd = D["prediction"][route].get(kind, {}).get(g) or {}
                d = dd.get("success")
                mei = 0.25 * (v(sa["recorded"]) - v(sa["success"])) if sa.get("recorded") and sa.get("success") else None
                du = dd.get("unknown")
                heavy = any(x is not None and x > UNKNOWN_MAX for x in (v(sa.get("unknown")), v(sb.get("unknown")))) or \
                    (du is not None and (du["ci95"][0] > 0 or du["ci95"][1] < 0))
                out["prediction"][route][f"{kind}_{g}"] = {
                    "a": sa.get("success"), "b": sb.get("success"), "difference": d, "mei": mei, "reading": cls_pred(d, mei),
                    "unknown_heavy": bool(heavy), "unknown": [v(sa.get("unknown")), v(sb.get("unknown"))], "unknown_difference": du,
                    "upper_bounds": [v(sa.get("success_upper_bound")), v(sb.get("success_upper_bound"))],
                    "among_readable_difference": dd.get("success_among_readable"),
                    "real_token_correct": [v(sa.get("real_token_correct")), v(sb.get("real_token_correct"))],
                    "real_token_unreadable": [v(sa.get("real_token_unreadable")), v(sb.get("real_token_unreadable"))],
                    "coherent": [v(sa.get("glimpse_periphery_coherent")), v(sb.get("glimpse_periphery_coherent"))]}
        out["calibration"][route] = {k: {"recorded": v((A["prediction"][route].get(k, {}).get("all") or {}).get("recorded")),
                                         "a": v((A["prediction"][route].get(k, {}).get("all") or {}).get("success")),
                                         "b": v((B["prediction"][route].get(k, {}).get("all") or {}).get("success"))} for k in CAL}
        out["failed_attempts"][route] = {}
        for k in ("drink_fail", "table_fail"):
            dd = D["prediction"][route].get(k, {}).get("all") or {}
            out["failed_attempts"][route][k] = {"difference": dd.get("false_change"), "reading": sign(dd.get("false_change")),
                                                "unknown_difference": dd.get("unknown")}
    for kind, gd in D["perception"].items():
        for g, m in gd.items():
            out["perception"][f"{kind}_{g}"] = {k: {"difference": x, "reading": sign(x, better_if_negative=True)} for k, x in m.items()}
    for kind, gd in D["action_routing"].items():
        for g, m in gd.items():
            out["action_routing"][f"{kind}_{g}"] = {"b_routing": (B["action_routing"].get(kind, {}).get(g) or {}).get("routing"),
                                                    "change": m["routing"], "reading": sign(m["routing"])}
    ra, rb = A["regression"], B["regression"]
    out["regression"] = {k: {"a": ra[k], "b": rb[k], "difference": rb[k]["est"] - ra[k]["est"]} for k in ra}
    out["regression"]["movement_shift"] = D["prediction"]["periphery"].get("move", {}).get("shift_sample")
    if "s1" in A and "s1" in B:
        gap = lambda c: {k: x["generated"]["est"] - x["real"]["est"] for k, x in c["entity_counts"].items()}
        out["regression"]["entity_gap_generated_minus_real"] = {"a": gap(A["s1"]), "b": gap(B["s1"])}
        dep = lambda c: {k: x["ap_by_stratum"] for k, x in c["belief_probe"]["departed"].items()}
        hud = lambda c: sum(x["est"] for x in c["belief_probe"]["hud_exact"]) / len(c["belief_probe"]["hud_exact"])
        out["regression"]["belief_departed_ap"] = {"a": dep(A["s1"]), "b": dep(B["s1"])}
        out["regression"]["belief_hud_exact_mean"] = {"a": hud(A["s1"]), "b": hud(B["s1"])}
    out["regression"]["qualification_lost"] = sorted(
        k for part in ("p", "g") for k, x in A["instrument_qualification"][part].items()
        if x.get("qualified") and not B["instrument_qualification"][part].get(k, {}).get("qualified"))
    (path.parent / f"readings_{b}-{a}.json").write_text(json.dumps(out, indent=1))
    for route, r in out["prediction"].items():
        for k, x in r.items():
            d = x["difference"]
            print(f"{route:9s} {k:22s} {x['reading']:13s}{' UNKNOWN-HEAVY' if x['unknown_heavy'] else ''} " +
                  ("" if d is None else f"{d['est']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] MEI {x['mei']:.3f}") +
                  f"  unknown {x['unknown']}  real-token {x['real_token_correct']}")
    for k, x in out["perception"].items():
        print("perception", k, {m: y["reading"] for m, y in x.items()})
    for k, x in out["action_routing"].items():
        print("action routing", k, x["reading"])
    for route, r in out["failed_attempts"].items():
        print("failed", route, {k: x["reading"] for k, x in r.items()})
    print("regression", json.dumps({k: (round(x["difference"], 4) if isinstance(x, dict) and "difference" in x else x)
                                    for k, x in out["regression"].items() if k not in ("belief_departed_ap",)}, default=str)[:900])


if __name__ == "__main__":
    main()
