"""The declared readings of PLAN.md applied to evaluate.py's results.json: classification only, no gates (user, 10-30:
the control arm is decided by hand after dissecting the M0.5 arm). For a comparison B - A (e.g. guided_1500 - base:
descriptive, continued joint training and placement are not separated; guided_1500 - random_1500: attribution):
- PREDICTION (glimpse primary, periphery secondary), per class group, success difference with its 95% interval and
  MEI = 0.25 x (A's recorded rate - A's success): POSITIVE (est >= MEI, lower > 0), DETECTED (lower > 0, est < MEI),
  NEGATIVE (upper < 0), BOUNDED NULL (interval holds 0, upper < MEI), INCONCLUSIVE (interval holds 0, upper >= MEI).
  Groups: drink / wood appearance, table (fixed outputs), drink / wood increment, sapling and damage (stochastic:
  calibration, the generated rate beside the recorded one, not classified). Failed attempts: false change, MORE if the
  interval is above 0, LESS if below, else NO CHANGE. Beside each: the same instrument on the rows' REAL tokens
  (must reproduce the recorded outcome) and the glimpse-periphery coherence of the generated frame.
- PERCEPTION (fovea reconstruction and TSP pixel NLL, cell and window): BETTER (upper < 0), WORSE (lower > 0), else NO
  CHANGE.
- MEMORY RELIANCE: the change of reliance, MORE / LESS / NO CHANGE by its interval; B's own reliance beside it.
- REGRESSION PANEL: differences of reward NLL, continuation BCE, TSP NLL (per-frame estimates; their intervals are not
  paired), movement's shift (paired), entity count gaps (generated - real) and reader qualification, listed.
Usage: python -m artifacts.experiments.20261030_m05_guided_fovea.readings RESULTS_JSON A B
"""
import json, sys
from pathlib import Path

PRED = {"drink": ("appearance", "increment"), "wood": ("appearance", "increment"), "table": ("all",)}
CAL = {"sapling": ("all",), "adjacent": ("all",), "risky": ("all",)}


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


def main():
    path, a, b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    res = json.loads(path.read_text())
    A, B, D = res["checkpoints"][a], res["checkpoints"][b], res["differences"][f"{b}-{a}"]
    out = {"a": a, "b": b, "prediction": {}, "calibration": {}, "failed_attempts": {}, "perception": {}, "memory": {}, "regression": {}}
    for route in ("guided", "periphery"):
        out["prediction"][route] = {}
        for kind, gs in PRED.items():
            for g in gs:
                s = A["prediction"][route].get(kind, {}).get(g)
                d = (D["prediction"][route].get(kind, {}).get(g) or {}).get("success")
                mei = (0.25 * (s["recorded"]["est"] - s["success"]["est"])
                       if s and s.get("recorded") and s.get("success") else None)
                sb = B["prediction"][route].get(kind, {}).get(g) or {}
                out["prediction"][route][f"{kind}_{g}"] = {"a": (s or {}).get("success"), "b": sb.get("success"),
                                                          "difference": d, "mei": mei, "reading": cls_pred(d, mei),
                                                          "real_token_correct": [(s or {}).get("real_token_correct"), sb.get("real_token_correct")],
                                                          "coherent": [(s or {}).get("glimpse_periphery_coherent"), sb.get("glimpse_periphery_coherent")]}
        out["calibration"][route] = {k: {"recorded": (A["prediction"][route].get(k, {}).get("all") or {}).get("recorded"),
                                         "a": (A["prediction"][route].get(k, {}).get("all") or {}).get("success"),
                                         "b": (B["prediction"][route].get(k, {}).get("all") or {}).get("success")} for k in CAL}
        out["failed_attempts"][route] = {k: {"difference": (D["prediction"][route].get(k, {}).get("all") or {}).get("false_change"),
                                             "reading": sign((D["prediction"][route].get(k, {}).get("all") or {}).get("false_change"))}
                                         for k in ("drink_fail", "table_fail")}
    for kind, gd in D["perception"].items():
        for g, m in gd.items():
            out["perception"][f"{kind}_{g}"] = {k: {"difference": v, "reading": sign(v, better_if_negative=True)} for k, v in m.items()}
    for kind, gd in D["memory"].items():
        for g, m in gd.items():
            out["memory"][f"{kind}_{g}"] = {"b_reliance": (B["memory"].get(kind, {}).get(g) or {}).get("reliance"),
                                            "change": m["reliance"], "reading": sign(m["reliance"])}
    ra, rb = A["regression"], B["regression"]
    out["regression"] = {k: {"a": ra[k], "b": rb[k], "difference": rb[k]["est"] - ra[k]["est"]} for k in ra}
    out["regression"]["movement_shift"] = D["prediction"]["periphery"].get("move", {}).get("shift_sample")
    if "entities" in A and "entities" in B:
        gap = lambda c: {k: v["generated"]["est"] - v["real"]["est"] for k, v in c["counts"].items()}
        out["regression"]["entity_gap_generated_minus_real"] = {"a": gap(A["entities"]), "b": gap(B["entities"])}
    out["regression"]["qualification_lost"] = sorted(
        k for part in ("p", "g") for k, v in A["instrument_qualification"][part].items()
        if v.get("qualified") and not B["instrument_qualification"][part].get(k, {}).get("qualified"))
    (path.parent / f"readings_{b}-{a}.json").write_text(json.dumps(out, indent=1))
    for route, r in out["prediction"].items():
        for k, v in r.items():
            d = v["difference"]
            rt = [None if x is None else round(x["est"], 3) for x in v["real_token_correct"]]
            print(f"{route:9s} {k:22s} {v['reading']:13s} " + ("" if d is None else
                  f"{d['est']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] MEI {v['mei']:.3f}") + f"  real-token check {rt}")
    for k, v in out["perception"].items():
        print("perception", k, {m: x["reading"] for m, x in v.items()})
    for k, v in out["memory"].items():
        print("memory", k, v["reading"])
    for route, r in out["failed_attempts"].items():
        print("failed", route, {k: v["reading"] for k, v in r.items()})
    print("regression", json.dumps({k: (round(v["difference"], 4) if isinstance(v, dict) and "difference" in v else v)
                                    for k, v in out["regression"].items()}, default=str)[:600])


if __name__ == "__main__":
    main()
