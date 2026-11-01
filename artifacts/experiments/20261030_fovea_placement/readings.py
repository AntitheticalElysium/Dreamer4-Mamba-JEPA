"""The declared readings of PLAN.md applied to evaluate.py's results.json (written before the full evaluation was read).
Per deterministic class (drink, wood, table), on the primary route (periphery) and subset (eye elsewhere), from A's
(the reference's) own values:
  MEI_p = 0.25 x (recorded projection - A's field projection at 6.67); MEI_r = 0.25 x (recorded rate - A's reader
  production success).
  POSITIVE: field difference (B - A) >= MEI_p with its 95% interval above 0, and the reader production difference's
  interval above 0. BOUNDED NULL: field interval's upper end < MEI_p and reader interval's upper end < MEI_r.
  Otherwise INCONCLUSIVE (also when a reading is missing).
Failed-attempt gate (drink, table): upper end of the false-change difference's interval < 0.05, on the same route and
subset (the primary readout's); the other routes and subsets are reported beside it, not gated.
Stage verdict: POSITIVE if >= 2 classes POSITIVE and the gate holds for both strata; NULL if all three BOUNDED NULL;
otherwise INCONCLUSIVE.
Usage: python -m artifacts.experiments.20261030_fovea_placement.readings RESULTS_JSON A B   (e.g. base guided_1500)
"""
import json, sys
from pathlib import Path

CLASSES, FAILS, GATE = ("drink", "wood", "table"), ("drink_fail", "table_fail"), 0.05


def main():
    path, a, b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    res = json.loads(path.read_text())
    ref = res["checkpoints"][a]["periphery"]
    diff = res["differences"][f"{b}-{a}"]
    out = {"a": a, "b": b, "classes": {}, "failed_attempts": {}}
    for k in CLASSES:
        s, d = ref[k]["eye_elsewhere"], diff["periphery"][k]["eye_elsewhere"]
        mei_p = 0.25 * (s["f_recorded"]["est"] - s["field"]["6.67"]["est"])
        mei_r = 0.25 * (s["recorded_end"]["est"] - s["reader_production_end"]["est"])
        f, r = d.get("field_6.67"), d.get("reader_production")
        if f is None or r is None:
            verdict = "INCONCLUSIVE (missing reading)"
        elif f["est"] >= mei_p and f["ci95"][0] > 0 and r["ci95"][0] > 0:
            verdict = "POSITIVE"
        elif f["ci95"][1] < mei_p and r["ci95"][1] < mei_r:
            verdict = "BOUNDED NULL"
        else:
            verdict = "INCONCLUSIVE"
        out["classes"][k] = {"mei_p": mei_p, "mei_r": mei_r, "field_6.67": f, "reader_production": r,
                             "rows": s["rows"], "rows_read": s["rows_read"], "rows_excluded": s["rows_excluded"],
                             "verdict": verdict}
    for k in FAILS:
        g = diff["periphery"][k]["eye_elsewhere"]["reader_production"]
        out["failed_attempts"][k] = {"gate": g, "holds": g is not None and g["ci95"][1] < GATE,
                                     "reported": {route: {sub: (v or {}).get("reader_production")
                                                          for sub, v in diff[route][k].items()} for route in diff}}
    n_pos = sum(c["verdict"] == "POSITIVE" for c in out["classes"].values())
    gate = all(f["holds"] for f in out["failed_attempts"].values())
    out["stage"] = ("POSITIVE" if n_pos >= 2 and gate else
                    "NULL" if all(c["verdict"] == "BOUNDED NULL" for c in out["classes"].values()) else "INCONCLUSIVE")
    (path.parent / f"readings_{b}-{a}.json").write_text(json.dumps(out, indent=1))
    for k, c in out["classes"].items():
        f, r = c["field_6.67"] or {}, c["reader_production"] or {}
        print(k, c["verdict"], f"MEI_p {c['mei_p']:.3f} field {f.get('est', float('nan')):+.3f} {f.get('ci95')}",
              f"| MEI_r {c['mei_r']:.3f} reader {r.get('est', float('nan')):+.3f} {r.get('ci95')}")
    for k, f in out["failed_attempts"].items():
        print(k, "gate holds" if f["holds"] else "gate FAILS", f["gate"])
    print("stage", out["stage"])


if __name__ == "__main__":
    main()
