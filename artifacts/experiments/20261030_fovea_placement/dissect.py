"""Per-row dissection of the stage-1 evaluation (user, 10-30: global metrics may mislead). For drink, wood, table
(sapling, move as references), base vs the guided checkpoints:
1. what the generated consequence cell holds (qualified reader): the consequence, the unchanged start value, other;
2. the field projection at every noise level of field_calibration.SIGMAS (where the consequence is lost);
3. churn: per row (same draws) and per context, base vs guided_1500;
4. by start value (HUD counters) or faced direction (table);
5. glimpse output with the eye on, dev G1 loss by class, failed attempts (from results.json).
Primary subset: periphery route, eye elsewhere; guided route, eye on, beside it. CPU only: reads EVAL_DIR's per-row
files and results.json. Rows are rebuilt from the dev labels as evaluate.main builds them (the context selection reads
the encoding only for episode lengths); the rebuild is checked against results.json's recorded rates.
Usage: python -m artifacts.experiments.20261030_fovea_placement.dissect [EVAL_DIR]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np

EV = importlib.import_module("artifacts.experiments.20261030_fovea_placement.evaluate")
CAT, FC = EV.CAT, EV.FC
LABELS = ("base", "guided_0300", "guided_0750", "guided_1500")
KINDS = ("drink", "wood", "table", "sapling", "move")
SUB = {"periphery": "eye_elsewhere", "guided": "eye_on"}


def rows_from_labels(L):
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    ctx = CAT.contexts(L, [{"p": range(T + 1)} for T in L["ep_T"]], np.random.default_rng(2031))[0]
    for kind, (cell, arr, col) in CAT.HUD.items():                      # evaluate.main's filter on hud_means' values
        x = L["s_body" if arr == 0 else "s_inv"]
        keys = set(np.unique(x[:, col]).astype(int).tolist())
        ctx[kind] = [r for r in ctx[kind] if {int(x[s_off[r[0]] + r[1] - 1, col]), int(x[s_off[r[0]] + r[1] - 1, col]) + 1} <= keys]
    return {k: ctx[k][:EV.N] for k in KINDS}


def rate(x, m):
    return None if not m.any() else round(float(np.asarray(x, float)[m].mean()), 4)


def main():
    ev = Path(sys.argv[1]) if len(sys.argv) > 1 else EV.HERE / "eval_stage1"
    res = json.loads((ev / "results.json").read_text())
    L = dict(np.load(EV.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    rows = rows_from_labels(L)
    out = {"sigmas": FC.SIGMAS, "kinds": {}}
    for kind in KINDS:
        info = EV.row_info(kind, rows[kind], L)
        K = {}
        for route, sub in SUB.items():
            z = {lab: dict(np.load(ev / f"{lab}_{route}_{kind}.npz")) for lab in LABELS
                 if (ev / f"{lab}_{route}_{kind}.npz").exists()}
            if not z:
                continue
            b = z["base"]
            idx = b["row_ctx"]
            m = b["valid"] & (b["eye_elsewhere"] if sub == "eye_elsewhere" else b["covered"])
            r = {"rows": int(m.sum()), "contexts": int(len(np.unique(idx[m])))}
            # rebuild check: recorded rate against results.json
            rec = np.array([info[i][3] for i in idx], float)
            js = res["checkpoints"]["base"][route][kind][sub]
            if "read_ok" in b:
                mr = m & b["read_ok"]
                r["rebuild_check"] = {"recorded_end": rate(rec, mr), "results_json": js["recorded_end"]["est"]}
                r["rows_read"] = int(mr.sum())
                r["outcome"] = {lab: {"consequence": rate(o["gen_end"], mr), "unchanged": rate(o["gen_start"], mr),
                                      "other": rate(~o["gen_end"] & ~o["gen_start"], mr),
                                      "real_end": rate(o["real_end"], mr)} for lab, o in z.items()}
                a1, b1 = b["gen_end"][mr], z["guided_1500"]["gen_end"][mr]
                r["churn_rows"] = {"both": int((a1 & b1).sum()), "base_only": int((a1 & ~b1).sum()),
                                   "guided_only": int((~a1 & b1).sum()), "neither": int((~a1 & ~b1).sum())}
                c = idx[mr]
                u, inv = np.unique(c, return_inverse=True)
                pa = np.bincount(inv, a1) / np.bincount(inv); pb = np.bincount(inv, b1) / np.bincount(inv)
                r["churn_contexts"] = {"contexts": int(len(u)), "base_never": float((pa == 0).mean()),
                                       "base_always": float((pa == 1).mean()), "guided_never": float((pb == 0).mean()),
                                       "guided_always": float((pb == 1).mean()),
                                       "corr_base_guided": float(np.corrcoef(pa, pb)[0, 1]) if pa.std() and pb.std() else None}
                key = (np.array([info[i][1] for i in idx]) if kind != "table" else np.array([info[i][0] for i in idx]))
                r["by_start_value" if kind != "table" else "by_faced_cell"] = {
                    int(v): {"rows": int((mr & (key == v)).sum()),
                             **{lab: {"consequence": rate(o["gen_end"], mr & (key == v)),
                                      "field_6.67": rate(o["f_den"][:, FC.SIGMAS.index(6.67)], m & (key == v))}
                                for lab, o in z.items() if lab in ("base", "guided_1500")}}
                    for v in np.unique(key[m])}
            r["field_by_sigma"] = {lab: [rate(o["f_den"][:, k], m) for k in range(len(FC.SIGMAS))] for lab, o in z.items()}
            r["f_real"] = rate(b["f_real"], m)
            r["f_production"] = {lab: rate(o["f_gen"], m) for lab, o in z.items()}
            K[route] = r
        g = {lab: res["checkpoints"][lab]["guided"][kind]["eye_on"] for lab in LABELS}
        K["glimpse_eye_on"] = {lab: {k: v[k] for k in ("glimpse_reader_rows", "glimpse_reader_production",
                                                       "glimpse_projection_rows", "glimpse_projection_production") if k in v}
                               for lab, v in g.items() if v}
        out["kinds"][kind] = K
    out["glimpse_projection_qualification"] = res["instrument_qualification"]["glimpse_projection"]
    out["class_losses"] = {lab: {k: round(v["se_per_element"], 5) for k, v in res["checkpoints"][lab]["costs"]["class_losses"].items()}
                           for lab in LABELS}
    out["entity_counts"] = {lab: res["checkpoints"][lab]["costs"]["entity_counts"] for lab in LABELS}
    out["failed_attempts"] = {k: {lab: {route: {s: (v or {}).get("reader_production_false_change")
                                                for s, v in res["checkpoints"][lab][route][k].items() if s != "covered_share"}
                                        for route in SUB} for lab in ("base", "guided_1500")}
                              for k in EV.FAIL_KINDS if k in res["rows"]}
    (ev / "dissect.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
