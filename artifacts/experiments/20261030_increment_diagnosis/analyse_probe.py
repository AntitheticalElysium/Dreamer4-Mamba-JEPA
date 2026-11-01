"""Readings of increment_probe.py (PLAN.md Part 1). Per variant and row group (appearance v = 0, increment v >= 1, and
by v): read-value rates relative to the shown start u (u + 1 = success, u = copy, u - 1), the true start v and end
v + 1, the appearance's "1", off-manifold (> OFF spreads); the field at sigma 20 on the shown axis. Paired contrasts
(same rows and draws; episode bootstrap, 95% intervals): B1 - base, B2 - base_rerun, C_*_den - C_identity_den,
C_*_full - C_identity_full, C_identity_den - base, C_identity_full - base_rerun. MOVE: shift projection of the sample
and of the field.
Usage: python -m artifacts.experiments.20261030_increment_diagnosis.analyse_probe PROBE_DIR
"""
import json, sys
from pathlib import Path

import numpy as np

from saccade.eval import gates as G

OFF = 3.0
REF = {"base_rerun": "base", "B1_noop_den": "base", "B2_noop_full": "base_rerun", "C_identity_den": "base",
       "C_identity_full": "base_rerun"}


def load(d, kind, v):
    f = d / f"{kind}_{v}.npz"
    if not f.exists():
        return None
    o = dict(np.load(f))
    rep = np.zeros(len(o["ctx"]), int)
    for i in range(1, len(rep)):
        rep[i] = rep[i - 1] + 1 if o["ctx"][i] == o["ctx"][i - 1] else 0
    o["key"] = o["ctx"] * 16 + rep
    return o


def measures(o, K):
    v = np.array(K["start_value"])[o["ctx"]]
    r, u = o["read"], o["shown"]
    distinct = (u >= 2) & (v >= 2)                                       # "1" is then neither start nor end, shown or true
    return {"success": r == u + 1, "copy": r == u, "down": r == u - 1, "one": np.where(distinct, r == 1, np.nan),
            "true_start": r == v, "true_end": r == v + 1, "off": o["off"] > OFF, "field": o["field"]}, v


def est(x, m, ep):
    x = np.asarray(x, float); m = m & np.isfinite(x)
    if not m.any():
        return None
    r = G.estimate(np.where(m, x, 0.0), m, ep)
    return {"est": round(r["est"], 4), "se": round(r["se"], 4), "n": int(m.sum())}


def main():
    d = Path(sys.argv[1])
    res = json.loads((d / "results.json").read_text())
    out = {}
    for kind, K in res["kinds"].items():
        ep_of = np.array(K["episode"])
        O = {v: load(d, kind, v) for v in K["variants"]}
        R = {"rerun_bp_max_abs_diff": K["rerun_bp_max_abs_diff"], "variants": {}, "contrasts": {}}
        if kind == "move":
            for vn, o in O.items():
                ep = ep_of[o["ctx"]]
                R["variants"][vn] = {"shift_sample": est(o["read"], np.ones(len(ep), bool), ep),
                                     "shift_field": est(o["field"], np.ones(len(ep), bool), ep)}
            for vn, ref in REF.items():
                if vn in O and ref in O:
                    a, b = O[ref], O[vn]
                    assert (a["key"] == b["key"]).all()
                    ep = ep_of[a["ctx"]]
                    R["contrasts"][f"{vn}-{ref}"] = {k: paired(a[k], b[k], np.ones(len(ep), bool), ep) for k in ("read", "field")}
            out[kind] = R
            continue
        for vn, o in O.items():
            ms, v = measures(o, K)
            ep = ep_of[o["ctx"]]
            groups = {"appearance": v == 0, "increment": v >= 1, **{f"v={k}": v == k for k in np.unique(v)}}
            R["variants"][vn] = {g: {k: est(x, m, ep) for k, x in ms.items()} for g, m in groups.items() if m.any()}
        pairs = dict(REF)
        for vn in O:
            if vn.startswith("C_") and not vn.startswith("C_identity"):
                pairs[vn] = "C_identity_" + vn.rsplit("_", 1)[1]
        for vn, ref in pairs.items():
            if vn not in O or ref not in O:
                continue
            b = O[vn]
            a = {k: x[np.isin(O[ref]["key"], b["key"])] for k, x in O[ref].items()}
            assert (a["key"] == b["key"]).all(), (vn, ref)
            ma, v = measures(a, K); mb, _ = measures(b, K)
            ep = ep_of[b["ctx"]]
            R["contrasts"][f"{vn}-{ref}"] = {g: {k: paired(ma[k], mb[k], m, ep) for k in ma}
                                             for g, m in (("appearance", v == 0), ("increment", v >= 1)) if m.any()}
        out[kind] = R
    (d / "readings.json").write_text(json.dumps(out, indent=1))
    show(out)


def paired(a, b, m, ep):
    dlt = np.asarray(b, float) - np.asarray(a, float)
    m = m & np.isfinite(dlt)
    if not m.any():
        return None
    r = G.estimate(np.where(m, dlt, 0.0), m, ep)
    return {"est": round(r["est"], 4), "ci95": [round(r["est"] - 1.96 * r["se"], 4), round(r["est"] + 1.96 * r["se"], 4)],
            "n": int(m.sum())}


def show(out):
    f = lambda r: "  -  " if r is None else f"{r['est']:.3f}"
    for kind, R in out.items():
        print(f"\n===== {kind}  (rerun |bp| diff {R['rerun_bp_max_abs_diff']:.3f})")
        if kind == "move":
            for vn, r in R["variants"].items():
                print(f"  {vn:16s} sample {f(r['shift_sample'])} field {f(r['shift_field'])}")
            continue
        for vn, G_ in R["variants"].items():
            for g in ("appearance", "increment"):
                if g in G_:
                    r = G_[g]
                    print(f"  {vn:18s} {g:10s} n {r['success']['n']:5d} success {f(r['success'])} copy {f(r['copy'])} "
                          f"down {f(r['down'])} one {f(r['one'])} true_v {f(r['true_start'])} true_v+1 {f(r['true_end'])} "
                          f"off {f(r['off'])} field {f(r['field'])}")
        for c, G_ in R["contrasts"].items():
            for g, r in G_.items():
                sel = {k: r[k] for k in ("success", "copy", "one", "true_start", "field") if r.get(k)}
                print(f"  {c:38s} {g:10s} " + " ".join(f"{k} {v['est']:+.3f}[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]" for k, v in sel.items()))


if __name__ == "__main__":
    main()
