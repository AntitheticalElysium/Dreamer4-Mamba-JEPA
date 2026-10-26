"""A3 analysis (protocol in eda_outofview_belief.py): belief shape and predictive loss of summaries by kind and gap.
Usage: ana_outofview_belief.py ROWS_JSON OUT_JSON
"""
import collections, json, sys
import numpy as np

BINS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 16))
LL = ("ll_exact", "ll_gauss", "ll_gauss_trunc", "ll_k4", "ll_k16")


def summarise(rows):
    if not rows:
        return None
    a = lambda k: np.array([r[k] for r in rows], float)
    out = {"n": len(rows), "ncells_median": float(np.median(a("ncells"))), "ncells_p90": float(np.percentile(a("ncells"), 90)),
           "spread_mean": float(a("spread").mean()), "split_support": float((a("components") > 1).mean()),
           "removed_frac_mean": float(a("removed_frac").mean()), "p_exist_mean": float(a("p_exist").mean()),
           "miss_rate": float((a("ll_exact") < np.log(2e-5)).mean())}
    for k in LL:
        out[k] = float(a(k).mean())
    for k in LL[1:]:
        out["loss_" + k[3:]] = float((a("ll_exact") - a(k)).mean())      # nats lost vs the exact belief
    return out


def main():
    rows = json.load(open(sys.argv[1]))
    res = {"events": dict(collections.Counter(f"{r['kind']}|{r['event']}" for r in rows))}
    for kind in ("zombies", "cows"):
        out = [r for r in rows if r["kind"] == kind and r["event"] == "out"]
        res[kind] = {"all": summarise(out),
                     "day": summarise([r for r in out if r["light"] >= 0.5]),
                     "night": summarise([r for r in out if r["light"] < 0.5]),
                     "by_gap": {f"{lo}-{hi}": summarise([r for r in out if lo <= r["k"] <= hi]) for lo, hi in BINS}}
        re = [r for r in rows if r["kind"] == kind and r["event"] == "reentry"]
        ks = np.array([r["k"] for r in re])
        res[kind]["reentry"] = {"n": len(re), "gap_median": float(np.median(ks)) if len(ks) else None,
                                "gap_p90": float(np.percentile(ks, 90)) if len(ks) else None,
                                "ll_mean": float(np.mean([r["ll_reentry"] for r in re])) if re else None,
                                "p_visible_mean": float(np.mean([r["p_visible"] for r in re])) if re else None}
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    for kind in ("zombies", "cows"):
        a = res[kind]["all"]
        print(kind, {k: round(v, 3) if isinstance(v, float) else v for k, v in a.items()})
        for g, s in res[kind]["by_gap"].items():
            if s:
                print(f"  gap {g:5s} n={s['n']:6d} cells~{s['ncells_median']:.1f} split={s['split_support']:.3f} "
                      f"removed={s['removed_frac_mean']:.3f} loss gauss {s['loss_gauss']:.3f} trunc {s['loss_gauss_trunc']:.3f} "
                      f"k4 {s['loss_k4']:.3f} k16 {s['loss_k16']:.3f} miss {s['miss_rate']:.3f}")
        print("  reentry", res[kind]["reentry"])
    print(res["events"])


if __name__ == "__main__":
    main()
