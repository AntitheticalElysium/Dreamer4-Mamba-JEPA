"""Codex's readings for this run, summarised with Codex's own estimators (followup/analyze.py: estimate, panel_summary)
and paired beta10 - p1f10 differences on the same scenes / contexts / draws (CPU).
Usage: python -m artifacts.experiments.20261101_beta_tsp.readings_summary
"""
import importlib, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/codex"
AN = importlib.import_module("artifacts.experiments.20261031_grounding_loss_review.followup.analyze")
A, B = "p1f10", "beta10"


def load(name, kind):
    return json.loads((OUT / f"{name}_{kind}.json").read_text())


def paired(fn, ra, rb, groups):
    return AN.estimate([fn(b) - fn(a) for a, b in zip(ra, rb)], groups)


def main():
    out = {"panel": {n: AN.panel_summary(OUT / f"{n}_panel.json") for n in (A, B)}, "panel_paired": {}}
    pa, pb = load(A, "panel")["rows"], load(B, "panel")["rows"]
    assert [(r["episode"], r["t"]) for r in pa] == [(r["episode"], r["t"]) for r in pb]
    fns = {"sensor_p": lambda r: r["sensor_variance"]["p"], "sensor_g": lambda r: r["sensor_variance"]["g"],
           "pixel_tsp": lambda r: r["pixel_mse"]["tsp"], "pixel_rec": lambda r: r["pixel_mse"]["rec"],
           "reward_nll_real": lambda r: r["reward_nll_real"],
           "generation_total": lambda r: r["generation_map_decomposition"]["total_pairwise_mse"],
           "generation_mean_mismatch": lambda r: r["generation_map_decomposition"]["mean_mismatch"],
           "generated_variance": lambda r: r["generation_map_decomposition"]["generated_variance"]}
    for stratum in ("bright", "dark"):
        ids = [i for i, r in enumerate(pa) if r["stratum"] == stratum]
        out["panel_paired"][stratum] = {k: paired(f, [pa[i] for i in ids], [pb[i] for i in ids], [pa[i]["episode"] for i in ids])
                                        for k, f in fns.items()}
        den = np.mean([fns["sensor_p"](pa[i]) for i in ids])                # 0 by day: the renderer adds static only at night
        out["panel_paired"][stratum]["sensor_p_ratio"] = float(np.mean([fns["sensor_p"](pb[i]) for i in ids]) / den) if den > 0 else None
    ma, mb = load(A, "movement"), load(B, "movement")
    assert ma["contexts"] == mb["contexts"] and ma["row_ctx"] == mb["row_ctx"]
    ok = np.asarray(ma["valid"], bool) & np.asarray(mb["valid"], bool)
    eps = np.asarray([ma["contexts"][i][0] for i in ma["row_ctx"]])
    out["movement"] = {n: {k: AN.estimate(np.asarray(m[k])[ok], eps[ok]) for k in ("f_real", "f_generated")} for n, m in ((A, ma), (B, mb))}
    res = (np.array(mb["f_generated"]) - np.array(mb["f_real"])) - (np.array(ma["f_generated"]) - np.array(ma["f_real"]))
    out["movement"]["paired_residual_beta_minus_p1f"] = AN.estimate(res[ok], eps[ok])
    out["counters"], out["noop_counters"] = {}, {}
    for n in (A, B):
        d, o = load(n, "counters"), {}
        for gaze, rr in d.items():
            o[gaze] = {}
            for kind in ("wood", "drink"):
                rows = [r for r in rr if r["kind"] == kind]
                g = [r["episode"] for r in rows]
                o[gaze][kind] = {k: AN.estimate([f(r) for r in rows], g) for k, f in
                                 [("verified_correct", lambda r: r["generated_supported"] and r["generated"] == r["truth"]),
                                  ("unknown", lambda r: not r["generated_supported"]),
                                  ("real_supported_correct", lambda r: r["real_supported"] and r["real_decoded"] == r["truth"]),
                                  ("tsp_pixel_mse", lambda r: r["tsp_pixel_mse"]), ("rec_pixel_mse", lambda r: r["rec_pixel_mse"])]}
        out["counters"][n] = o
        d, o = load(n, "noop_counters"), {}
        for gaze, rr in d.items():
            o[gaze] = {}
            for kind in ("wood", "drink"):
                rows = [r for r in rr if r["kind"] == kind]
                o[gaze][kind] = {k: AN.estimate([f(r) for r in rows], [r["episode"] for r in rows]) for k, f in
                                 [("verified_false_increase", lambda r: r["generated_supported"] and r["generated"] > r["start"]),
                                  ("unknown", lambda r: not r["generated_supported"])]}
        out["noop_counters"][n] = o
    out["lighting"] = {n: {k: float(np.mean([r[k] for r in load(n, "lighting")["rows"]])) for k in ("p", "g", "v")} for n in (A, B)}
    out["real_counters"] = {n: {g: {k: float(np.mean([r["read"] == r["truth"] for r in rr if r["kind"] == k])) for k in ("wood", "drink")}
                                for g, rr in load(n, "real_counters_cpu").items()} for n in (A, B)}
    sp = OUT / "sampler_results.json"
    if sp.exists():
        s = json.loads(sp.read_text())["models"]
        out["sampler"] = {}
        for n in s:
            c, m = s[n]["panels"]["counts"], s[n]["panels"]["move"]
            out["sampler"][n] = {"real_count": {k: v["est"] for k, v in c["real_count"].items()},
                                 **{alg: {"count": {k: c[alg][k]["count"]["est"] for k in c[alg]},
                                          "shift": m[alg]["shift"]["est"]} for alg in ("euler_3_churn_1", "euler_8_churn_0", "euler_32_churn_1")},
                                 "real_shift": m["real"]["est"]}
    (HERE / "readings/codex_summary.json").write_text(json.dumps(out, indent=1))
    pp = out["panel_paired"]
    for st in ("bright", "dark"):
        print(st, {k: (round(v["mean"], 5), [round(x, 5) for x in v["ci95"]]) if isinstance(v, dict) else round(v, 3) for k, v in pp[st].items()})
    print("movement", {n: {k: round(v["mean"], 4) for k, v in out["movement"][n].items()} for n in (A, B)},
          "paired", round(out["movement"]["paired_residual_beta_minus_p1f"]["mean"], 4), out["movement"]["paired_residual_beta_minus_p1f"]["ci95"])
    for n in (A, B):
        print(n, "DO", {g: {k: round(v["verified_correct"]["mean"], 3) for k, v in x.items()} for g, x in out["counters"][n].items()},
              "NOOP false+", {g: {k: round(v["verified_false_increase"]["mean"], 3) for k, v in x.items()} for g, x in out["noop_counters"][n].items()},
              "real", out["real_counters"][n], "lighting", {k: round(v, 5) for k, v in out["lighting"][n].items()})
    if "sampler" in out:
        print(json.dumps(out["sampler"], indent=1))


if __name__ == "__main__":
    main()
