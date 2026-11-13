"""Round 2 tables (PLAN.md round 2) from readings/round2/*.json; the bar arithmetic is fixed here, before the results.

    python -m artifacts.experiments.20261113_token_forecast_gate.summarise2
"""
import importlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "readings/round2"
G = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.gate")
CK = (1, 2, 10)
f = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"
g = lambda c: f"{c[0]:+.2e} [{c[1]:+.2e}, {c[2]:+.2e}]"
excl = lambda c: c[1] > 0


def main():
    out = {}
    for k in CK:
        p = R / f"ckpt_{k:02d}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rep = d["round1_plain_reproduction"]
        print(f"\n=== checkpoint {k}  plain held-out now {rep['now']:.6f} vs round 1 {rep['round1']:.6f}")
        for cur in ("plain", "categorical"):
            h = d[f"heldout/{cur}"]
            print(f"  {cur:11s} held-out head {h['head']:.4f}  shuffled {h['shuffled_belief']:.4f}  ratio {h['shuffled_belief'] / h['head']:.2f}")
        for name in ("energy", "hunger", "recovery", "cooldown"):
            for cur in ("plain", "categorical"):
                s = d[f"{name}/{cur}"]
                print(f"  {name:8s} {cur:11s} foil>=cons {s['foil_beats_consequence']} (oracle {s['oracle_foil_beats_consequence']}) /{s['pairs']}"
                      f"  cos {f(s['cosine_mean_ci'])} ({s['cosine_positive']})  digit-cell cos {f(s['cosine_digit_cell_mean_ci'])}"
                      f"  |gsep| {s['grad_sep_norm_median']:.2e}  kappa {s['kappa_mean']:+.4f}")
            a = [e["cosine"] for e in d[f"{name}/categorical"]["per_pair"]]
            b = [e["cosine"] for e in d[f"{name}/plain"]["per_pair"]]
            print(f"  {name:8s} cosine categorical - plain {f(G.boot_mean(np.subtract(a, b)))}")
        print(f"  reward on recovery: cos {f(d['recovery/reward']['cosine_mean_ci'])} ({d['recovery/reward']['cosine_positive']})")
        for name in ("energy", "hunger", "recovery"):
            m = d[f"matched/{name}"]
            for cur in [c for c in m if c != "categorical_minus_plain"]:
                print(f"  matched {name:8s} {cur:11s} held-out dkappa {g(m[cur]['kappa_heldout_sgd']['mean_ci'])}  in-pair {g(m[cur]['kappa_in_pair_sgd']['mean_ci'])}"
                      f"  AdamW held-out {g(m[cur]['kappa_heldout']['mean_ci'])}  dlogD {f(m[cur]['log_distance_heldout']['mean_ci'])}")
            print(f"  matched {name:8s} categorical - plain held-out dkappa {g(m['categorical_minus_plain']['kappa_heldout_sgd'])}"
                  f"  in-pair {g(m['categorical_minus_plain']['kappa_in_pair_sgd'])}")
        ref = d["matched/recovery"]["reward"]
        ref_ok = {"heldout": excl(ref["kappa_heldout_sgd"]["mean_ci"]), "in_pair": excl(ref["kappa_in_pair_sgd"]["mean_ci"])}
        e = d["energy/categorical"]
        bars = {"1_nuisance": e["foil_beats_consequence"]["6"] < e["pairs"] / 2,
                "2_direction": excl(e["cosine_mean_ci"]),
                "3_forecaster": d["heldout/categorical"]["head"] < d["heldout/categorical"]["shuffled_belief"]}
        diff = d["matched/energy"]["categorical_minus_plain"]
        bars["4_adds_over_plain"] = (excl(diff["kappa_heldout_sgd"]) if ref_ok["heldout"] else
                                     excl(diff["kappa_in_pair_sgd"]) if ref_ok["in_pair"] else "uninformative (reference not positive)")
        print(f"  reference valid: {ref_ok}  bars (categorical, energy): {bars}")
        for key in ("hunger/none", "hunger/amplified", "hunger/sham", "energy/none", "energy/attenuated"):
            s = d[f"d2/{key}"]
            print(f"  D2 {key:18s} scale {s['scale_median']:.2f}  D {s['distance_median']:.2e}  plain cos {f(s['plain/cosine_mean_ci'])}"
                  f"  categorical cos {f(s['categorical/cosine_mean_ci'])}  digit-cell cos {f(s['plain/cosine_digit_cell_mean_ci'])}"
                  f"  |gsep| {s['plain/grad_sep_norm_median']:.2e}")
        amp, sham, att = (d[f"d2/{x}"]["plain/cosine_mean_ci"] for x in ("hunger/amplified", "hunger/sham", "energy/attenuated"))
        verdict = ("input-limited supported" if excl(amp) and not excl(sham) and att[1] <= 0 <= att[2]
                   else "not supported" if amp[1] <= 0 <= amp[2] else "mixed")
        print(f"  D2 reading: {verdict}")
        out[k] = {"bars": bars, "reference_valid": ref_ok, "d2": verdict}
    for k in CK:
        p = R / f"d1_ckpt_{k:02d}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        print(f"\n=== D1 checkpoint {k}  (AP: tokens / input / state; prevalence)")
        for key, v in d.get("validation", {}).items():
            print(f"  validation {key}: now {v['now']:.5f} vs section 6 {v['section6']:.5f}")
        for name, st in d.items():
            if isinstance(st, dict) and "p" in st:
                print(f"  {name:18s} " + " / ".join(f"{st[s]['ap']['ap']:.3f}" for s in ("p", "x", "b")) + f"  ({st['p']['prevalence']:.3f}, {st['p']['events']})")
    return out


if __name__ == "__main__":
    main()
