"""Decision check (2026-10-03): what does a real-fitted head miss in the imagined depth-k state -- the drawn outcome (HUD) or the
map? (check_transfer_subst at H1: the real HUD alone closes 99.8-99.9% of the transfer gap. check_h16_value: one faithful future per
action carries 57% of the H16 decision; check_damage_rule: from a 4-frame input a zombie hit is a coin flip in the commonest case,
so a deterministic L1 world draws no hit. Sealed deepeval: transfer16 0.565-0.574, real16 0.652.)
deepeval's DEV split (3,096 roots; NOT the sealed judge block; DEV also selected deepeval's heads, so every arm here carries that
mild selection), deepeval's imagination (teval convention, recorded continuation) and its own real_k heads (3 seeds, cached),
read on depth-k states built from the imagined state and the REAL depth-k frame (key sequence 0):
  imagined, +real_hud (tokens 63-80), +real_near (3 x 3 around the player), +real_map (tokens 0-62), real
Expected safe on DEV roots with opportunity at k (seed mean), k in 1, 4, 16. Also the imagined vs real health (teval HUD probe) at
depth 16: mean, and the per-root correlation of -health with P(dead by 16) across the 17 first actions.
Reading, declared before running: h16_hud_bottleneck = (+real_hud - imagined) >= 0.5 x (real - imagined) at k = 16 in every world.
Usage: check_h16_subst.py <world.pt> ...
Result (2026-10-03; DEV opportunity roots k1 640 / k4 1,309 / k16 2,590; real 0.999 / 0.738 / 0.691):
  k16 imagined -> +real_hud (HUD share of the gap) / +real_map: s7 18k 0.597 -> 0.688 (0.960) / 0.591; s7 36k 0.600 -> 0.687
  (0.951) / 0.589; s8 18k 0.585 -> 0.685 (0.938) / 0.579; s8 36k 0.600 -> 0.688 (0.966) / 0.594. k4 HUD share 0.906-0.948; k1
  0.998-0.999. The real map never helps at k16 (share -0.06 to -0.12). h16_hud_bottleneck TRUE.
  Imagined health at 16 (opportunity roots): 5.93 / 6.03 / 5.94 / 5.94 vs real 2.42; danger_corr 0.15 / 0.35 / 0.08 / 0.35 vs real 0.54.
  So at every depth what a real-fitted head misses in imagination is the drawn outcome (damage / death), not the map; the imagined
  futures keep ~3.5 more health than the real ones.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import deepeval as E  # noqa: E402
D, T = E.D, E.T
NEAR = [21, 22, 23, 30, 31, 32, 39, 40, 41]
CELLS = {"+real_hud": list(range(63, 81)), "+real_near": NEAR, "+real_map": list(range(63))}


def main():
    from d4mj.config import config_from_dict
    from observability import expected_safe
    import spatial as S
    dev_ = torch.device("cuda")
    log = lambda **kw: print(json.dumps(kw), flush=True)
    data = E.token_cache(dev_, log)
    fit, dev = data["fit"], data["dev"]
    rows = torch.arange(len(dev["seed"]))
    heads = {k: [D.fit_head(fit[f"real{k}"], fit[f"p{k}"], dev[f"real{k}"], dev[f"p{k}"], dev_, s,
                            E.CACHE / "heads" / f"real{k}_s{s}.pt")[0] for s in range(E.SEEDS)] for k in E.DEPTHS}
    opp = {k: dev[f"p{k}"].amax(1) > dev[f"p{k}"].amin(1) for k in E.DEPTHS}
    safe = lambda k, x: float(torch.stack([expected_safe(D.judge_risk(m, x, rows, dev_), dev[f"p{k}"])[0]
                                           for m in heads[k]]).mean(0)[opp[k]].mean())
    tm, tr, ts = T.split()
    probes = T.Probes(T.build_cache("raw", torch.device("cpu")), tm, tr, ts)
    health = lambda x: torch.cat([probes.hud(x[i:i + 64, :, 63:81].float().flatten(2).flatten(0, 1))[:, 0].view(-1, E.N)
                                  for i in range(0, len(x), 64)]) * 9

    def danger(h, p):
        a = p - p.mean(1, keepdim=True); b = -h + h.mean(1, keepdim=True)
        den = a.norm(dim=1) * b.norm(dim=1)
        ok = den > 1e-9
        return float(((a * b).sum(1)[ok] / den[ok]).mean())

    real = {k: safe(k, dev[f"real{k}"]) for k in E.DEPTHS}
    h_real = health(dev["real16"])
    out = {"dev_roots": len(rows), "opportunity": {k: int(m.sum()) for k, m in opp.items()}, "real": real,
           "real_health16": {"mean": float(h_real[opp[16]].mean()), "danger_corr": danger(h_real[opp[16]], dev["p16"][opp[16]])}}
    log(**out)
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    for path in sys.argv[1:]:
        world, st = T.load_world(Path(path), dev_)
        name = st["name"]
        gen = E.imagine(world, config, dev, 5, f"{E.CACHE}/h16subst_{name}_dev", dev_)
        r = {}
        for k in E.DEPTHS:
            rk = {"imagined": safe(k, gen[k])}
            for arm, cells in CELLS.items():
                x = gen[k].clone(); x[:, :, cells] = dev[f"real{k}"][:, :, cells]
                rk[arm] = safe(k, x)
            gap = real[k] - rk["imagined"]
            rk["real"] = real[k]
            rk["hud_share"] = (rk["+real_hud"] - rk["imagined"]) / gap if gap > 0 else None
            rk["map_share"] = (rk["+real_map"] - rk["imagined"]) / gap if gap > 0 else None
            r[f"k{k}"] = rk
        h = health(gen[16])
        r["imagined_health16"] = {"mean": float(h[opp[16]].mean()), "danger_corr": danger(h[opp[16]], dev["p16"][opp[16]])}
        out[name] = r
        log(world=name, **r)
        del world, gen
        for f in E.CACHE.glob(f"h16subst_{name}_dev_*"):
            f.unlink()
        torch.cuda.empty_cache()
    ws = [k for k in out if isinstance(out[k], dict) and "k16" in out[k]]
    out["readings"] = {"h16_hud_bottleneck": all((out[w]["k16"]["hud_share"] or 0) >= 0.5 for w in ws)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
