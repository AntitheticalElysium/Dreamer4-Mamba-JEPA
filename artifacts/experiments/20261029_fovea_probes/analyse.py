"""Readouts of PLAN.md (F1 retention; F2 R1-R5) from measure.py's f1.npz / f2.npz; CPU, no model.
Written before the measurement finished. Episode-bootstrap SE (200 resamples) throughout.
Usage: python -m artifacts.experiments.20261029_fovea_probes.analyse RUN_DIR
"""
import json, sys
from pathlib import Path

import numpy as np

from saccade.eval import s1 as S

ROOT = Path(__file__).resolve().parents[3]
KINDS = ("zombies", "cows", "skeletons", "arrows")
CENTRES = np.array([(r, c) for r in range(1, 8) for c in range(1, 8)])
D3 = np.array([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)])      # window offsets, row-major (model.window)
DIRS = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
BOOT = 200


def boot(fn, ep, rng=np.random.default_rng(0)):
    """fn(mask over rows) -> scalar; bootstrap over episodes."""
    ids = np.unique(ep)
    est = fn(np.ones(len(ep), bool))
    vals = []
    for _ in range(BOOT):
        pick = rng.choice(ids, len(ids))
        w = np.bincount(np.searchsorted(ids, pick), minlength=len(ids))[np.searchsorted(ids, ep)]
        vals.append(fn(w))
    return {"est": float(est), "se": float(np.nanstd(vals))}


def keys(pos, fix):
    """World / HUD keys of the 9 glimpse cells: map cells -> world (row, col); HUD cells -> (-100 - screen index, 0)."""
    sr, sc = fix[:, None, 0] + D3[:, 0], fix[:, None, 1] + D3[:, 1]
    world = np.stack([pos[:, None, 0] + sr - 3, pos[:, None, 1] + sc - 4], -1)
    hud = np.stack([-100 - (sr * 9 + sc), np.zeros_like(sr)], -1)
    return np.where((sr < 7)[..., None], world, hud), sr * 9 + sc


def content(L, g, k):
    """Per (state g, key k [.., 2]): 0 terrain, 1-4 the entity kind present, 5 HUD."""
    out = np.where(k[..., 0] <= -100, 5, 0)
    for i, kind in enumerate(KINDS):
        p, m = L[f"s_{kind}_p"][g], L[f"s_{kind}_m"][g]                      # [n, M, 2], [n, M]
        hit = (m[:, None, :] & (p[:, None, :, :] == k.reshape(len(g), -1, 1, 2)).all(-1)).any(-1).reshape(k.shape[:-1])
        out = np.where(hit & (out == 0), i + 1, out)
    return out


def f1(run, L):
    d = np.load(run / "f1.npz")
    T = d["ep_T"]; s_off = np.r_[0, np.cumsum(T + 1)]
    n = s_off[-1]
    pos, light = L["s_pos"][:n].astype(int), L["s_light"][:n]
    k, scr = keys(pos, d["fix"].astype(int))                                  # [n, 9, 2], [n, 9]
    ep = np.repeat(np.arange(len(T)), T + 1)
    last = np.zeros((n, 9), int)                                              # most recent lag 1..8 seen, 0 = not
    for lag in range(8, 0, -1):
        src = np.arange(n) - lag
        ok = (src >= 0) & (ep[np.maximum(src, 0)] == ep)
        prev = k[np.maximum(src, 0)]                                          # [n, 9, 2] keys seen at t - lag
        seen = ((k[:, :, None, :] == prev[:, None, :, :]).all(-1).any(-1)) & ok[:, None]
        last = np.where(seen, lag, last)
    cont = content(L, np.arange(n), k)
    night = np.repeat((light < 0.5)[:, None], 9, 1)
    rows = dict(tsp=d["tsp"].ravel(), rec=d["rec"].ravel(), last=last.ravel(), scr=scr.ravel(), night=night.ravel(),
                cont=cont.ravel(), ep=np.repeat(ep, 9))
    first = np.repeat(np.r_[[t > 0 for e in range(len(T)) for t in range(T[e] + 1)]], 9)
    rows = {a: b[first] for a, b in rows.items()}

    strat = rows["scr"] + 81 * rows["night"]

    def gain(lag, sel):
        """Seen-weighted mean over (screen cell, night) strata of NLL(not seen in 8 frames) - NLL(seen at lag)."""
        seen, unseen = sel & (rows["last"] == lag), sel & (rows["last"] == 0)

        def fn(w):
            sums = lambda m, x: np.bincount(strat[m], weights=(w * x)[m], minlength=162)
            ws, wu = sums(seen, 1.0 + 0 * rows["tsp"]), sums(unseen, 1.0 + 0 * rows["tsp"])
            ok = (ws > 0) & (wu > 0)
            diff = sums(unseen, rows["tsp"])[ok] / wu[ok] - sums(seen, rows["tsp"])[ok] / ws[ok]
            return (ws[ok] * diff).sum() / ws[ok].sum() if ok.any() else np.nan
        return boot(fn, rows["ep"])

    res = {"cells": int(len(rows["ep"])), "seen_share_by_lag": {str(l): float((rows["last"] == l).mean()) for l in range(9)}}
    for name, sel in (("all_day", ~rows["night"]), ("all_night", rows["night"]),
                      *((f"{c}_{dn}", (rows["cont"] == i) & (rows["night"] == (dn == "night")))
                        for i, c in enumerate(("terrain", *KINDS, "hud")) for dn in ("day", "night"))):
        if (sel & (rows["last"] == 1)).sum() < 50:
            continue
        res[name] = {"n_seen_lag1": int((sel & (rows["last"] == 1)).sum()),
                     "retention_gain_nats_per_pixel": {str(l): gain(l, sel) for l in (1, 2, 4, 8)},
                     "input_ceiling_tsp_minus_rec": boot(lambda w, s=sel: (w * s * (rows["tsp"] - rows["rec"])).sum()
                                                         / max((w * s).sum(), 1e-9), rows["ep"]),
                     "tsp_mean": float(rows["tsp"][sel].mean())}
        print("F1", name, res[name]["retention_gain_nats_per_pixel"]["1"], flush=True)
    return res


def f2(run, L):
    d = np.load(run / "f2.npz")
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    g = s_off[d["e"]] + d["t"]
    loss = d["loss"].mean(2)                                                  # [n, 2 sets, 49, 3] mean over sigma
    tot = loss.sum(-1)                                                        # [n, 2, 49]
    val = tot.mean(-1, keepdims=True) - tot                                   # value: better than the average fixation
    ep, night, nset = d["e"], L["s_light"][g] < 0.5, d["night_set"].astype(bool)
    pos = L["s_pos"][g].astype(int)
    wk = np.stack([np.stack([pos[:, 0] + CENTRES[None, :, 0] + dr - 3, pos[:, 1] + CENTRES[None, :, 1] + dc - 4], -1)
                   for dr, dc in D3], 2)                                      # [n, 49, 9, 2] world keys of each window
    scr = (CENTRES[:, None, 0] + D3[None, :, 0]) * 9 + CENTRES[:, None, 1] + D3[None, :, 1]   # [49, 9]
    hud_cover = (scr >= 63).any(-1)
    wk = np.where((scr < 63)[None, ..., None], wk, -1000)
    kinds = np.stack([content(L, g, wk[:, :, j]) for j in range(9)], -1)     # [n, 49, 9]
    face = pos + DIRS[L["s_dir"][g]]
    has_face = (wk == face[:, None, None, :]).all(-1).any(-1)
    nxt = np.minimum(g + 1, len(L["s_light"]) - 1)
    cd0, cd1, zp, zm = L["s_zombies_cd"][g], L["s_zombies_cd"][nxt], L["s_zombies_p"][g].astype(int), L["s_zombies_m"][g]
    prv = np.maximum(g - 1, 0)
    attack_next = zm & (cd1 == 5) & (cd0 <= 0)
    attacked_now = zm & (cd0 == 5) & (L["s_zombies_cd"][prv] <= 0)
    in_win = lambda flag: ((wk[:, :, :, None, :] == zp[:, None, None, :, :]).all(-1) & flag[:, None, None, :]).any((-1, -2))
    res = {"samples": int(len(ep)), "random_set": int((~nset).sum()), "night_set": int(nset.sum())}

    def setwise(sel_rows, name):
        V = val[sel_rows]; e_ = ep[sel_rows]
        out = {"n": int(sel_rows.sum()),
               "value_map_A": V[:, 0].mean(0).reshape(7, 7).round(4).tolist(),
               "noise_set_agreement": float(np.nanmean([np.corrcoef(v[0], v[1])[0, 1] for v in V])),
               "mean_value_spread": float(V[:, 0].std(-1).mean()), "loss_mean": float(tot[sel_rows].mean())}
        odd = (e_ % 2).astype(bool)

        def r2(w):
            gains = []
            for fit, ev in ((~odd, odd), (odd, ~odd)):
                for a, b in ((0, 1), (1, 0)):
                    wf, we = w * fit, w * ev
                    if wf.sum() == 0 or we.sum() == 0:
                        continue
                    fixed = int(np.argmax((wf[:, None] * V[:, a]).sum(0)))
                    oracle = V[np.arange(len(V)), b, V[:, a].argmax(-1)]
                    gains.append(((we * oracle).sum() - (we * V[:, b, fixed]).sum()) / we.sum())
            return float(np.mean(gains))
        out["R2_state_dependence"] = boot(r2, e_)
        out["R2_oracle_value_B"] = boot(lambda w: (w * V[np.arange(len(V)), 1, V[:, 0].argmax(-1)]).sum() / w.sum(), e_)
        excess = V[:, 0] - V[:, 0].mean(0, keepdims=True)                     # position-matched
        K = kinds[sel_rows]
        groups = {**{k: (K == i + 1).any(-1) for i, k in enumerate(KINDS)}, "no_entity": ~((K >= 1) & (K <= 4)).any(-1),
                  "hud_cover": np.broadcast_to(hud_cover, K.shape[:2]), "faced_cell": has_face[sel_rows],
                  "attacker_next": in_win(attack_next)[sel_rows], "attacked_now": in_win(attacked_now)[sel_rows]}
        out["R3_R4_excess_value"] = {k: {"windows": int(m.sum()),
                                         "excess": boot(lambda w, m=m: (w[:, None] * m * excess).sum() / max((w[:, None] * m).sum(), 1e-9), e_)}
                                     for k, m in groups.items() if m.sum() >= 20}
        info = d["tsp_L"][sel_rows] - d["rec_L"][sel_rows]
        rank = lambda x: x.argsort(-1).argsort(-1)
        rho = np.array([np.corrcoef(rank(a), rank(b))[0, 1] for a, b in zip(V[:, 0], info)])
        empty = ~((K >= 1) & (K <= 4)).any((-1, -2))
        out["R5_value_vs_info_spearman"] = {"frames_no_entity_in_any_window": float(np.nanmean(rho[empty])) if empty.any() else None,
                                            "frames_with_entity": float(np.nanmean(rho[~empty])) if (~empty).any() else None,
                                            "n_empty": int(empty.sum())}
        res[name] = out
        print("F2", name, out["n"], "R2", out["R2_state_dependence"], "agreement", round(out["noise_set_agreement"], 3), flush=True)

    setwise(~nset & ~night, "random_day")
    setwise(~nset & night, "random_night")
    setwise(nset, "night_with_mob")
    return res


def main():
    run = Path(sys.argv[1])
    L = dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    T = np.load(run / "f1.npz")["ep_T"]
    L = S.first_episodes(L, len(T))
    res = {"F1": f1(run, L), "F2": f2(run, L)}
    (run / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
