"""Read-only post-hoc checks specified in PROTOCOL.md; no model calls."""
import importlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
A = importlib.import_module("artifacts.experiments.20261029_fovea_probes.analyse")
RUN = ROOT / "artifacts/experiments/20261029_fovea_probes/run"


def main():
    d = np.load(RUN / "f1.npz")
    L = np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz")
    T = d["ep_T"]
    ep = np.repeat(np.arange(len(T)), T + 1)
    n = len(ep)
    cur = np.arange(n)
    prev = np.maximum(cur - 1, 0)
    valid_time = (cur > 0) & (ep == ep[prev])
    pos = L["s_pos"][:n].astype(int)
    keys, scr = A.keys(pos, d["fix"].astype(int))
    cont = A.content(L, cur, keys)
    cont_prev = A.content(L, prev, keys)
    seen = []
    last = np.zeros((n, 9), int)
    for lag in range(8, 0, -1):
        src = np.maximum(cur - lag, 0)
        ok = (cur >= lag) & (ep == ep[src])
        yes = (keys[:, :, None] == keys[src, None, :]).all(-1).any(-1) & ok[:, None]
        last = np.where(yes, lag, last)
        if lag == 1:
            seen = yes

    # Inclusion probability of this target cell under the 49 equiprobable crop centres at t-1.
    hud = scr >= 63
    sr = np.where(hud, scr // 9, keys[..., 0] - pos[prev, 0, None] + 3)
    sc = np.where(hud, scr % 9, keys[..., 1] - pos[prev, 1, None] + 4)
    covers = lambda s: (np.abs(s[..., None] - np.arange(1, 8)) <= 1).sum(-1)
    p = covers(sr) * covers(sc) / 49
    p = np.where(hud | ((sr >= 0) & (sr < 7) & (sc >= 0) & (sc < 9)), p, 0)
    eligible = valid_time[:, None] & (p > 0) & (p < 1)
    assert (seen[eligible] == (last[eligible] == 1)).all()

    night = np.broadcast_to((L["s_light"][:n] < 0.5)[:, None], (n, 9))
    static = (pos == pos[prev]).all(-1) & valid_time
    terrain = (cont == 0) & (cont_prev == 0)
    vr, vc = (scr // 9).clip(0, 6), scr % 9
    view = L["s_view"][:n]
    same_code = view[cur[:, None], vr, vc] == view[prev[:, None], vr, vc]
    strict = static[:, None] & terrain & same_code & (scr != 31) & ~hud
    y = d["tsp"].astype(float)
    episodes = np.broadcast_to(ep[:, None], y.shape)
    strata = scr + 81 * night
    rng = np.random.default_rng(20291029)
    draws = rng.multinomial(len(T), np.ones(len(T)) / len(T), size=500)

    def contrast(sel):
        sel = sel & eligible
        aa, yy, pp, ee = seen[sel], y[sel], p[sel], episodes[sel]
        agg = np.stack([
            np.bincount(ee, minlength=len(T)),
            np.bincount(ee, weights=aa * yy / pp, minlength=len(T)),
            np.bincount(ee, weights=aa / pp, minlength=len(T)),
            np.bincount(ee, weights=(~aa) * yy / (1 - pp), minlength=len(T)),
            np.bincount(ee, weights=(~aa) / (1 - pp), minlength=len(T)),
        ], axis=1)

        def estimates(s):
            return np.stack([(s[..., 3] - s[..., 1]) / s[..., 0],
                             s[..., 3] / s[..., 4] - s[..., 1] / s[..., 2]], axis=-1)
        est = estimates(agg.sum(0))
        boot = estimates(draws @ agg)
        return {"units": int(sel.sum()), "exposed": int(aa.sum()),
                "HT_gain": float(est[0]), "HT_se": float(boot[:, 0].std()),
                "normalized_weight_gain": float(est[1]), "normalized_weight_se": float(boot[:, 1].std())}

    def original(sel):
        s1 = (sel & valid_time[:, None] & (last == 1)).ravel()
        s0 = (sel & valid_time[:, None] & (last == 0)).ravel()
        st = strata.ravel()
        z = y.ravel()
        n1 = np.bincount(st[s1], minlength=162)
        n0 = np.bincount(st[s0], minlength=162)
        ok = (n1 > 0) & (n0 > 0)
        m1 = np.bincount(st[s1], weights=z[s1], minlength=162) / np.maximum(n1, 1)
        m0 = np.bincount(st[s0], weights=z[s0], minlength=162) / np.maximum(n0, 1)
        return float(((m0 - m1)[ok] * n1[ok]).sum() / n1[ok].sum())

    groups = {"all_day": ~night, "all_night": night,
              "static_terrain_day": static[:, None] & terrain & ~night,
              "strict_static_terrain_day": strict & ~night,
              "strict_static_terrain_night": strict & night}
    result = {"post_hoc": True, "world_model_calls": 0, "episodes": len(T),
              "F1": {name: {"original_lag1_vs_not_seen8_gain": original(sel), **contrast(sel)}
                     for name, sel in groups.items()}}

    f2 = np.load(RUN / "f2.npz")
    j = json.loads((RUN / "results.json").read_text())
    gg = np.r_[0, np.cumsum(L["ep_T"] + 1)][f2["e"]] + f2["t"]
    sets = {"random_day": ~f2["night_set"] & (L["s_light"][gg] >= .5),
            "random_night": ~f2["night_set"] & (L["s_light"][gg] < .5),
            "night_with_mob": f2["night_set"]}
    tot = f2["loss"].mean(2).sum(-1)
    centres = np.array([(r, c) for r in range(1, 8) for c in range(1, 8)])
    hud_cover = centres[:, 0] >= 6
    result["F2"] = {}
    for name, sel in sets.items():
        val = tot[sel].mean(-1, keepdims=True) - tot[sel]
        excess = val[:, 0] - val[:, 0].mean(0, keepdims=True)
        gain = j["F2"][name]["R2_state_dependence"]["est"]
        result["F2"][name] = {"frames": int(sel.sum()), "loss": float(tot[sel].mean()),
                              "reported_relative_gain_percent": 100 * gain / float(tot[sel].mean()),
                              "position_centered_HUD_excess_forced_zero": float(excess[:, hud_cover].mean())}
    (OUT / "checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
