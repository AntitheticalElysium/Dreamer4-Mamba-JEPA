"""Residual shares of next-frame and horizon targets (PLAN.md). CPU.

    python -m artifacts.experiments.20261114_horizon_eda.horizon
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from saccade import data as D
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
GAMMAS, KMAX, N_EP = (0.0, 0.8, 0.9, 0.95), 60, 400


def cells(frames):
    f = torch.as_tensor(frames).reshape(-1, 9, 7, 9, 7, 3).permute(0, 1, 3, 5, 2, 4).reshape(-1, 3, 7, 7).float()
    return (torch.round(F.adaptive_avg_pool2d(f, 3)).reshape(len(frames), 81, 27) / 127.5 - 1).numpy()


def transport(x0, d):
    """Map rows (cells 0..62, 7 x 9) moved by displacement d (target (r, c) <- source (r + dr, c + dc)); HUD kept."""
    out = x0.copy()
    m = x0[:63].reshape(7, 9, -1)
    t = m.copy()
    dr, dc = int(d[0]), int(d[1])
    rs, cs = slice(max(0, -dr), 7 - max(0, dr)), slice(max(0, -dc), 9 - max(0, dc))
    rq, cq = slice(max(0, dr), 7 + min(0, dr)), slice(max(0, dc), 9 + min(0, dc))
    if rs.start < rs.stop and cs.start < cs.stop:
        t[rs, cs] = m[rq, cq]
    out[:63] = t.reshape(63, -1)
    return out


def main():
    torch.set_num_threads(6)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    eps = S.episodes(L)
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    pick = np.sort(np.random.default_rng(1114).choice(len(eps), N_EP, replace=False))
    groups = ("map_day", "map_night", "need_digits", "need_digits_timer_only", "health", "hud_other")
    acc = {(g, b, gm): 0.0 for g in groups for b in ("copy", "transport") for gm in GAMMAS}
    frames_n = 0
    for e in pick:
        ep = eps[e]
        x = cells(D.RP.episode_fields(ep.shard, ep.slot)["observations"].numpy())   # [T + 1, 81, 27]
        a, b = off[e], off[e + 1]
        pos, light, body = L["s_pos"][a:b].astype(int), L["s_light"][a:b], L["s_body"][a:b].astype(int)
        T = ep.T
        drops = (body[1:, 1:] < body[:-1, 1:]).any(1)                     # need digits down at transition t -> t + 1
        rises = (body[1:, 1:] > body[:-1, 1:]).any(1)
        for t in range(T):
            K = min(KMAX, T - t)
            ks = np.arange(1, K + 1)
            tr = np.stack([transport(x[t], pos[t + k] - pos[t]) for k in ks])        # [K, 81, 27]
            fut = x[t + 1:t + K + 1]
            night = light[t] < .5
            for gm in GAMMAS:
                w = np.ones(1) if gm == 0 else (1 - gm) * gm ** (ks - 1)
                kk = 1 if gm == 0 else K
                w = w[:kk] / w[:kk].sum()
                y = (w[:, None, None] * fut[:kk]).sum(0)
                eff = kk if gm == 0 else int(min(K, np.ceil(3 / (1 - gm))))
                timer_only = drops[t:t + eff].any() and not rises[t:t + eff].any()
                for base, yh in (("copy", x[t]), ("transport", (w[:, None, None] * tr[:kk]).sum(0))):
                    r = ((y - yh) ** 2).sum(-1)                                  # [81]
                    acc[("map_night" if night else "map_day", base, gm)] += float(r[:63].sum())
                    acc[("need_digits", base, gm)] += float(r[64:67].sum())
                    if timer_only:
                        acc[("need_digits_timer_only", base, gm)] += float(r[64:67].sum())
                    acc[("health", base, gm)] += float(r[63])
                    acc[("hud_other", base, gm)] += float(r[67:].sum())
        frames_n += T
    res = {"episodes": int(N_EP), "frames": int(frames_n), "kmax": KMAX, "shares": {}, "energy_per_frame": {}}
    for base in ("copy", "transport"):
        for gm in GAMMAS:
            tot = sum(acc[(g, base, gm)] for g in groups if g != "need_digits_timer_only")
            key = f"{base}/gamma={gm}"
            res["shares"][key] = {g: acc[(g, base, gm)] / tot for g in groups}
            res["energy_per_frame"][key] = tot / frames_n
            print(key, json.dumps({g: round(v, 4) for g, v in res["shares"][key].items()}), round(tot / frames_n, 3), flush=True)
    (HERE / "horizon.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
