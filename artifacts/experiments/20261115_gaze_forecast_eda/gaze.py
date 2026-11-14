"""Gaze-forecast EDA (PLAN.md): the chooser's distribution at real, no-event and brightness-foil frames on all DEV.

    MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu python -m artifacts.experiments.20261115_gaze_forecast_eda.gaze
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
NP = importlib.import_module("artifacts.experiments.20261111_p1f5.native_pairs")
LB = importlib.import_module("artifacts.experiments.20261114_learnability.learnable")
CLOCKS = {"food_down": (0, 64), "drink_down": (1, 65), "energy_down_awake": (2, 66), "energy_up_asleep": (3, 66),
          "recovery": (4, 63)}                                          # events() column, digit cell
FOILS = (1, 3, 6)
P = 8


def kl(p, q):
    return (p * (p.clamp(min=1e-12).log() - q.clamp(min=1e-12).log())).sum(-1)


def main():
    dev = torch.device("cuda")
    model, cfg = NP.IF.load(EXP / "20261113_p1f6/run/ckpt_10.pt")
    state = torch.load(EXP / "20261113_p1f6/run/chooser_last.pt", map_location="cpu", weights_only=False)
    pol = dict(state["config"])
    head = C.Head(pol["history"]).to(dev)
    head.load_state_dict(state["head"])
    head.eval()
    win = C.WIN.to(dev)
    cover = torch.zeros(81, 49, device=dev)
    cover[win.T.flatten(), torch.arange(49, device=dev).repeat(9)] = 1     # cover[cell, window] = window shows cell
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    y_ev, ok_ev = LB.events(L)
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    pos, light = L["s_pos"].astype(int), L["s_light"].astype(float)
    eps = S.episodes(L)
    rows = {k: [] for k in CLOCKS}
    copy_kl, moved_all, night_all = [], [], []

    def scores(frames, xx, idx):
        with TR.autocast(dev, cfg):
            p, _ = C.coarse(model.perception, frames)
        with torch.no_grad(), torch.autocast(dev.type, enabled=False):
            x = tuple(z[idx] for z in xx)
            taken, noop = head(x), head((x[0], x[1], torch.zeros_like(x[2])))
            c = (taken - noop).square().sum(-1)[:, win].sum(-1)
            s = (p.reshape(-1, 81, 16).float() - taken).square().sum(-1)[:, win].sum(-1)
            return C.policy(c, s, pol)

    for e, ep in enumerate(eps):
        T = int(L["ep_T"][e])
        chunk = D.gather([ep], [[None] * P + [(0, 1, t, 1, 1) for t in range(T + 1)]])
        fr = torch.as_tensor(chunk["frames"], device=dev)
        with torch.no_grad(), TR.autocast(dev, cfg):
            pv = [C.coarse(model.perception, z) for z in fr[0].split(512)]
        p = torch.cat([a for a, _ in pv])[None].float()
        v = torch.cat([b for _, b in pv])[None].float()
        xx, _ = C.inputs(p, v, chunk, P, pol["history"])
        prob = scores(fr[0, P:], xx, torch.arange(T + 1, device=dev))      # [T + 1, 49], frame t = target t
        g = off[e] + np.arange(1, T + 1)
        copy_kl.append(kl(prob[1:], prob[:-1]).cpu().numpy())
        moved_all.append((pos[g] != pos[g - 1]).any(1))
        night_all.append(light[g] < 0.5)
        for name, (col, cell) in CLOCKS.items():
            ts = np.flatnonzero(y_ev[off[e]:off[e] + T + 1, col])
            ts = ts[ts >= 1]
            if not len(ts):
                continue
            r, c = divmod(cell, 9)
            base = fr[0, P + ts].clone()
            base[:, 7 * r:7 * r + 7, 7 * c:7 * c + 7] = fr[0, P + ts - 1, 7 * r:7 * r + 7, 7 * c:7 * c + 7]
            idx = torch.as_tensor(ts, device=dev)
            q0 = scores(base, xx, idx)
            real = prob[idx]
            ev = kl(real, q0)
            out = {"t": ts, "g": off[e] + ts, "event_kl": ev.cpu().numpy(), "cover_real": (real @ cover[cell]).cpu().numpy(),
                   "cover_noevent": (q0 @ cover[cell]).cpu().numpy()}
            for d in FOILS:
                foil = (base.to(torch.int16) + d).clamp(0, 255).to(torch.uint8)
                out[f"foil{d}_kl"] = kl(scores(foil, xx, idx), q0).cpu().numpy()
            rows[name].append(out)
        if e % 100 == 0:
            print("episode", e, len(eps), flush=True)
    copy_kl, moved, night = (np.concatenate(x) for x in (copy_kl, moved_all, night_all))
    frames = len(copy_kl)
    res = {"frames_with_previous": frames, "copy_kl_mean": float(copy_kl.mean()),
           "copy_kl_mean_by": {"moved": float(copy_kl[moved].mean()), "still": float(copy_kl[~moved].mean()),
                               "day": float(copy_kl[~night].mean()), "night": float(copy_kl[night].mean())},
           "clocks": {}}
    gpos = np.concatenate([off[e] + np.arange(1, int(L["ep_T"][e]) + 1) for e in range(len(eps))])
    is_moved = np.zeros(off[-1], bool); is_moved[gpos] = moved
    is_night = np.zeros(off[-1], bool); is_night[gpos] = night
    for name, parts in rows.items():
        z = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        n, col = len(z["t"]), CLOCKS[name][0]
        price = n / frames * float(z["event_kl"].mean())
        strata = {}
        for sname, m in (("moved", is_moved[z["g"]]), ("still", ~is_moved[z["g"]]), ("day", ~is_night[z["g"]]), ("night", is_night[z["g"]])):
            strata[sname] = {"events": int(m.sum()), "event_kl_mean": float(z["event_kl"][m].mean()) if m.any() else None}
        res["clocks"][name] = {
            "events": n, "eligible_frames": int(ok_ev[:, col].sum()), "event_kl_mean": float(z["event_kl"].mean()),
            "event_kl_median": float(np.median(z["event_kl"])), "price_per_frame": price, "share_of_copy_kl": price / res["copy_kl_mean"],
            "cover_real": float(z["cover_real"].mean()), "cover_noevent": float(z["cover_noevent"].mean()),
            "foil_wins": {str(d): float((z[f"foil{d}_kl"] >= z["event_kl"]).mean()) for d in FOILS},
            "foil_kl_mean": {str(d): float(z[f"foil{d}_kl"].mean()) for d in FOILS}, "strata": strata}
        print(name, json.dumps({k: v for k, v in res["clocks"][name].items() if k != "strata"}), flush=True)
    pix = ("food_down", "drink_down", "energy_down_awake", "energy_up_asleep")
    res["pixel_clock_share"] = sum(res["clocks"][k]["share_of_copy_kl"] for k in pix)
    res["checkpoint_sha256"] = TR.file_sha256(EXP / "20261113_p1f6/run/ckpt_10.pt")
    (HERE / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print("pixel_clock_share", res["pixel_clock_share"], flush=True)


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    main()
