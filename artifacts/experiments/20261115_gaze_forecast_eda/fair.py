"""Fair denominator for the gaze-forecast share (PLAN.md, "Fair denominator"): a forecaster of the chooser's 49-window
distribution from exactly the chooser's own inputs (last 8 frames' coarse tokens and outcomes, the action into the
frame), fitted on 4/5 of DEV episodes; on the held-out fifth, its KL to the real distribution against the pixel clocks'
price (event KL against the no-event counterfactual, as gaze.py) on the same frames. CPU, full precision.

    MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m artifacts.experiments.20261115_gaze_forecast_eda.fair
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from saccade import chooser as C, data as D, model as M, train as TR
from saccade.eval import s1 as S
from . import gaze as G

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DEV = torch.device("cpu")
H = 8


class Forecaster(nn.Module):
    """The chooser head's trunk (C.Head), read out as one score per cell; a window's logit is the sum of its nine cells'
    scores plus a per-window bias."""

    def __init__(self):
        super().__init__()
        self.trunk = C.Head(H)
        self.score = nn.Linear(64, 1)
        self.bias = nn.Parameter(torch.zeros(49))

    def forward(self, ph, ctx, act):
        t = self.trunk
        x = t.local(ph.transpose(1, 2).flatten(2)) + t.context(ctx)[:, None] + t.action(act)[:, None] + t.position
        return self.score(t.blocks(x))[..., 0][:, C.WIN].sum(-1) + self.bias


def main(epochs=4, batch=512):
    torch.manual_seed(0)
    ck = torch.load(EXP / "20261113_p1f6/run/ckpt_10.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = M.M0(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    state = torch.load(EXP / "20261113_p1f6/run/chooser_last.pt", map_location="cpu", weights_only=False)
    pol = dict(state["config"])
    head = C.Head(pol["history"])
    head.load_state_dict(state["head"])
    head.eval()
    win, cover = C.WIN, torch.zeros(81, 49)
    cover[win.T.flatten(), torch.arange(49).repeat(9)] = 1
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    y_ev, _ = G.LB.events(L)
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    pos, light = L["s_pos"].astype(int), L["s_light"].astype(float)
    eps = S.episodes(L)
    fold = np.arange(len(eps)) % 5

    @torch.no_grad()
    def scores(frames, xx, idx):
        p, _ = C.coarse(model.perception, frames)
        x = tuple(z[idx] for z in xx)
        taken, noop = head(x), head((x[0], x[1], torch.zeros_like(x[2])))
        c = (taken - noop).square().sum(-1)[:, win].sum(-1)
        s = (p.reshape(-1, 81, 16).float() - taken).square().sum(-1)[:, win].sum(-1)
        return C.policy(c, s, pol)

    toks, ctxs, acts, probs, glob, epi, prev = [], [], [], [], [], [], []
    events = {k: [] for k in G.CLOCKS}
    base = 0
    for e, ep in enumerate(eps):
        T = int(L["ep_T"][e])
        chunk = D.gather([ep], [[None] * G.P + [(0, 1, t, 1, 1) for t in range(T + 1)]])
        fr = torch.as_tensor(chunk["frames"])
        with torch.no_grad():
            pv = [C.coarse(model.perception, z) for z in fr[0].split(512)]
        p = torch.cat([a for a, _ in pv])[None].float()
        v = torch.cat([b for _, b in pv])[None].float()
        xx, _ = C.inputs(p, v, chunk, G.P, pol["history"])
        idx = torch.arange(1, T + 1)
        prob = scores(fr[0, G.P + 1:], xx, idx)                          # frames 1..T (frame 0 unscored, as in training)
        toks.append(p[0, G.P:].half())                                   # frames 0..T, for building histories
        ctxs.append(xx[1][idx].half()); acts.append(xx[2][idx]); probs.append(prob)
        glob.append(base + idx.numpy()); epi.append(np.full(T, e)); prev.append(np.r_[False, np.ones(T - 1, bool)])
        base += T + 1
        if fold[e] == 4:                                                 # the held-out fifth: event prices on its frames
            for name, (col, cell) in G.CLOCKS.items():
                ts = np.flatnonzero(y_ev[off[e]:off[e] + T + 1, col]); ts = ts[ts >= 1]
                if not len(ts):
                    continue
                r, c = divmod(cell, 9)
                cf = fr[0, G.P + ts].clone()
                cf[:, 7 * r:7 * r + 7, 7 * c:7 * c + 7] = fr[0, G.P + ts - 1, 7 * r:7 * r + 7, 7 * c:7 * c + 7]
                q0 = scores(cf, xx, torch.as_tensor(ts))
                events[name].append(np.stack([np.full(len(ts), e), ts, G.kl(prob[ts - 1], q0).numpy()], 1))
        if e % 100 == 0:
            print("episode", e, len(eps), flush=True)
    toks = torch.cat(toks); ctx = torch.cat(ctxs); act = torch.cat(acts); prob = torch.cat(probs)
    glob = np.concatenate(glob); epi = np.concatenate(epi); prev = np.concatenate(prev)
    t_in_ep = np.concatenate([np.arange(1, int(L["ep_T"][e]) + 1) for e in range(len(eps))])
    g_lab = off[epi] + t_in_ep
    moved = (pos[g_lab] != pos[g_lab - 1]).any(1)
    night = light[g_lab] < 0.5
    ok = ctx.float().reshape(-1, H, 29)[..., 28]                       # the chooser's own history-validity flags

    def history(rows):
        hist = torch.as_tensor(glob[rows])[:, None] - torch.arange(H, 0, -1)[None]
        return toks[hist.clamp(min=0)].float() * ok[rows][..., None, None]

    train = np.flatnonzero(fold[epi] != 4); test = np.flatnonzero(fold[epi] == 4)
    net = Forecaster()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    log = []
    for epoch in range(epochs):
        for g in opt.param_groups:
            g["lr"] = 1e-3 if epoch < epochs - 1 else 3e-4
        perm = rng.permutation(train)
        losses = []
        for a in range(0, len(perm), batch):
            rows = perm[a:a + batch]
            q = net(history(rows), ctx[rows].float(), act[rows]).log_softmax(-1)
            loss = -(prob[rows] * q).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        log.append({"epoch": epoch, "train_cross_entropy": float(np.mean(losses))})
        print(log[-1], flush=True)
    net.eval()
    with torch.no_grad():
        q = torch.cat([net(history(test[a:a + 4096]), ctx[test[a:a + 4096]].float(), act[test[a:a + 4096]]).softmax(-1)
                       for a in range(0, len(test), 4096)])
    kl_f = G.kl(prob[test], q).numpy()
    has_prev = prev[test]
    kl_copy = G.kl(prob[test][has_prev], prob[test - 1][has_prev]).numpy()
    m, n_ = moved[test], night[test]
    res = {"train_log": log, "test_frames": len(test), "forecaster_kl_mean": float(kl_f.mean()),
           "forecaster_kl_by": {"moved": float(kl_f[m].mean()), "still": float(kl_f[~m].mean()),
                                "day": float(kl_f[~n_].mean()), "night": float(kl_f[n_].mean())},
           "copy_kl_mean_same_frames": float(kl_copy.mean()), "clocks": {}}
    row_of = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(epi[test], t_in_ep[test]))}
    for name, parts in events.items():
        z = np.concatenate(parts)
        price = len(z) / len(test) * float(z[:, 2].mean())
        at = np.array([row_of[(int(e), int(t))] for e, t, _ in z])
        res["clocks"][name] = {"events": len(z), "event_kl_mean": float(z[:, 2].mean()), "price_per_frame": price,
                               "share_of_forecaster_kl": price / res["forecaster_kl_mean"],
                               "forecaster_kl_at_events": float(kl_f[at].mean())}
        print(name, json.dumps(res["clocks"][name]), flush=True)
    pix = ("food_down", "drink_down", "energy_down_awake", "energy_up_asleep")
    res["pixel_clock_share"] = sum(res["clocks"][k]["share_of_forecaster_kl"] for k in pix)
    res["pixel_clock_share_of_copy_kl_same_frames"] = sum(res["clocks"][k]["price_per_frame"] for k in pix) / res["copy_kl_mean_same_frames"]
    (HERE / "fair.json").write_text(json.dumps(res, indent=1) + "\n")
    print("pixel_clock_share", res["pixel_clock_share"], "copy-denominator on the same frames",
          res["pixel_clock_share_of_copy_kl_same_frames"], flush=True)


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    main()
