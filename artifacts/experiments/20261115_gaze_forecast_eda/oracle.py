"""Oracle phase (PLAN.md, "Oracle phase"): does knowing the clocks' phase help forecast the gaze? fair.py's forecaster,
with one extra input: the simulator's four hidden counters at frame t - 1 (exact replay, `../20261107_f4_followups/
counters.py`), true against shuffled within (split, asleep at t - 1); per counter, the true arm with only that counter
shuffled at test time. Same chooser, inputs, capacity, split, budget and batch order as fair.py. CPU, full precision.
The counters are an evaluation device only; nothing here is a design.

    MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m artifacts.experiments.20261115_gaze_forecast_eda.oracle
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from saccade import chooser as C, data as D, model as M
from saccade.eval import s1 as S
from . import fair as FR, gaze as G

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
H = FR.H
COUNTERS = ("hunger", "thirst", "fatigue", "recover")
SCALE, SHIFT, BINS = (2, 2, 1, 2), (0, 0, 10, 30), (51, 41, 41, 81)   # exact values -> bin (all multiples of 0.5)
CLOCK_COUNTER = {"food_down": "hunger", "drink_down": "thirst", "energy_down_awake": "fatigue",
                 "energy_up_asleep": "fatigue", "recovery": "recover"}
PIXEL = ("hunger", "thirst", "fatigue")


class Oracle(FR.Forecaster):
    def __init__(self):
        super().__init__()
        self.phase = nn.Linear(sum(BINS), 64)

    def forward(self, ph, ctx, act, z):
        t = self.trunk
        x = t.local(ph.transpose(1, 2).flatten(2)) + t.context(ctx)[:, None] + t.action(act)[:, None] + t.position
        x = x + self.phase(z)[:, None]
        return self.score(t.blocks(x))[..., 0][:, C.WIN].sum(-1) + self.bias


def onehot(idx):
    return torch.cat([F.one_hot(torch.as_tensor(idx[:, k]), n).float() for k, n in enumerate(BINS)], -1)


def permute_within(rows, strata, rng):
    """A permutation of `rows` that maps each row to a row of the same stratum."""
    out = rows.copy()
    for s in np.unique(strata[rows]):
        m = np.flatnonzero(strata[rows] == s)
        out[m] = rows[m][rng.permutation(len(m))]
    return out


def main(epochs=4, batch=512):
    ck = torch.load(EXP / "20261113_p1f6/run/ckpt_10.pt", map_location="cpu", weights_only=False)
    model = M.M0(ck["cfg"])
    model.load_state_dict(ck["model"])
    model.eval()
    state = torch.load(EXP / "20261113_p1f6/run/chooser_last.pt", map_location="cpu", weights_only=False)
    pol = dict(state["config"])
    head = C.Head(pol["history"])
    head.load_state_dict(state["head"])
    head.eval()
    win = C.WIN
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    K = np.load(EXP / "20261107_f4_followups/readings/counters_dev.npz")
    y_ev, ok_ev = G.LB.events(L)
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    pos, light, asleep = L["s_pos"].astype(int), L["s_light"].astype(float), L["s_sleep"].astype(bool)
    eps = S.episodes(L)
    fold = np.arange(len(eps)) % 5

    # Known answer: the counters at t - 1 decide each clock event at t (as checked before this plan was fixed).
    prev = {k: np.r_[np.nan, K[k][:-1]] for k in COUNTERS}
    rule = {0: ((prev["hunger"] >= 24.5) & ~asleep) | ((prev["hunger"] >= 25) & asleep),
            1: ((prev["thirst"] >= 19.5) & ~asleep) | ((prev["thirst"] >= 20) & asleep),
            2: (prev["fatigue"] >= 30) & ~asleep, 3: (prev["fatigue"] <= -10) & asleep}
    known = {G.LB.EVENTS[j]: float((p[ok_ev[:, j]] == y_ev[ok_ev[:, j], j]).mean()) for j, p in rule.items()}
    assert min(known.values()) > 0.999, known
    print("known answer", known, flush=True)

    @torch.no_grad()
    def scores(frames, xx, idx):
        p, _ = C.coarse(model.perception, frames)
        x = tuple(z[idx] for z in xx)
        taken, noop = head(x), head((x[0], x[1], torch.zeros_like(x[2])))
        c = (taken - noop).square().sum(-1)[:, win].sum(-1)
        s = (p.reshape(-1, 81, 16).float() - taken).square().sum(-1)[:, win].sum(-1)
        return C.policy(c, s, pol)

    toks, ctxs, acts, probs, glob, epi = [], [], [], [], [], []
    base = 0
    for e, ep in enumerate(eps):                                      # fair.py's build, without the counterfactuals
        T = int(L["ep_T"][e])
        chunk = D.gather([ep], [[None] * G.P + [(0, 1, t, 1, 1) for t in range(T + 1)]])
        fr = torch.as_tensor(chunk["frames"])
        with torch.no_grad():
            pv = [C.coarse(model.perception, z) for z in fr[0].split(512)]
        p = torch.cat([a for a, _ in pv])[None].float()
        v = torch.cat([b for _, b in pv])[None].float()
        xx, _ = C.inputs(p, v, chunk, G.P, pol["history"])
        idx = torch.arange(1, T + 1)
        probs.append(scores(fr[0, G.P + 1:], xx, idx))
        toks.append(p[0, G.P:].half())
        ctxs.append(xx[1][idx].half()); acts.append(xx[2][idx])
        glob.append(base + idx.numpy()); epi.append(np.full(T, e))
        base += T + 1
        if e % 100 == 0:
            print("episode", e, len(eps), flush=True)
    toks = torch.cat(toks); ctx = torch.cat(ctxs); act = torch.cat(acts); prob = torch.cat(probs)
    glob = np.concatenate(glob); epi = np.concatenate(epi)
    t_in_ep = np.concatenate([np.arange(1, int(L["ep_T"][e]) + 1) for e in range(len(eps))])
    g_lab = off[epi] + t_in_ep                                         # label row of frame t
    assert np.array_equal(g_lab, glob)
    moved = (pos[g_lab] != pos[g_lab - 1]).any(1)
    night = light[g_lab] < 0.5
    ok = ctx.float().reshape(-1, H, 29)[..., 28]
    z_true = np.stack([np.round(K[k][g_lab - 1] * a + b).astype(np.int64) for k, a, b in zip(COUNTERS, SCALE, SHIFT)], 1)
    assert all((z_true[:, k] >= 0).all() and (z_true[:, k] < n).all() for k, n in enumerate(BINS))

    def history(rows):
        hist = torch.as_tensor(glob[rows])[:, None] - torch.arange(H, 0, -1)[None]
        return toks[hist.clamp(min=0)].float() * ok[rows][..., None, None]

    train = np.flatnonzero(fold[epi] != 4); test = np.flatnonzero(fold[epi] == 4)
    strata = asleep[g_lab - 1].astype(int)
    rng = np.random.default_rng(1)
    z_shuf = z_true.copy()
    for rows in (train, test):
        z_shuf[rows] = z_true[permute_within(rows, strata, rng)]

    def fit(z):
        torch.manual_seed(0)
        net = Oracle()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        order = np.random.default_rng(0)
        log = []
        for epoch in range(epochs):
            for g in opt.param_groups:
                g["lr"] = 1e-3 if epoch < epochs - 1 else 3e-4
            perm = order.permutation(train)
            losses = []
            for a in range(0, len(perm), batch):
                rows = perm[a:a + batch]
                q = net(history(rows), ctx[rows].float(), act[rows], onehot(z[rows])).log_softmax(-1)
                loss = -(prob[rows] * q).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss))
            log.append({"epoch": epoch, "train_cross_entropy": float(np.mean(losses))})
            print(log[-1], flush=True)
        net.eval()
        return net, log

    @torch.no_grad()
    def held_out_kl(net, z):
        q = torch.cat([net(history(test[a:a + 4096]), ctx[test[a:a + 4096]].float(), act[test[a:a + 4096]],
                           onehot(z[test[a:a + 4096]])).softmax(-1) for a in range(0, len(test), 4096)])
        return G.kl(prob[test], q).numpy()

    net_s, log_s = fit(z_shuf)
    kl_s = held_out_kl(net_s, z_shuf)
    net_t, log_t = fit(z_true)
    kl_t = held_out_kl(net_t, z_true)
    m, n_ = moved[test], night[test]
    by = lambda x: {"all": float(x.mean()), "moved": float(x[m].mean()), "still": float(x[~m].mean()),
                    "day": float(x[~n_].mean()), "night": float(x[n_].mean())}
    fair = json.loads((HERE / "fair.json").read_text())
    res = {"known_answer": known, "test_frames": len(test), "train_log": {"shuffled": log_s, "true": log_t},
           "kl_shuffled": by(kl_s), "kl_true": by(kl_t), "gain_total": by(kl_s - kl_t),
           "fair_forecaster_kl": fair["forecaster_kl_mean"], "counters": {}, "clocks": {}}
    gains = {}
    for k, name in enumerate(COUNTERS):                              # the true arm with only this counter shuffled
        z = z_true.copy()
        z[test, k] = z_true[permute_within(test, strata, np.random.default_rng(2 + k)), k]
        gains[name] = held_out_kl(net_t, z) - kl_t
        res["counters"][name] = by(gains[name])
        print(name, json.dumps(res["counters"][name]), flush=True)
    ev_test, ok_test = y_ev[g_lab[test]], ok_ev[g_lab[test]]
    for j, clock in enumerate(G.LB.EVENTS):
        g = gains[CLOCK_COUNTER[clock]]
        ev, el = ev_test[:, j], ok_test[:, j]
        res["clocks"][clock] = {
            "counter": CLOCK_COUNTER[clock], "events": int(ev.sum()), "eligible": int(el.sum()),
            "gain_at_events": float(g[ev].mean()), "gain_at_eligible_no_event": float(g[el & ~ev].mean()),
            "gain_at_ineligible": float(g[~el].mean()),
            "gain_per_frame_from_events": float(g[ev].sum() / len(test)),
            "gain_per_frame_from_eligible_no_event": float(g[el & ~ev].sum() / len(test)),
            "first_order_price_per_frame": fair["clocks"][clock]["price_per_frame"]}
        print(clock, json.dumps(res["clocks"][clock]), flush=True)
    price = {c: fair["clocks"][c]["price_per_frame"] for c in CLOCK_COUNTER}
    pix_price = sum(price[c] for c in CLOCK_COUNTER if CLOCK_COUNTER[c] in PIXEL)
    pix_gain = sum(res["counters"][c]["all"] for c in PIXEL)
    res["recovered_fraction"] = {
        "pixel_clocks": pix_gain / pix_price,
        "hunger": res["counters"]["hunger"]["all"] / price["food_down"],
        "thirst": res["counters"]["thirst"]["all"] / price["drink_down"],
        "fatigue": res["counters"]["fatigue"]["all"] / (price["energy_down_awake"] + price["energy_up_asleep"]),
        "recover": res["counters"]["recover"]["all"] / price["recovery"]}
    res["memory_share_of_shuffled_kl"] = {"pixel_clocks": pix_gain / res["kl_shuffled"]["all"],
                                          "recover": res["counters"]["recover"]["all"] / res["kl_shuffled"]["all"]}
    (HERE / "oracle.json").write_text(json.dumps(res, indent=1) + "\n")
    print("recovered fraction", json.dumps(res["recovered_fraction"]), "memory share",
          json.dumps(res["memory_share_of_shuffled_kl"]), flush=True)


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    main()
