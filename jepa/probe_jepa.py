"""Honesty probe in the JEPA latent. Open-loop rollout of the Mamba-JEPA dynamics on
REAL held-out latent sequences; per step log candidate signals vs TRUE error:
  u_s     : multi-head disagreement (the AHEAD-style signal)
  delta   : Mamba Delta (due-diligence: expected to stay flat)
  curv    : predicted-latent-trajectory curvature (geometric candidate from
            "On Training in Imagination" — a possible long-horizon signal)
  m       : per-step latent RMS change (dynamism normalizer)
True error = MSE(open-loop predicted latent, real next latent) in std-normalized space.

Key question: does u_s DEGRADE over the horizon here too (the attractor false-negative),
and does any signal hold up in the long-horizon regime?
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from dynamics import JEPADynamics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def auc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    return (r[label == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.inference_mode()
def probe_window(model, L, A, s, C, H, dev):
    states = model.init_state(1, C + H + 2, dev)
    model.set_delta_capture(False)
    for t in range(C - 1):
        model.step(L[s + t:s + t + 1].unsqueeze(0), A[s + t:s + t + 1], states)
    model.set_delta_capture(True)
    preds, _h, states = model.step(L[s + C - 1:s + C].unsqueeze(0), A[s + C - 1:s + C], states)
    cur = preds.mean(0)                       # (1,1,D) predicted latent for index s+C
    prev, prev_v = None, None
    rows = []
    for k in range(H):
        real = L[s + C + k:s + C + k + 1].unsqueeze(0)     # (1,1,D)
        err = F.mse_loss(cur, real).item()
        u_s = preds.var(0).mean().item()
        d = model.read_delta()
        d_mean = float(d.mean()) if d is not None else float("nan")
        d_max = float(d.max()) if d is not None else float("nan")
        m = 0.0 if prev is None else (cur - prev).pow(2).mean().sqrt().item()
        v = None if prev is None else (cur - prev)
        curv = 0.0
        if v is not None and prev_v is not None:
            curv = float(1.0 - F.cosine_similarity(v.flatten(), prev_v.flatten(), dim=0))
        rows.append(dict(k=k + 1, err=err, u_s=u_s, delta_mean=d_mean, delta_max=d_max, curv=curv, m=m))
        prev, prev_v = cur, v
        if k < H - 1:
            preds, _h, states = model.step(cur, A[s + C + k:s + C + k + 1], states)
            cur = preds.mean(0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join(REPO, "ckpt", "jepa_dyn.pt"))
    ap.add_argument("--data", type=str, default=os.path.join(REPO, "data", "mm_vjepa_train.npz"))
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--max_windows", type=int, default=150)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)  # our own file (has numpy mean/std)
    model = JEPADynamics(latent_dim=ck["latent_dim"], action_dim=ck["action_dim"], n_heads=5).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    d = np.load(args.data)
    lat = (d["latents"].astype(np.float32) - ck["mean"]) / ck["std"]
    act = d["actions"].astype(np.int64)
    n = len(act); split = int(n * 0.8)          # probe the held-out (val) tail
    L = torch.tensor(lat, device=dev); A = torch.tensor(act, device=dev)

    need = args.ctx + args.horizon + 1
    starts = list(range(split, n - need, 12))[:args.max_windows]
    print(f"probing {len(starts)} held-out windows (ctx={args.ctx}, H={args.horizon})")
    rows = []
    for wi, s in enumerate(starts):
        for r in probe_window(model, L, A, s, args.ctx, args.horizon, dev):
            r["window"] = wi; rows.append(r)

    keys = ["window", "k", "err", "u_s", "delta_mean", "delta_max", "curv", "m"]
    data = {kk: np.array([r[kk] for r in rows], dtype=np.float32) for kk in keys}
    out = os.path.join(REPO, "probe_logs", "jepa_openloop.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True); np.savez(out, **data)

    err = data["err"]; k = data["k"]
    print(f"\nrecords={len(err)}  err by horizon: k1={err[k==1].mean():.3f} -> k{int(k.max())}={err[k==k.max()].mean():.3f}")
    for name in ["u_s", "delta_mean", "delta_max", "curv"]:
        s_all = spearmanr(data[name], err).correlation
        def bucket(lo, hi):
            m = (k >= lo) & (k <= hi); hi_e = (err[m] > np.quantile(err[m], 0.75)).astype(int)
            return spearmanr(data[name][m], err[m]).correlation, auc(data[name][m], hi_e)
        re, ae = bucket(1, 5); rl, al = bucket(max(1, int(k.max()) - 4), int(k.max()))
        print(f"{name:11s} rho_all={s_all:+.3f} | early rho/auc={re:+.3f}/{ae:.3f}  late rho/auc={rl:+.3f}/{al:.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
