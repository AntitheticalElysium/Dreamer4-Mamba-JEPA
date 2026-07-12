"""Q1: is the Mamba the bottleneck, or the loss?  WM-fidelity ablation on HELD-OUT latents.

Trains small JEPADynamics variants (depth x width x loss) on the offline motion latents and
measures next-latent prediction quality on a held-out split. If ratio barely moves with
depth/width but drops with cosine, the Mamba is exonerated and the loss/representation is the
limiter. No policy involved -- pure dynamics fidelity, a few min per config.

Metrics on held-out:
  ratio  = MSE(pred_t, z_{t+1}) / MSE(z_t, z_{t+1})   (persistence baseline; <1 means the WM
           beats "predict no change"; lower is better)
  cos    = mean cosine(pred_t, z_{t+1})               (Dreamer-CDP's target; higher is better)
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dynamics import JEPADynamics, curvature_loss


def make_seqs(z, a, T, n, rng):
    hi = len(a) - T - 1
    s = rng.integers(0, hi, size=n)
    idx = s[:, None] + np.arange(T)[None, :]
    return z[idx], a[idx]


def evaluate(model, zt, at, dev):
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        preds, _ = model(zt, at)                 # (K,B,T,D)
        pred = preds.float().mean(0)             # ensemble mean
    tgt = zt.roll(-1, 1)                          # z_{t+1}
    # drop last timestep (roll wraps)
    pred, tgt, cur = pred[:, :-1], tgt[:, :-1], zt[:, :-1]
    mse_pred = F.mse_loss(pred, tgt).item()
    mse_pers = F.mse_loss(cur, tgt).item()
    cos = F.cosine_similarity(pred, tgt, dim=-1).mean().item()
    return mse_pred / mse_pers, cos


def run_cfg(z, a, dev, n_layers, d_model, loss_kind, steps, T, D, A, seed=0):
    rng = np.random.default_rng(seed)
    ntr = int(len(a) * 0.9)
    ztr, atr = z[:ntr], a[:ntr]; zva, ava = z[ntr:], a[ntr:]
    model = JEPADynamics(latent_dim=D, action_dim=A, d_model=d_model, n_layers=n_layers, n_heads=5).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    zva_t, ava_t = make_seqs(zva, ava, T, 512, rng)
    zva_t = torch.tensor(zva_t, device=dev); ava_t = torch.tensor(ava_t, device=dev)
    model.train()
    for st in range(steps):
        zb, ab = make_seqs(ztr, atr, T, 32, rng)
        zb = torch.tensor(zb, device=dev); ab = torch.tensor(ab, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds, _ = model(zb, ab)
            tgt = zb.roll(-1, 1).unsqueeze(0).expand_as(preds)
            if loss_kind == "cosine":
                pl = (1.0 - F.cosine_similarity(preds, tgt, dim=-1)).mean()
            else:
                pl = F.mse_loss(preds, tgt)
            loss = pl + 0.01 * curvature_loss(preds.mean(0))
        opt.zero_grad(); loss.backward(); opt.step()
    ratio, cos = evaluate(model, zva_t, ava_t, dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model; torch.cuda.empty_cache()
    return ratio, cos, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "crafter_motion_full.npz"))
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--T", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"

    d = np.load(args.data)
    lat = d["latents"].astype(np.float32)
    mean, std = lat.mean(0), lat.std(0) + 1e-6
    z = ((lat - mean) / std).astype(np.float32)
    a = d["actions"].astype(np.int64)
    D, A = z.shape[1], int(a.max()) + 1
    print(f"latents {z.shape} actions {a.shape} D={D} A={A}  steps={args.steps} T={args.T}\n")

    # (n_layers, d_model, loss)
    configs = [
        (2, 512, "mse"),      # current
        (4, 512, "mse"),      # deeper
        (2, 768, "mse"),      # wider
        (4, 768, "mse"),      # deeper+wider
        (2, 512, "cosine"),   # current arch, Dreamer-CDP loss
        (4, 768, "cosine"),   # bigger + cosine
    ]
    print(f"{'layers':>6} {'d_model':>7} {'loss':>7} {'params':>9} {'ratio↓':>8} {'cos↑':>7}")
    for nl, dm, lk in configs:
        ratio, cos, npar = run_cfg(z, a, dev, nl, dm, lk, args.steps, args.T, D, A)
        print(f"{nl:>6} {dm:>7} {lk:>7} {npar/1e6:>8.2f}M {ratio:>8.3f} {cos:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
