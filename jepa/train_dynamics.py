"""Train the Mamba-JEPA world model on cached V-JEPA latents: dynamics (JEPA regression)
+ reward (symlog-two-hot) + termination (BCE), when rewards/terminals are present.
Gate metric: next-latent MSE vs the copy baseline (z_{t+1}=z_t) on held-out data.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from dynamics import JEPADynamics, curvature_loss

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_windows(n, T, stride):
    return list(range(0, n - T - 1, stride))


def batches(starts, bs, rng):
    starts = starts.copy(); rng.shuffle(starts)
    for i in range(0, len(starts), bs):
        yield starts[i:i + bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=os.path.join(REPO, "data", "crafter_vjepa.npz"))
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--curv", type=float, default=0.01)
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "ckpt", "jepa_wm.pt"))
    args = ap.parse_args()
    dev = "cuda"

    d = np.load(args.data)
    lat = d["latents"].astype(np.float32)
    act = d["actions"].astype(np.int64)
    adim = int(d["action_dim"])
    n = len(act); split = int(n * 0.8)
    mean = lat[:split].mean(0, keepdims=True); std = lat[:split].std(0, keepdims=True) + 1e-6
    L = torch.tensor((lat - mean) / std, device=dev); A = torch.tensor(act, device=dev)
    has_rew = "rewards" in d.files
    if has_rew:
        R = torch.tensor(d["rewards"].astype(np.float32), device=dev)
        Tm = torch.tensor(d["terminals"].astype(np.float32), device=dev)
        print(f"reward nonzero {100 * (d['rewards'] != 0).mean():.2f}%  terminals {int(d['terminals'].sum())}")
    print(f"data: latents={lat.shape} action_dim={adim} split@{split}")

    model = JEPADynamics(latent_dim=lat.shape[1], action_dim=adim, n_heads=5).to(dev)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    train_starts = make_windows(split, args.seq, args.seq // 2)

    def eval_val():
        model.eval()
        vs = make_windows(n - split, args.seq, args.seq)
        me, ce, us = [], [], []
        with torch.inference_mode():
            for s0 in vs:
                s = split + s0
                z = L[s:s + args.seq].unsqueeze(0); a = A[s:s + args.seq].unsqueeze(0)
                tgt = L[s + 1:s + args.seq + 1].unsqueeze(0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds, _ = model(z, a); preds = preds.float()
                me.append(F.mse_loss(preds.mean(0), tgt).item())
                ce.append(F.mse_loss(z, tgt).item())
                us.append(preds.var(0).mean().item())
        model.train()
        return np.mean(me), np.mean(ce), np.mean(us)

    step = 0
    while step < args.steps:
        for bs_starts in batches(train_starts, args.bs, rng):
            z = torch.stack([L[s:s + args.seq] for s in bs_starts])
            a = torch.stack([A[s:s + args.seq] for s in bs_starts])
            tgt = torch.stack([L[s + 1:s + args.seq + 1] for s in bs_starts])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds, h = model(z, a)
                mse = F.mse_loss(preds, tgt.unsqueeze(0).expand_as(preds))
                curv = curvature_loss(preds.mean(0))
                loss = mse + args.curv * curv
                rl_i = tl_i = 0.0
                if has_rew:
                    r = torch.stack([R[s:s + args.seq] for s in bs_starts])
                    tm = torch.stack([Tm[s:s + args.seq] for s in bs_starts])
                    rew_logits, term_logit = model.reward_term(h)
                    rew_loss = model.symlog(rew_logits, r)
                    term_loss = F.binary_cross_entropy_with_logits(term_logit, tm)
                    loss = loss + rew_loss + term_loss
                    rl_i, tl_i = rew_loss.item(), term_loss.item()
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 500 == 0 or step == 1:
                me, ce, us = eval_val()
                print(f"step {step:5d}  mse {mse.item():.4f} rew {rl_i:.3f} term {tl_i:.3f} | "
                      f"val_model {me:.4f} val_copy {ce:.4f} ratio {me/ce:.3f}  u_s {us:.4f}")
            if step >= args.steps:
                break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                "action_dim": adim, "latent_dim": lat.shape[1]}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
