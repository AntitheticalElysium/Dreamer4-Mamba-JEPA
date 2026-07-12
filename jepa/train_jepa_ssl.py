"""Faithful CNN-JEPA self-supervised pretraining of the ConvBackbone on Crafter frames,
following github.com/kaland313/CNN-JEPA (train_ijepacnn.py). NO labels -- this is what makes
the encoder GENERALIZE (our earlier supervised BC/reward encoder overfit GCPPO and was random
on unseen worlds).

Algorithm (per their train_val_step):
  1. mask patches at the 4x4 feature-map resolution (random, mask_ratio=0.6).
  2. encode ONLY visible patches with the SparK masked-conv backbone (masked positions zeroed
     after each conv so they can't leak).
  3. fill masked slots with a learnable mask-token; a small depthwise-separable conv predictor
     predicts the masked-patch features.
  4. target = EMA-momentum backbone on the FULL image, detached. momentum cosine 0.996 -> 1.0.
  5. loss = smooth_l1 on L2-NORMALIZED features (over channels), MASKED patches only.

Saves the pretrained backbone -> ckpt/ssl_backbone.pt for the world model to build on.
"""
import os, sys, argparse, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from conv_encoder import ConvBackbone, LayerNorm2d, EMATarget
from train_jepa_wm import load_ppo_frames


def cosine_momentum(step, total, base=0.996, final=1.0):
    return final - (final - base) * (math.cos(math.pi * step / total) + 1) / 2


class Predictor(nn.Module):
    """3x depthwise-separable conv (kernel 3) predictor -- CNN-JEPA IN-100 config."""
    def __init__(self, c, n_layers=3, k=3):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv2d(c, c, k, padding=k // 2, groups=c),  # depthwise
                       nn.Conv2d(c, c, 1),                            # pointwise
                       LayerNorm2d(c), nn.SiLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def random_mask(B, f, keep, dev, rng):
    """context mask (B,1,f,f) True=visible with `keep` visible patches; target = complement."""
    idx = torch.from_numpy(np.stack([rng.permutation(f * f)[:keep] for _ in range(B)])).to(dev)
    ctx = torch.zeros(B, f * f, dtype=torch.bool, device=dev).scatter_(1, idx, True).view(B, 1, f, f)
    return ctx, ~ctx


@torch.no_grad()
def probe_action_decode(backbone, obs_u8, act, dev, epochs=25):
    """Sanity metric (in-PPO-dist): frozen backbone features -> action, held-out acc."""
    frames = obs_u8.reshape(-1, 3, 64, 64); a = torch.from_numpy(act.reshape(-1)).to(dev)
    embs = []
    for i in range(0, frames.shape[0], 1024):
        x = torch.from_numpy(frames[i:i + 1024]).to(dev).float() / 255.0
        embs.append(backbone(x).flatten(1).cpu())
    z = torch.cat(embs).to(dev); N = z.shape[0]; s = int(N * 0.8)
    z = (z - z[:s].mean(0)) / (z[:s].std(0) + 1e-6)
    with torch.enable_grad():
        net = nn.Sequential(nn.Linear(z.shape[1], 512), nn.GELU(), nn.Linear(512, 17)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); rng = np.random.default_rng(0)
        for _ in range(epochs):
            for i in range(0, s, 256):
                b = rng.integers(0, s, 256)
                loss = F.cross_entropy(net(z[b]), a[b]); opt.zero_grad(); loss.backward(); opt.step()
        return (net(z[s:]).argmax(-1) == a[s:]).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfiles", type=int, default=40)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask_ratio", type=float, default=0.6)
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "ckpt", "ssl_backbone.pt"))
    ap.add_argument("--eval_every", type=int, default=2000)
    args = ap.parse_args()
    dev = "cuda"

    obs, act, _ = load_ppo_frames(args.nfiles)
    frames = obs.reshape(-1, 3, 64, 64)                       # (N,3,64,64) uint8, all frames
    N = frames.shape[0]; f = 4; keep = round(f * f * (1 - args.mask_ratio))
    print(f"SSL frames={N} | fmap {f}x{f} keep={keep}/{f*f} (mask_ratio {args.mask_ratio})", flush=True)

    backbone = ConvBackbone().to(dev)
    ema = EMATarget(backbone, tau=0.996); ema.target.to(dev)
    mask_token = nn.Parameter(torch.zeros(1, backbone.out_ch, 1, 1, device=dev)); nn.init.trunc_normal_(mask_token, std=0.02)
    predictor = Predictor(backbone.out_ch).to(dev)
    params = list(backbone.parameters()) + list(predictor.parameters()) + [mask_token]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    print(f"backbone {sum(p.numel() for p in backbone.parameters())/1e6:.2f}M + "
          f"predictor {sum(p.numel() for p in predictor.parameters())/1e6:.2f}M", flush=True)

    rng = np.random.default_rng(0); t0 = time.time()
    for st in range(1, args.steps + 1):
        backbone.train(); predictor.train()
        b = rng.integers(0, N, size=args.bs)
        x = torch.from_numpy(frames[b]).to(dev).float() / 255.0
        ctx, tgt = random_mask(args.bs, f, keep, dev, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            up = ctx.repeat_interleave(64 // f, 2).repeat_interleave(64 // f, 3).to(x.dtype)
            feat = backbone(x * up, active=ctx)                      # visible-only encode
            feat_m = torch.where(ctx.expand_as(feat), feat, mask_token.expand_as(feat).to(feat.dtype))
            p = predictor(feat_m)                                    # predict masked features
            with torch.no_grad():
                h = ema.target(x, active=None)                       # full image, EMA target
            p = F.normalize(p.float(), dim=1); h = F.normalize(h.float(), dim=1)
            loss = F.smooth_l1_loss(p, h, reduction='none').sum(1, keepdim=True)   # (B,1,f,f)
            loss = (loss * tgt).sum() / (tgt.sum() + 1e-8)           # masked patches only
        opt.zero_grad(); loss.backward(); opt.step()
        ema.update(backbone, tau=cosine_momentum(st, args.steps))

        if st % args.eval_every == 0 or st == 1:
            backbone.eval()
            with torch.no_grad():
                fstd = backbone(torch.from_numpy(frames[b]).to(dev).float() / 255.0).flatten(1).std(0).mean().item()
            acc = probe_action_decode(backbone, obs[:8000], act[:8000], dev)
            print(f"step {st:6d} [{time.time()-t0:5.0f}s] loss {loss.item():.4f} | featstd {fstd:.3f} "
                  f"| in-dist act-acc {acc:.3f} (overfit-enc was 0.44)", flush=True)

    torch.save({"backbone": backbone.state_dict(), "embed_note": "ssl-pretrained ConvBackbone"}, args.out)
    print(f"saved SSL backbone -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
