"""Diagnostic probe: how much CONTROL-relevant info does each V-JEPA pooling preserve?
Encode competent-PPO frames with mean / grid4 / grid8 pooling and train a simple MLP to
predict the PPO action from the latent alone (z -> action). Higher held-out accuracy =
the latent preserves more of what a competent policy needs. If richer spatial pooling
wins big over mean-pool, the representation is the ceiling (motivates the encoder shore-up).
"""
import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from vjepa_encoder import VJEPAEncoder


def load_ppo(nfiles=2):
    files = sorted(glob.glob(os.path.join(REPO, "data", "crafter_ppo", "ex", "*.pt")))[:nfiles]
    frames, prevs, acts = [], [], []
    for f in files:
        for t in torch.load(f, map_location="cpu", weights_only=False):
            hwc = np.clip(t["obs"].numpy().transpose(0, 2, 3, 1) * 255, 0, 255).astype(np.uint8)
            frames.append(hwc); prevs.append(np.concatenate([hwc[:1], hwc[:-1]], 0))
            acts.append(t["actions"].reshape(-1).numpy().astype(np.int64))
    return np.concatenate(frames), np.concatenate(prevs), np.concatenate(acts)


def probe(z, a, dev, epochs=30):
    n = len(a); s = int(n * 0.8)
    z = torch.tensor(z, device=dev); a = torch.tensor(a, device=dev)
    zm, zs = z[:s].mean(0), z[:s].std(0) + 1e-6
    z = (z - zm) / zs
    net = nn.Sequential(nn.Linear(z.shape[1], 512), nn.GELU(), nn.Linear(512, 17)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        idx = rng.permutation(s)
        for i in range(0, s, 256):
            b = idx[i:i + 256]
            loss = F.cross_entropy(net(z[b]), a[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        tr = (net(z[:s]).argmax(-1) == a[:s]).float().mean().item()
        te = (net(z[s:]).argmax(-1) == a[s:]).float().mean().item()
    return tr, te


def main():
    dev = "cuda"
    frames, prevs, acts = load_ppo(nfiles=2)
    print(f"frames={len(frames)}  action balance top: {np.bincount(acts, minlength=17)[:6].tolist()}...")
    # majority-class baseline
    maj = np.bincount(acts).max() / len(acts)
    print(f"majority-class acc baseline: {maj:.3f}")
    for name, kw in [("mean", dict(pool="mean")), ("grid4", dict(pool="grid", grid=4)),
                     ("grid8", dict(pool="grid", grid=8))]:
        enc = VJEPAEncoder(res=256, device=dev, **kw)
        z = enc.encode(frames, motion=True, prev_frames=prevs).numpy().astype(np.float32)
        tr, te = probe(z, acts, dev)
        print(f"{name:6s} dim={z.shape[1]:6d}  train-acc {tr:.3f}  HELDOUT-acc {te:.3f}")
        del enc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
