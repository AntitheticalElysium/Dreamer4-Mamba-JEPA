"""Q2: are we over-committed to the frozen V-JEPA encoder?

Decisive test: how much CONTROL-relevant info does each representation expose, measured by
held-out PPO-action predictability (z -> action). Three arms on the SAME frames:
  A. frozen V-JEPA, mean-pool (768)         -- what we use now
  B. frozen V-JEPA, grid8-pool (49152)      -- frozen but spatial
  C. tiny CNN trained from raw pixels (motion 2-frame stack) -- "train our own encoder"

If C >> A,B: the frozen off-distribution ViT is leaving control info on the table for Crafter
=> over-committed, train our own (or a trainable adapter). If C ~ A,B: frozen is adequate.
The CNN is a fair motion comparison (6-channel [f_{t-1}, f_t] input, DreamerV3-style).
"""
import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from vjepa_encoder import VJEPAEncoder


def load_ppo(nfiles):
    files = sorted(glob.glob(os.path.join(REPO, "data", "crafter_ppo", "ex", "*.pt")))[:nfiles]
    frames, prevs, acts = [], [], []
    for f in files:
        for t in torch.load(f, map_location="cpu", weights_only=False):
            hwc = np.clip(t["obs"].numpy().transpose(0, 2, 3, 1) * 255, 0, 255).astype(np.uint8)
            frames.append(hwc); prevs.append(np.concatenate([hwc[:1], hwc[:-1]], 0))
            acts.append(t["actions"].reshape(-1).numpy().astype(np.int64))
    return np.concatenate(frames), np.concatenate(prevs), np.concatenate(acts)


def probe_latent(z, a, dev, epochs=30):
    """MLP probe z->action, held-out acc (last 20%)."""
    n = len(a); s = int(n * 0.8)
    z = torch.tensor(z, device=dev); a = torch.tensor(a, device=dev)
    zm, zs = z[:s].mean(0), z[:s].std(0) + 1e-6
    z = (z - zm) / zs
    net = nn.Sequential(nn.Linear(z.shape[1], 512), nn.GELU(), nn.Linear(512, 17)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        for i in range(0, s, 256):
            b = rng.permutation(s)[i:i + 256] if i == 0 else rng.integers(0, s, 256)
            loss = F.cross_entropy(net(z[b]), a[b]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        te = (net(z[s:]).argmax(-1) == a[s:]).float().mean().item()
    return te


class TinyCNN(nn.Module):
    """DreamerV3-style small conv encoder on 6ch (prev+cur) 64x64 -> 768 -> action logits."""
    def __init__(self, out=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1), nn.SiLU(),     # 64->32
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),    # 32->16
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),   # 16->8
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),  # 8->4
        )
        self.head = nn.Linear(256 * 4 * 4, out)
        self.cls = nn.Sequential(nn.GELU(), nn.Linear(out, 17))

    def emb(self, x):
        return self.head(self.conv(x).flatten(1))

    def forward(self, x):
        return self.cls(self.emb(x))


def probe_cnn(frames, prevs, acts, dev, epochs=15):
    n = len(acts); s = int(n * 0.8)
    x = np.concatenate([prevs, frames], axis=-1).transpose(0, 3, 1, 2).astype(np.float32) / 255.0  # N,6,64,64
    a = torch.tensor(acts, device=dev)
    net = TinyCNN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        idx = rng.permutation(s)
        for i in range(0, s, 256):
            b = idx[i:i + 256]
            xb = torch.tensor(x[b], device=dev)
            loss = F.cross_entropy(net(xb), a[b]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(s, n, 512):
            xb = torch.tensor(x[i:i + 512], device=dev)
            correct += (net(xb).argmax(-1) == a[i:i + 512]).sum().item()
    return correct / (n - s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfiles", type=int, default=4)  # ~16k frames
    ap.add_argument("--skip_grid", action="store_true")  # grid8 arm is VRAM-heavy; skip to co-run with KL
    args = ap.parse_args()
    dev = "cuda"
    frames, prevs, acts = load_ppo(args.nfiles)
    maj = np.bincount(acts, minlength=17).max() / len(acts)
    print(f"frames={len(frames)}  majority-class baseline acc={maj:.3f}\n")

    print(f"{'arm':>28} {'dim':>8} {'HELDOUT action-acc':>18}")
    arms = [("A frozen-VJEPA mean", dict(pool="mean"))]
    if not args.skip_grid:
        arms.append(("B frozen-VJEPA grid8", dict(pool="grid", grid=8)))
    for name, kw in arms:
        enc = VJEPAEncoder(res=256, device=dev, **kw)
        z = enc.encode(frames, motion=True, prev_frames=prevs).numpy().astype(np.float32)
        acc = probe_latent(z, acts, dev)
        print(f"{name:>28} {z.shape[1]:>8} {acc:>18.3f}", flush=True)
        del enc, z; torch.cuda.empty_cache()

    acc_cnn = probe_cnn(frames, prevs, acts, dev)
    print(f"{'C tiny-CNN (trained)':>28} {768:>8} {acc_cnn:>18.3f}", flush=True)
    print(f"\n(baseline {maj:.3f}) -- if C >> A,B, frozen V-JEPA is the ceiling for Crafter.")


if __name__ == "__main__":
    main()
