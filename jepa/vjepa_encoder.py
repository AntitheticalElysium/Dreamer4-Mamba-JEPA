"""Frozen V-JEPA 2.1 ViT-B encoder wrapper. Verified facts (jepa/verify_vjepa.py):
encoder is 86.8M params, embed_dim 768, patch 16, RoPE; input must be 5-D (B,C,T,H,W)
with T even (tubelet=2). Per-frame latent = replicate frame to T=2 -> spatial tokens.
res 256 -> 256 tokens/frame. Pool=True mean-pools to a 768-d per-frame vector.
"""
import os
import sys
import torch
import torch.nn.functional as F

VJEPA_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "vjepa2")
if VJEPA_REPO not in sys.path:
    sys.path.insert(0, VJEPA_REPO)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VJEPAEncoder:
    def __init__(self, res=256, pool="mean", grid=4, device="cuda"):
        """pool: 'mean' (global -> 768) | 'grid' (GxG spatial pool -> G*G*768,
        preserves coarse spatial layout, needed for spatially-structured envs like
        Crafter where global pooling washes out dynamics) | 'none' (all patch tokens)."""
        from src.hub.backbones import vjepa2_1_vit_base_384
        enc = vjepa2_1_vit_base_384(pretrained=True)
        enc = enc[0] if isinstance(enc, (tuple, list)) else enc  # (encoder, predictor)
        self.enc = enc.to(device).eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.res, self.pool, self.grid, self.device = res, pool, grid, device
        self.mean, self.std = _MEAN.to(device), _STD.to(device)
        self.n_tokens = (res // 16) ** 2
        self.dim = {"mean": 768, "grid": 768 * grid * grid, "none": 768 * self.n_tokens}[pool]

    def _prep(self, x):  # x (B,C,H,W) uint8-scaled -> resized+normalized
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def encode(self, frames_hwc_uint8, batch=32, motion=False, prev_frames=None):
        """frames: np.uint8 [N,H,W,C] (in sequence order) -> latents [N, self.dim].
        motion=True: each latent = V-JEPA of the real 2-frame clip [f_{t-1}, f_t]. If
        prev_frames is given (np.uint8 [N,H,W,C]) it is used as f_{t-1} (lets the caller
        respect episode/segment boundaries); otherwise f_{t-1} = global shift by 1."""
        out = []
        x_all = torch.from_numpy(frames_hwc_uint8).to(self.device).float() / 255.0
        x_all = x_all.permute(0, 3, 1, 2)  # N,C,H,W
        if not motion:
            prev_all = None
        elif prev_frames is not None:
            prev_all = torch.from_numpy(prev_frames).to(self.device).float().permute(0, 3, 1, 2) / 255.0
        else:
            prev_all = torch.cat([x_all[:1], x_all[:-1]], 0)  # f_{t-1}
        n = self.res // 16
        for i in range(0, x_all.shape[0], batch):
            x = self._prep(x_all[i:i + batch])
            if motion:
                xp = self._prep(prev_all[i:i + batch])
                x = torch.stack([xp, x], dim=2)          # B,C,T=2,H,W (real motion)
            else:
                x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # B,C,T=2,H,W (duplicate)
            y = self.enc(x)                            # B,N,768
            if self.pool == "mean":
                y = y.mean(dim=1)                      # B,768
            elif self.pool == "grid":
                B = y.shape[0]
                yg = y.transpose(1, 2).reshape(B, 768, n, n)         # B,768,n,n
                yg = F.adaptive_avg_pool2d(yg, self.grid)            # B,768,G,G
                y = yg.reshape(B, -1)                                # B, 768*G*G
            else:  # 'none' -> flat patch tokens
                y = y.reshape(y.shape[0], -1)
            out.append(y.float().cpu())
        return torch.cat(out, 0)
