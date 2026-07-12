"""CNN encoder for the Mamba-JEPA world model, now with a mask-aware backbone so it can be
pretrained with the FAITHFUL CNN-JEPA self-supervised objective (Kalapos & Gyires-Toth 2024,
github.com/kaland313/CNN-JEPA) before it feeds the dynamics.

Why this rewrite: our first attempt trained the encoder with temporal-prediction + supervised
BC/reward losses on narrow GCPPO data and NO masking -> it overfit the PPO distribution (0.44
in-dist action-decode) and was random on unseen worlds. CNN-JEPA's generalization comes from a
self-supervised MASKED-PREDICTION objective with an EMA target and normalized features, no labels.

Components here:
  - LayerNorm2d: channel-wise LayerNorm (per spatial position). This is DreamerV3's actual
    "channel-wise layer normalization" (arXiv 2301.04104) -- NOT GroupNorm(1) which mixes spatial
    stats. Per-position norm also means masked positions never corrupt visible positions' stats,
    which is required for the SSL masking to be clean.
  - ConvBackbone: the verified DreamerV3 CNN ({32,64,128,256}, k4/s2, LayerNorm2d, SiLU), with an
    optional SparK-style masking path: after each conv, zero the masked positions (sp_conv_forward)
    so the masked input can't leak. active mask is at the 4x4 feature-map resolution.
  - ConvEncoder: ConvBackbone -> flatten -> Linear(embed_dim), the dense per-frame embedding used
    by the Mamba dynamics.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm: normalize over C for each (B,H,W) position (DreamerV3 spec)."""
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):                       # (B,C,H,W)
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ConvBackbone(nn.Module):
    """DreamerV3 CNN with optional SparK masking. forward(x, active) where active is a
    (B,1,f,f) context mask (True = visible); when given, conv outputs at masked positions are
    zeroed after every layer (github.com/kaland313/CNN-JEPA sparse_encoder.sp_conv_forward)."""
    def __init__(self, ch=(32, 64, 128, 256), in_ch=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        c = in_ch
        for oc in ch:
            self.convs.append(nn.Conv2d(c, oc, 4, stride=2, padding=1))
            self.norms.append(LayerNorm2d(oc))
            c = oc
        self.out_ch = ch[-1]

    def forward(self, x, active=None):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            if active is not None:
                a = active.repeat_interleave(x.shape[2] // active.shape[2], 2) \
                          .repeat_interleave(x.shape[3] // active.shape[3], 3).to(x.dtype)
                x = x * a                       # SparK: zero masked conv outputs
            x = F.silu(norm(x))
            if active is not None:
                x = x * a                       # keep masked positions zero through norm/act
        return x                                # (B, out_ch, 4, 4)


class ConvEncoder(nn.Module):
    """Per-frame embedding for the world model: ConvBackbone -> flatten -> Linear(embed_dim).
    embed_dim default 512 (= Mamba d_model / DreamerV3-S deter); ablation knob."""
    def __init__(self, embed_dim=512, ch=(32, 64, 128, 256), in_ch=3):
        super().__init__()
        self.backbone = ConvBackbone(ch, in_ch)
        self.proj = nn.Linear(ch[-1] * 4 * 4, embed_dim)
        self.embed_dim = embed_dim
        self.in_ch = in_ch

    def forward(self, x):
        return self.proj(self.backbone(x).flatten(1))


class EMATarget:
    """Stop-grad EMA copy of a module (I-JEPA / BYOL momentum target)."""
    def __init__(self, module, tau=0.996):
        self.target = copy.deepcopy(module).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.tau = tau

    @torch.no_grad()
    def update(self, module, tau=None):
        t = self.tau if tau is None else tau
        for pt, ps in zip(self.target.parameters(), module.parameters()):
            pt.mul_(t).add_(ps.detach(), alpha=1.0 - t)
        for bt, bs in zip(self.target.buffers(), module.buffers()):
            bt.copy_(bs)


def variance_reg(z, eps=1e-4, target_std=1.0):
    """VICReg variance term (collapse backstop): hinge keeping per-dim std >= target."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(target_std - std).mean()
