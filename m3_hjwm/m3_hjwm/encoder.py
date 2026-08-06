from __future__ import annotations
import copy
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .utils import RMSNorm, ema_update


class ResidualConvBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        y = F.silu(self.norm1(x))
        y = self.conv1(y)
        y = F.silu(self.norm2(y))
        return x + self.conv2(y)


class DenseVisualEncoder(nn.Module):
    """Small dense encoder for 64x64 control environments.

    Produces an 8x8 token grid at patch_size=8 and intermediate feature maps for
    deep JEPA supervision. This is intentionally not a frozen internet-video model:
    rare task variables must be allowed to shape the representation online.
    """
    def __init__(self, in_channels: int, token_dim: int, patch_size: int, depth: int):
        super().__init__()
        if patch_size not in (4, 8, 16):
            raise ValueError("reference implementation supports patch sizes 4, 8, or 16")
        stages = int(torch.log2(torch.tensor(float(patch_size))).item())
        channels = [min(token_dim, 32 * (2 ** i)) for i in range(stages)]
        layers = []
        c = in_channels
        for oc in channels:
            layers.extend([
                nn.Conv2d(c, oc, 4, stride=2, padding=1),
                nn.GroupNorm(1, oc),
                nn.SiLU(),
                ResidualConvBlock(oc),
            ])
            c = oc
        self.down = nn.ModuleList(layers)
        self.project = nn.Conv2d(c, token_dim, 1)
        self.depth = depth
        self.token_dim = token_dim

    def forward(self, obs: Tensor) -> tuple[Tensor, list[Tensor]]:
        # Accept uint8 or [0,1] float, [B,C,H,W].
        x = obs.float()
        if obs.dtype == torch.uint8:
            x = x / 255.0
        intermediates: list[Tensor] = []
        for layer in self.down:
            x = layer(x)
            if isinstance(layer, ResidualConvBlock):
                intermediates.append(x)
        x = self.project(x)
        tokens = x.flatten(2).transpose(1, 2)  # [B,N,D]
        return tokens, intermediates


class MaskTokenInjector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, tokens: Tensor, target_mask: Tensor) -> Tensor:
        mask = target_mask.flatten(1).unsqueeze(-1)
        return torch.where(mask, self.mask_token.to(tokens.dtype), tokens)


class JEPATargetEncoder(nn.Module):
    """EMA target encoder. It is never updated by gradients."""
    def __init__(self, context: DenseVisualEncoder, decay: float):
        super().__init__()
        self.encoder = copy.deepcopy(context)
        self.decay = decay
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    @torch.no_grad()
    def update(self, context: DenseVisualEncoder) -> None:
        ema_update(self.encoder, context, self.decay)

    @torch.no_grad()
    def forward(self, obs: Tensor) -> tuple[Tensor, list[Tensor]]:
        return self.encoder(obs)
