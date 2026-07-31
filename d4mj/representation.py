import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import Backbone, Layout
from .config import Config
from .state import Memory


class Encoder(nn.Module):
    """All of C* dot E*: patch projection, MAE replacement, latent tokens, causal
    backbone, bottleneck, tanh.

    One signature serves window training, frozen episode scanning, recurrent
    execution and diagnostics. The caller supplies and receives the bounded-window
    memory, so a batched scan and frame-by-frame recurrence are identical by
    construction. `p_mask` is zero on every Z* path -- masking is a Phase-1A
    training mechanism, and a cached target produced under a random mask would make
    the same frame yield different latents.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.n_latents = config.n_latents
        self.patch_proj = nn.Linear(config.patch_dim, config.d_model_encoder)
        self.latents = nn.Parameter(torch.randn(config.n_latents, config.d_model_encoder) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(config.d_model_encoder) * 0.02)
        self.backbone = Backbone(
            config,
            Layout.encoder(config),
            "encoder",
            config.d_model_encoder,
            config.n_heads_encoder,
            config.depth_encoder,
        )
        self.bottleneck = nn.Linear(config.d_model_encoder, config.d_bottleneck)

    def forward(
        self,
        patches: Tensor,
        memory: Memory | None = None,
        p_mask: float = 0.0,
        rng: torch.Generator | None = None,
    ) -> tuple[Tensor, Memory, Tensor]:
        b, t = patches.shape[:2]
        tokens = self.patch_proj(patches)
        keep = torch.rand(tokens.shape[:3], generator=rng, device=tokens.device) >= p_mask
        tokens = torch.where(keep[..., None], tokens, self.mask_token)
        latents = self.latents.expand(b, t, -1, -1)

        encoded, memory = self.backbone(torch.cat([latents, tokens], dim=2), memory)
        z = torch.tanh(self.bottleneck(encoded[:, :, : self.n_latents]))
        return z, memory, ~keep


class Decoder(nn.Module):
    """Diagnostic and Phase-1A only. Never in the control path, and for a JEPA arm
    trained only after the encoder is frozen, so reconstruction cannot leak into
    the representation objective."""

    def __init__(self, config: Config):
        super().__init__()
        self.n_latents = config.n_latents
        self.up = nn.Linear(config.d_bottleneck, config.d_model_encoder)
        self.queries = nn.Parameter(torch.randn(config.n_patches, config.d_model_encoder) * 0.02)
        self.backbone = Backbone(
            config,
            Layout.encoder(config),
            "decoder",
            config.d_model_encoder,
            config.n_heads_encoder,
            config.depth_encoder,
        )
        self.head = nn.Linear(config.d_model_encoder, config.patch_dim)

    def forward(self, z: Tensor, memory: Memory | None = None) -> tuple[Tensor, Memory]:
        b, t = z.shape[:2]
        tokens = torch.cat([self.up(z), self.queries.expand(b, t, -1, -1)], dim=2)
        decoded, memory = self.backbone(tokens, memory)
        return torch.sigmoid(self.head(decoded[:, :, self.n_latents :])), memory


class Projector(nn.Module):
    """Where SIGReg acts, keeping Z* itself unconstrained -- LeJEPA regularises a
    projection and probes the backbone embedding. Unused by the MAE arm."""

    def __init__(self, config: Config, width: int = 512):
        super().__init__()
        flat = config.n_latents * config.d_bottleneck
        self.net = nn.Sequential(
            nn.Linear(flat, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Linear(width, width)
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z.flatten(2).flatten(0, 1))


def pack(z: Tensor, config: Config) -> Tensor:
    """(B, T, n_latents, d_bottleneck) -> (B, T, n_spatial, d_spatial), Dreamer 4's
    own reshape: 512 x 16 becomes 256 x 32."""
    b, t = z.shape[:2]
    return z.reshape(b, t, config.n_spatial, config.d_spatial)


def reconstruction_loss(predicted: Tensor, target: Tensor, masked: Tensor) -> Tensor:
    """Scored on replaced patches only, per Dreamer 4's masked autoencoding."""
    error = (predicted - target).pow(2).mean(-1)
    weight = masked.float()
    return (error * weight).sum() / weight.sum().clamp(min=1.0)


def representation_loss(
    online: Encoder, target: Encoder | None, projector: Projector | None, batch, config: Config
) -> Tensor:
    raise NotImplementedError("Stage B: EMA views and SIGReg placement are open")


@torch.no_grad()
def update_target(online: Encoder, target: Encoder, momentum: float) -> None:
    for source, copy in zip(online.parameters(), target.parameters()):
        copy.lerp_(source, 1.0 - momentum)
