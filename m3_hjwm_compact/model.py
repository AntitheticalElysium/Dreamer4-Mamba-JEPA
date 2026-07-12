"""Compact M3-HJWM world model.

Final-state contract
--------------------
obs_t --online JEPA encoder--> dense tokens x_t
history/actions --temporal backend--> context c_t
(c_t, action_t) --future predictor--> target-like tokens y_hat_{t+1}
(y_hat_{t+1}, action_t, cache_t) --temporal step--> c_{t+1}
c_{t+1} --task heads--> reward_{t+1}, continuation_{t+1}

The full representation encoder (visual stem + spatial mixer + registers) has an
EMA target copy. No target-side trainable layer is accidentally left outside EMA.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import copy, math, warnings
from typing import Literal, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration and small utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 64
    in_channels: int = 3
    patch_size: int = 8                   # 64x64 -> 8x8 local tokens
    token_dim: int = 64                   # 6 GB-safe default
    registers: int = 2
    spatial_depth: int = 1
    spatial_heads: int = 4

    # Backends are selected explicitly so a failed Mamba import can never turn a
    # named experiment into a GRU experiment. GRU is the portable default.
    temporal_backend: Literal["auto", "mamba3", "mamba2", "gru"] = "gru"
    temporal_depth: int = 1
    mamba_d_state: int = 32
    mamba_headdim: int = 16

    action_dim: int = 17
    # Mixtures remain an experimental control until the mode-validity gates pass.
    predictor: Literal["deterministic", "mixture"] = "deterministic"
    predictor_depth: int = 2
    modes: int = 4
    mode_balance_temperature: float = 0.1
    horizon_bins: int = 32

    reward_bins: int = 255
    reward_low: float = -20.0
    reward_high: float = 20.0
    ema_decay: float = 0.996
    # The masked temporal hybrid remains an explicit ablation. It is not the
    # default: replacing post-convolution tokens is not an I-JEPA context
    # encoder, and it creates a real-token/generation mismatch at imagination.
    mask_ratio: float = 0.0
    target_blocks: int = 4
    # The anti-collapse auxiliary operates on L2-normalized directions scaled by
    # sqrt(dim), so an isotropic spherical representation has unit per-dimension
    # variance without forcing the raw control-state magnitude to grow.
    variance_target: float = 1.0
    # Number of autoregressive transitions in the optional V-JEPA-2-AC-shaped
    # bridge loss. The loss weight lives in LossConfig and defaults to zero until
    # the corrected representation and multi-seed gates pass.
    rollout_steps: int = 2
    reliability_hidden: int = 64

    def validate(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.token_dim % self.spatial_heads:
            raise ValueError("token_dim must be divisible by spatial_heads")
        if self.predictor == "mixture" and self.modes < 2:
            raise ValueError("mixture predictor needs >=2 modes")
        if self.mode_balance_temperature <= 0:
            raise ValueError("mode_balance_temperature must be positive")
        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1); 0 disables masking")
        if self.rollout_steps < 2:
            raise ValueError("rollout_steps must be at least 2")
        if self.variance_target <= 0:
            raise ValueError("variance_target must be positive")


@dataclass(frozen=True)
class LossConfig:
    jepa: float = 1.0
    mode_router: float = 0.05
    mode_balance: float = 0.01
    reward: float = 1.0
    continuation: float = 1.0
    # VICReg-inspired anti-collapse regularization on online tokens. Unlike the
    # original image-level VICReg objective, dense streams are separate variables:
    # statistics are taken across observations at each fixed stream, never across
    # token positions. The 25:1 variance:covariance ratio follows VICReg after a
    # common rescaling; its absolute scale remains an empirical hyperparameter.
    variance: float = 1.0
    covariance: float = 0.04
    # Optional two-step autoregressive bridge, adapted from V-JEPA 2-AC Eq. 3-4.
    # It is deliberately opt-in pending corrected, multi-seed Phase B/D evidence.
    rollout: float = 0.0
    # Reliability auxiliaries are opt-in and must not shape the world model while
    # they are uncalibrated shadow signals.
    manifold: float = 0.0
    energy: float = 0.0


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def two_hot(x: Tensor, bins: int, low: float, high: float) -> Tensor:
    x = symlog(x).clamp(low, high)
    p = (x - low) / (high - low) * (bins - 1)
    lo, hi = p.floor().long(), p.ceil().long()
    out = torch.zeros(*x.shape, bins, device=x.device, dtype=x.dtype)
    out.scatter_add_(-1, lo[..., None], (hi.float() - p + (hi == lo))[..., None])
    out.scatter_add_(-1, hi[..., None], (p - lo.float())[..., None])
    return out


def decode_two_hot(logits: Tensor, low: float, high: float) -> Tensor:
    support = torch.linspace(low, high, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return symexp((logits.softmax(-1) * support).sum(-1))


def cosine_distance(a: Tensor, b: Tensor) -> Tensor:
    return 1.0 - (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(-1)


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    source_params = dict(source.named_parameters())
    for name, p in target.named_parameters():
        p.mul_(decay).add_(source_params[name], alpha=1.0 - decay)
    source_buffers = dict(source.named_buffers())
    for name, b in target.named_buffers():
        if name in source_buffers:
            b.copy_(source_buffers[name])


def variance_covariance_losses(
    x: Tensor, gamma: float = 1.0, eps: float = 1e-4
) -> tuple[Tensor, Tensor]:
    """Streamwise VICReg-inspired variance and covariance terms.

    For ``[..., streams, dim]`` inputs, leading axes are observations and each
    stream is regularized independently. Flattening streams into the sample axis
    is invalid for dense representations: a fixed position codebook would then
    satisfy both terms while ignoring every observation. A two-dimensional
    ``[samples, dim]`` tensor is treated as one stream.

    Computed in float32 because near-collapsed bf16 variances can underflow.
    """
    if x.ndim < 2:
        raise ValueError("anti-collapse input must have a feature dimension")
    if x.ndim == 2:
        streams = x.float()[:, None, :]
    else:
        streams = x.reshape(-1, x.shape[-2], x.shape[-1]).float()
    samples = streams.shape[0]
    if samples < 2:
        raise ValueError("anti-collapse statistics require at least two observations")
    std = (streams.var(0) + eps).sqrt()
    variance = F.relu(gamma - std).mean()
    centered = streams - streams.mean(0, keepdim=True)
    cov = torch.einsum("nsd,nse->sde", centered, centered) / (samples - 1)
    off_diagonal = cov - torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
    covariance = off_diagonal.pow(2).sum(dim=(-2, -1)).mean() / streams.shape[-1]
    return variance, covariance


def effective_rank(x: Tensor, eps: float = 1e-8) -> Tensor:
    x = x.reshape(-1, x.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    p = singular / (singular.sum() + eps)
    return torch.exp(-(p * (p + eps).log()).sum())


def multi_block_mask(
    batch: int, grid: int, ratio: float, blocks: int, device: torch.device
) -> Tensor:
    """Boolean target mask [B,grid,grid]; context receives its complement."""
    target = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    if ratio == 0.0:
        return target
    wanted = max(1, round(grid * grid * ratio))
    for b in range(batch):
        for _ in range(blocks * 8):
            if int(target[b].sum()) >= wanted:
                break
            bh = int(torch.randint(max(1, grid // 4), max(2, 3 * grid // 4 + 1), (1,), device=device))
            bw = int(torch.randint(max(1, grid // 4), max(2, 3 * grid // 4 + 1), (1,), device=device))
            y = int(torch.randint(0, grid - bh + 1, (1,), device=device))
            x = int(torch.randint(0, grid - bw + 1, (1,), device=device))
            target[b, y:y + bh, x:x + bw] = True
        if int(target[b].sum()) < wanted:
            idx = torch.randperm(grid * grid, device=device)[:wanted]
            target[b].flatten()[idx] = True
    return target


# ---------------------------------------------------------------------------
# Dense JEPA representation encoder
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim), nn.SiLU(), nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(1, dim), nn.SiLU(), nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class SpatialBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        y = self.n1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        return x + self.mlp(self.n2(x))


class RepresentationEncoder(nn.Module):
    """Visual stem + registers + spatial mixer.

    The entire module is duplicated for the EMA target, avoiding a moving target
    through shared trainable spatial layers.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        stages = int(math.log2(cfg.patch_size))
        channels = [min(cfg.token_dim, 32 * 2**i) for i in range(stages)]
        layers, c = [], cfg.in_channels
        for out in channels:
            layers += [
                nn.Conv2d(c, out, 4, stride=2, padding=1),
                nn.GroupNorm(1, out), nn.SiLU(), ConvBlock(out),
            ]
            c = out
        self.stem = nn.Sequential(*layers)
        self.project = nn.Conv2d(c, cfg.token_dim, 1)
        self.registers = nn.Parameter(torch.randn(1, cfg.registers, cfg.token_dim) * 0.02)
        self.spatial = nn.ModuleList(
            [SpatialBlock(cfg.token_dim, cfg.spatial_heads) for _ in range(cfg.spatial_depth)]
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.token_dim) * 0.02)
        self.register_count = cfg.registers

    def forward(
        self,
        obs: Tensor,
        target_mask: Tensor | None = None,
        visible_index: Tensor | None = None,
    ) -> Tensor:
        """Encode a frame.

        `visible_index` [B, K] selects local tokens BEFORE the spatial mixer and
        drops the rest (I-JEPA context encoding: the mixer attends over visible
        tokens + registers only). Mutually exclusive with `target_mask`, which
        substitutes a mask token (legacy hybrid path). Known deviation from ViT
        I-JEPA: the conv stem's receptive field leaks some masked-patch pixels
        into visible tokens; see reviews/2026-07-13-step1-protocol.md.
        """
        if target_mask is not None and visible_index is not None:
            raise ValueError("target_mask and visible_index are mutually exclusive")
        x = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs.float()
        local = self.project(self.stem(x)).flatten(2).transpose(1, 2)
        if target_mask is not None:
            mask = target_mask.flatten(1)[..., None]
            local = torch.where(mask, self.mask_token.to(local.dtype), local)
        if visible_index is not None:
            local = local.gather(
                1, visible_index[..., None].expand(-1, -1, local.shape[-1])
            )
        return self.mix(local)

    def mix(self, local: Tensor) -> Tensor:
        """Prepend registers and run the spatial mixer over given local tokens."""
        regs = self.registers.expand(local.shape[0], -1, -1)
        tokens = torch.cat([regs, local], dim=1)
        for block in self.spatial:
            tokens = block(tokens)
        return tokens


class EMARepresentationEncoder(nn.Module):
    def __init__(self, online: RepresentationEncoder, decay: float):
        super().__init__()
        self.model = copy.deepcopy(online)
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, online: RepresentationEncoder) -> None:
        ema_update(self.model, online, self.decay)

    @torch.no_grad()
    def forward(self, obs: Tensor) -> Tensor:
        return self.model(obs)

    def train(self, mode: bool = True):
        # Parent-module .train() recursively visits children. Keep the teacher in
        # inference mode even if stochastic/normalizing layers are added later.
        super().train(False)
        self.model.eval()
        return self


# ---------------------------------------------------------------------------
# Temporal backend
# ---------------------------------------------------------------------------

@dataclass
class TemporalState:
    cache: Any
    output: Tensor


class GRUTemporal(nn.Module):
    """CPU-safe reference backend; not the research claim."""
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.cells = nn.ModuleList([nn.GRUCell(dim, dim) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.dim, self.depth = dim, depth

    def init_state(self, batch: int, streams: int, device, dtype) -> TemporalState:
        h = [torch.zeros(batch * streams, self.dim, device=device, dtype=dtype) for _ in range(self.depth)]
        return TemporalState(h, torch.zeros(batch, streams, self.dim, device=device, dtype=dtype))

    def step(self, x: Tensor, state: TemporalState, reset: Tensor | None = None):
        b, s, d = x.shape
        y = x.reshape(b * s, d)
        old = list(state.cache)
        if reset is not None:
            keep = (~reset.bool())[:, None].expand(b, s).reshape(-1, 1).to(y.dtype)
            old = [h * keep for h in old]
        new = []
        for cell, norm, h in zip(self.cells, self.norms, old):
            h = cell(y, h)
            y = y + norm(h)
            new.append(h)
        out = y.reshape(b, s, d)
        return out, TemporalState(new, out)

    def sequence(self, x: Tensor, resets: Tensor | None = None):
        b, t, s, _ = x.shape
        state = self.init_state(b, s, x.device, x.dtype)
        ys = []
        for i in range(t):
            y, state = self.step(x[:, i], state, None if resets is None else resets[:, i])
            ys.append(y)
        return torch.stack(ys, 1), state


class MambaSequenceAdapter(nn.Module):
    """Official mamba_ssm sequence adapter.

    Recurrent cache APIs have changed across Mamba-3 revisions and may depend on
    Triton/CUDA hardware. The agent must pin the installed commit and implement
    `step()` from that exact official code, then pass equivalence tests.
    """
    def __init__(self, cfg: ModelConfig, version: str):
        super().__init__()
        self.version = version
        try:
            if version == "mamba3":
                from mamba_ssm.modules.mamba3 import Mamba3
                cls = Mamba3
                kwargs = dict(
                    d_model=cfg.token_dim,
                    d_state=cfg.mamba_d_state,
                    headdim=cfg.mamba_headdim,
                )
            else:
                from mamba_ssm.modules.mamba2 import Mamba2
                cls = Mamba2
                kwargs = dict(
                    d_model=cfg.token_dim,
                    d_state=cfg.mamba_d_state,
                    headdim=cfg.mamba_headdim,
                    use_mem_eff_path=False,
                )
            self.layers = nn.ModuleList([cls(**kwargs) for _ in range(cfg.temporal_depth)])
            self.norms = nn.ModuleList([nn.LayerNorm(cfg.token_dim) for _ in self.layers])
        except Exception as exc:
            raise RuntimeError(f"cannot construct official {version}: {exc}") from exc

    def sequence(self, x: Tensor, resets: Tensor | None = None):
        if resets is not None and bool(resets[:, 1:].any()):
            raise NotImplementedError("segment sequences at episode reset boundaries")
        b, t, s, d = x.shape
        y = x.permute(0, 2, 1, 3).reshape(b * s, t, d)
        for layer, norm in zip(self.layers, self.norms):
            y = y + layer(norm(y))
        y = y.reshape(b, s, t, d).permute(0, 2, 1, 3)
        return y, TemporalState(None, y[:, -1])

    def init_state(self, batch: int, streams: int, device, dtype):
        flat_batch = batch * streams
        caches = [
            tuple(
                tensor
                for tensor in layer.allocate_inference_cache(
                    flat_batch, max_seqlen=1, device=device, dtype=dtype
                )
            )
            for layer in self.layers
        ]
        output = torch.zeros(batch, streams, self.layers[0].d_model, device=device, dtype=dtype)
        return TemporalState(caches, output)

    @staticmethod
    def _reset_cache(cache: tuple[Tensor, ...], reset_rows: Tensor) -> None:
        if not bool(reset_rows.any()):
            return
        for tensor in cache:
            tensor[reset_rows] = 0

    def step(self, x: Tensor, state: TemporalState, reset: Tensor | None = None):
        if state.cache is None:
            raise RuntimeError("Mamba recurrent step requires an allocated official cache")
        b, s, d = x.shape
        y = x.reshape(b * s, d)
        if reset is not None:
            reset_rows = reset.bool()[:, None].expand(b, s).reshape(-1)
            for cache in state.cache:
                self._reset_cache(cache, reset_rows)

        next_caches = []
        for layer, norm, cache in zip(self.layers, self.norms, state.cache):
            residual = y
            normalized = norm(y)
            if self.version == "mamba3":
                update, *next_cache = layer.step(normalized, *cache)
            else:
                update, *next_cache = layer.step(normalized[:, None], *cache)
                update = update[:, 0]
            y = residual + update
            next_caches.append(tuple(next_cache))
        output = y.reshape(b, s, d)
        return output, TemporalState(next_caches, output)


class TemporalModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.temporal_backend == "auto":
            raise ValueError(
                "temporal backend must be explicit; benchmark and select mamba2, "
                "mamba3, or gru without silent substitution"
            )
        wanted = cfg.temporal_backend
        if wanted in ("mamba3", "mamba2"):
            self.impl = MambaSequenceAdapter(cfg, wanted)
            self.name = wanted
        else:
            self.impl = GRUTemporal(cfg.token_dim, cfg.temporal_depth)
            self.name = "gru"

    def sequence(self, x, resets=None):
        return self.impl.sequence(x, resets)

    def init_state(self, *args, **kwargs):
        return self.impl.init_state(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.impl.step(*args, **kwargs)


# ---------------------------------------------------------------------------
# Future predictor, task heads, and reliability
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    selected: Tensor
    all_modes: Tensor | None
    assignment: Tensor | None
    router_logits: Tensor | None
    regression: Tensor
    router_loss: Tensor
    balance_loss: Tensor


class FuturePredictor(nn.Module):
    """Deterministic or hard-assigned multi-predictor JEPA future model.

    This is a small adaptation of pinned JEPA predictor structure, not a source
    reproduction. I-JEPA uses a ViT predictor with fixed sinusoidal positions;
    V-JEPA-2-AC interleaves action/proprioception tokens over time and uses causal
    attention. Here, learned positions and action/horizon tokens enter a two-block
    single-step spatial predictor. The source-backed invariant is cross-token
    attention: Crafter view shifts move content BETWEEN token positions, which a
    per-token map cannot express.

    The mixture is an experimental backend motivated by MoP-JEPA. Note the
    paper mixes over a POOLED latent with well-separated modes; measured Crafter
    branch dispersion (0.0075 pooled cosine) is far below its separation regime,
    so the deterministic predictor is the Crafter default.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        grid = cfg.image_size // cfg.patch_size
        self.streams = cfg.registers + grid * grid
        self.action = nn.Embedding(cfg.action_dim, cfg.token_dim)
        self.horizon = nn.Embedding(cfg.horizon_bins, cfg.token_dim)
        self.modes = cfg.modes if cfg.predictor == "mixture" else 1
        self.mode_embed = nn.Parameter(torch.randn(self.modes, cfg.token_dim) * 0.02)
        # The encoder is convolutional with no explicit positions; the predictor
        # needs them to express directional movement (I-JEPA: predictor_pos_embed).
        self.pos_embed = nn.Parameter(torch.randn(1, self.streams, cfg.token_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [SpatialBlock(cfg.token_dim, cfg.spatial_heads) for _ in range(cfg.predictor_depth)]
        )
        self.norm = nn.LayerNorm(cfg.token_dim)
        self.out = nn.Linear(cfg.token_dim, cfg.token_dim)
        self.router = nn.Sequential(
            nn.LayerNorm(3 * cfg.token_dim),
            nn.Linear(3 * cfg.token_dim, self.modes),
        )

    def all_predictions(self, context: Tensor, action: Tensor, horizon: Tensor):
        b, s, d = context.shape
        x = context
        if s == self.streams:
            x = x + self.pos_embed
        # else: pooled/synthetic harness contexts (single stream); positions are
        # meaningless there and the conditioning tokens still carry action/horizon.
        x = x[:, None].expand(b, self.modes, s, d) + self.mode_embed[None, :, None]
        x = x.reshape(b * self.modes, s, d)
        # V-JEPA-2-AC-inspired (not identical): conditioning enters as tokens in
        # the attention sequence, not as a broadcast add over every token.
        conditioning = torch.stack([self.action(action), self.horizon(horizon)], dim=1)
        conditioning = conditioning[:, None].expand(b, self.modes, 2, d).reshape(b * self.modes, 2, d)
        x = torch.cat([conditioning, x], dim=1)
        for block in self.blocks:
            x = block(x)
        modes = self.out(self.norm(x[:, 2:])).reshape(b, self.modes, s, d)
        # MoP-JEPA defines context as (state, action). Omitting the action makes
        # deployment routing incapable of representing action-dependent branches.
        route_context = torch.cat(
            [context.mean(1), self.action(action), self.horizon(horizon)], dim=-1
        )
        logits = self.router(route_context)
        return modes, logits

    def forward(self, context: Tensor, action: Tensor, horizon: Tensor, target: Tensor | None = None):
        modes, logits = self.all_predictions(context, action, horizon)
        b = context.shape[0]
        if target is None:
            idx = logits.argmax(-1) if self.modes == 1 else torch.distributions.Categorical(logits=logits).sample()
            selected = modes[torch.arange(b, device=context.device), idx]
            zero = context.new_zeros(())
            return Prediction(selected, modes if self.modes > 1 else None, idx, logits, zero, zero, zero)

        dist = cosine_distance(modes, target[:, None]).mean(-1)  # [B,K]
        assignment = dist.argmin(-1)
        selected = modes[torch.arange(b, device=context.device), assignment]
        regression = dist.gather(1, assignment[:, None]).mean()
        router_loss = F.cross_entropy(logits, assignment.detach()) if self.modes > 1 else regression.new_zeros(())
        if self.modes > 1:
            # The paper's KL over hard argmin assignments is nondifferentiable and
            # cannot itself prevent dead heads. This explicitly labelled soft
            # surrogate has a gradient to the predictor heads; hard usage remains
            # a diagnostic in the verification harness.
            soft_assignment = torch.softmax(
                -dist / self.cfg.mode_balance_temperature, dim=-1
            )
            usage = soft_assignment.mean(0).clamp_min(1e-8)
            balance = (usage * (usage.log() + math.log(self.modes))).sum()
        else:
            balance = regression.new_zeros(())
        return Prediction(selected, modes if self.modes > 1 else None, assignment, logits, regression, router_loss, balance)


class RewardHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.token_dim), nn.Linear(cfg.token_dim, 2 * cfg.token_dim),
            nn.SiLU(), nn.Linear(2 * cfg.token_dim, cfg.reward_bins),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def loss(self, logits: Tensor, reward: Tensor):
        target = two_hot(reward, self.cfg.reward_bins, self.cfg.reward_low, self.cfg.reward_high)
        return -(target * logits.log_softmax(-1)).sum(-1)

    def decode(self, logits: Tensor):
        return decode_two_hot(logits, self.cfg.reward_low, self.cfg.reward_high)


class ContinueHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class ReliabilitySystem(nn.Module):
    """Shadow-first hallucination estimator.

    Signals are not assumed valid. Train `predictor` against actual held-out
    multi-step latent error, then measure calibration before applying weights.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.token_dim
        self.projector = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.energy_action = nn.Embedding(cfg.action_dim, d)
        self.energy = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))
        self.predictor = nn.Sequential(
            nn.Linear(4, cfg.reliability_hidden), nn.SiLU(),
            nn.Linear(cfg.reliability_hidden, cfg.reliability_hidden), nn.SiLU(),
            nn.Linear(cfg.reliability_hidden, 1),
        )

    def compatibility(self, context: Tensor, action: Tensor, future: Tensor):
        x = torch.cat([context.mean(1), self.energy_action(action), future.mean(1)], -1)
        return self.energy(x).squeeze(-1)

    def auxiliary_losses(self, context: Tensor, action: Tensor, real_future: Tensor):
        noisy = real_future + 0.05 * torch.randn_like(real_future)
        manifold = F.mse_loss(self.projector(noisy), real_future.detach())
        pos = self.compatibility(context, action, real_future)
        neg = self.compatibility(context, action, real_future.roll(1, 0))
        energy = F.softplus(pos).mean() + F.softplus(-neg).mean()
        return manifold, energy

    def signals(
        self, context: Tensor, action: Tensor, future: Tensor,
        all_modes: Tensor | None, value_disagreement: Tensor,
    ):
        if all_modes is None:
            dispersion = future.new_zeros(future.shape[0])
        else:
            dispersion = ((all_modes - all_modes.mean(1, keepdim=True)) ** 2).mean((1, 2, 3))
        energy = self.compatibility(context, action, future)
        manifold = ((future - self.projector(future)) ** 2).mean((1, 2))
        return torch.stack([dispersion, energy, manifold, value_disagreement], -1)

    def predicted_error(self, signals: Tensor):
        return F.softplus(self.predictor(signals).squeeze(-1))


# ---------------------------------------------------------------------------
# Complete world model
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    temporal: TemporalState
    tokens: Tensor
    revision: int


@dataclass
class WorldOutput:
    loss: Tensor
    metrics: dict[str, Tensor]
    context: Tensor
    targets: Tensor
    prediction: Prediction
    reward_logits: Tensor
    continue_logits: Tensor


class M3HJWM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self._revision = 0
        self.online_encoder = RepresentationEncoder(cfg)
        self.target_encoder = EMARepresentationEncoder(self.online_encoder, cfg.ema_decay)
        # The final index is a true begin-of-sequence action, distinct from
        # Crafter's valid action 0.
        self.action_input = nn.Embedding(cfg.action_dim + 1, cfg.token_dim)
        self.temporal = TemporalModel(cfg)
        self.future = FuturePredictor(cfg)
        self.reward = RewardHead(cfg)
        self.continuation = ContinueHead(cfg.token_dim)
        self.reliability = ReliabilitySystem(cfg)

    @property
    def streams(self):
        g = self.cfg.image_size // self.cfg.patch_size
        return self.cfg.registers + g * g

    def pool(self, tokens: Tensor):
        return tokens[..., :self.cfg.registers, :].mean(-2) if self.cfg.registers else tokens.mean(-2)

    def initial_state(self, batch: int, device, dtype=torch.float32):
        temporal = self.temporal.init_state(batch, self.streams, device, dtype)
        return WorldState(temporal, temporal.output, self._revision)

    def _assert_fresh(self, state: WorldState) -> None:
        if state.revision != self._revision:
            raise RuntimeError(
                "stale recurrent world state: rebuild it from a real replay prefix "
                "after every world-model update"
            )

    def _previous_action_indices(self, previous_action: Tensor) -> Tensor:
        bos = torch.full_like(previous_action, self.cfg.action_dim)
        return torch.where(previous_action < 0, bos, previous_action)

    def observe_step(self, obs: Tensor, previous_action: Tensor, state: WorldState, reset: Tensor | None = None):
        self._assert_fresh(state)
        tokens = self.online_encoder(obs)
        previous_action = self._previous_action_indices(previous_action)
        x = tokens + self.action_input(previous_action)[:, None]
        out, temporal = self.temporal.step(x, state.temporal, reset)
        return WorldState(temporal, out, self._revision)

    def imagine_step(self, state: WorldState, action: Tensor, deterministic_mode: bool = False):
        self._assert_fresh(state)
        b = action.shape[0]
        horizon = torch.ones(b, dtype=torch.long, device=action.device)
        modes, logits = self.future.all_predictions(state.tokens, action, horizon)
        idx = logits.argmax(-1) if deterministic_mode else torch.distributions.Categorical(logits=logits).sample()
        generated = modes[torch.arange(b, device=action.device), idx]
        pred = Prediction(
            generated, modes if self.future.modes > 1 else None, idx, logits,
            generated.new_zeros(()), generated.new_zeros(()), generated.new_zeros(()),
        )
        x = generated + self.action_input(action)[:, None]
        next_tokens, temporal = self.temporal.step(x, state.temporal)
        next_state = WorldState(temporal, next_tokens, self._revision)
        control = self.pool(next_tokens)
        return next_state, self.reward(control), self.continuation(control), pred

    def _rollout_bridge_loss(
        self,
        context_repr: Tensor,
        previous_action: Tensor,
        actions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Final-state autoregressive loss adapted from V-JEPA 2-AC Eq. 3-4.

        A real prefix initializes the temporal model. Generated target-like tokens
        are then fed back through the exact deployment composition for
        ``cfg.rollout_steps`` transitions. As in V-JEPA 2-AC post-training, the
        visual representation is treated as fixed for this auxiliary: gradients
        cross predictor -> temporal -> predictor, but do not use the bridge to make
        the encoder itself easier to predict.

        The compact model uses cosine distance rather than V-JEPA 2-AC's L1 loss,
        and its temporal core is separate from the spatial predictor. These are
        explicit adaptations, not claims of source equivalence.
        """
        b, t = context_repr.shape[:2]
        steps = self.cfg.rollout_steps
        if t <= steps:
            raise ValueError(
                f"rollout_steps={steps} requires more than {steps} observations"
            )
        prefix = t - steps
        # Stop the auxiliary at the visual representation, while retaining
        # gradients for the action embedding and temporal/predictor parameters.
        inputs = (
            context_repr[:, :prefix].detach()
            + self.action_input(previous_action[:, :prefix])[:, :, None]
        )
        horizon = torch.ones(b, dtype=torch.long, device=context_repr.device)
        prediction = None
        for index in range(steps):
            context, _ = self.temporal.sequence(inputs)
            action = actions[:, prefix - 1 + index]
            modes, logits = self.future.all_predictions(context[:, -1], action, horizon)
            # Deterministic is the validated/default predictor. For an experimental
            # mixture this follows deployment's router choice; the argmax itself is
            # intentionally non-differentiable and does not replace router training.
            assignment = logits.argmax(-1)
            prediction = modes[
                torch.arange(b, device=context_repr.device), assignment
            ]
            if index < steps - 1:
                generated_input = prediction + self.action_input(action)[:, None]
                inputs = torch.cat([inputs, generated_input[:, None]], dim=1)
        final_target = targets[:, prefix + steps - 1]
        return cosine_distance(prediction, final_target).mean()

    def forward(self, batch: dict[str, Tensor], weights: LossConfig = LossConfig()):
        obs, actions = batch["obs"], batch["actions"].long()
        rewards, continues = batch["rewards"].float(), batch["continues"].float()
        resets = batch.get("resets")
        b, t = obs.shape[:2]
        expected = (b, t - 1)
        if actions.shape != expected or rewards.shape != expected or continues.shape != expected:
            raise ValueError("actions/rewards/continues must all be [B,T-1]")

        flat = obs.reshape(b * t, *obs.shape[2:])
        grid = self.cfg.image_size // self.cfg.patch_size
        # mask_ratio == 0 selects the unmasked (Dreamer-CDP-shaped) objective so
        # masked-vs-unmasked stays a controlled comparison, not a hard-coded choice.
        if self.cfg.mask_ratio > 0:
            mask = multi_block_mask(b * t, grid, self.cfg.mask_ratio, self.cfg.target_blocks, obs.device)
        else:
            mask = None
        context_repr = self.online_encoder(flat, mask).reshape(b, t, self.streams, self.cfg.token_dim)
        with torch.no_grad():
            targets = self.target_encoder(flat).reshape(b, t, self.streams, self.cfg.token_dim)

        previous_action = batch.get("previous_actions")
        if previous_action is None:
            previous_action = torch.full(
                (b, t), -1, dtype=torch.long, device=obs.device
            )
            previous_action[:, 1:] = actions
        elif previous_action.shape != (b, t):
            raise ValueError("previous_actions must be [B,T] when supplied")
        previous_action = self._previous_action_indices(previous_action.long())
        temporal_input = context_repr + self.action_input(previous_action)[:, :, None]
        context, _ = self.temporal.sequence(temporal_input, resets)

        horizon = torch.ones(b * (t - 1), dtype=torch.long, device=obs.device)
        flat_context = context[:, :-1].reshape(-1, self.streams, self.cfg.token_dim)
        flat_target = targets[:, 1:].reshape(-1, self.streams, self.cfg.token_dim)
        flat_action = actions.reshape(-1)
        prediction = self.future(flat_context, flat_action, horizon, flat_target)

        # Post-transition heads: action_t aligns with context_{t+1}.
        post = self.pool(context[:, 1:])
        reward_logits = self.reward(post)
        continue_logits = self.continuation(post)
        reward_loss = self.reward.loss(reward_logits, rewards).mean()
        continue_loss = F.binary_cross_entropy_with_logits(continue_logits, continues)
        if weights.manifold != 0.0 or weights.energy != 0.0:
            manifold_loss, energy_loss = self.reliability.auxiliary_losses(
                flat_context, flat_action, flat_target
            )
        else:
            manifold_loss = flat_context.new_zeros(())
            energy_loss = flat_context.new_zeros(())

        if weights.variance != 0.0 or weights.covariance != 0.0:
            # Random masks are themselves a source of per-stream variance. Never
            # let the anti-collapse term be satisfied by mask randomness: use a
            # dense online pass for this auxiliary when the prediction path is
            # masked. The unmasked default reuses the existing representation.
            regularization_repr = context_repr
            if mask is not None:
                regularization_repr = self.online_encoder(flat).reshape(
                    b, t, self.streams, self.cfg.token_dim
                )
            # The predictive loss is cosine-based. Normalize away token norms so
            # the variance hinge cannot be satisfied by magnitude changes along a
            # collapsed direction; sqrt(D) preserves VICReg's gamma=1 scale.
            regularization_repr = F.normalize(
                regularization_repr.float(), dim=-1
            ) * math.sqrt(self.cfg.token_dim)
            variance_loss, covariance_loss = variance_covariance_losses(
                regularization_repr, gamma=self.cfg.variance_target
            )
        else:
            variance_loss = flat_context.new_zeros(())
            covariance_loss = flat_context.new_zeros(())

        if weights.rollout != 0.0:
            rollout_loss = self._rollout_bridge_loss(
                context_repr, previous_action, actions, targets
            )
        else:
            rollout_loss = flat_context.new_zeros(())

        loss = (
            weights.jepa * prediction.regression
            + weights.mode_router * prediction.router_loss
            + weights.mode_balance * prediction.balance_loss
            + weights.reward * reward_loss
            + weights.continuation * continue_loss
            + weights.variance * variance_loss
            + weights.covariance * covariance_loss
            + weights.rollout * rollout_loss
            + weights.manifold * manifold_loss
            + weights.energy * energy_loss
        )
        target_samples = targets.detach().reshape(-1, self.streams, self.cfg.token_dim).float()
        target_stream_mean = target_samples.mean(0, keepdim=True)
        target_observation_variance = (target_samples - target_stream_mean).pow(2).mean()
        target_position_variance = (
            target_stream_mean - target_stream_mean.mean(1, keepdim=True)
        ).pow(2).mean()
        patch_targets = target_samples[:, self.cfg.registers :]
        pooled_targets = (
            patch_targets.mean(1) if patch_targets.shape[1] else target_samples.mean(1)
        )
        metrics = {
            "loss": loss.detach(),
            "jepa": prediction.regression.detach(),
            "router": prediction.router_loss.detach(),
            "balance": prediction.balance_loss.detach(),
            "reward": reward_loss.detach(),
            "continuation": continue_loss.detach(),
            "variance": variance_loss.detach(),
            "covariance": covariance_loss.detach(),
            "rollout": rollout_loss.detach(),
            "manifold": manifold_loss.detach(),
            "energy": energy_loss.detach(),
            # Keep the historical key for artifact compatibility, but make its
            # flattened nature explicit and pair it with observation-sensitive
            # diagnostics. Flat rank alone is not an anti-collapse certificate.
            "target_effective_rank": effective_rank(targets.detach()),
            "target_flat_effective_rank": effective_rank(targets.detach()),
            "target_pooled_effective_rank": effective_rank(pooled_targets),
            "target_fixed_stream_variance": target_observation_variance,
            "target_position_variance": target_position_variance,
            "target_observation_variance_fraction": target_observation_variance
            / (target_observation_variance + target_position_variance).clamp_min(1e-12),
        }
        return WorldOutput(
            loss,
            metrics,
            context,
            targets,
            prediction,
            reward_logits,
            continue_logits,
        )

    @torch.no_grad()
    def update_target(self):
        self.target_encoder.update(self.online_encoder)

    def mark_parameters_updated(self):
        self._revision += 1
