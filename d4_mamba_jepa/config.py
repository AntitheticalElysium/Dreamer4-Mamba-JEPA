"""Explicit configuration axes for the four-arm factorial."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class D4LiteConfig:
    # Environment / data.
    image_size: int = 64
    channels: int = 3
    patch_size: int = 8
    sequence_length: int = 16
    n_actions: int = 17

    # Causal tokenizer. It remains Transformer-based in all initial arms.
    tokenizer_d_model: int = 64
    tokenizer_heads: int = 4
    tokenizer_depth: int = 4
    tokenizer_time_every: int = 2
    tokenizer_mlp_ratio: float = 4.0
    n_latents: int = 16
    d_bottleneck: int = 16
    mae_p_min: float = 0.0
    mae_p_max: float = 0.9

    # Interactive dynamics.
    dynamics_d_model: int = 64
    dynamics_heads: int = 4
    dynamics_depth: int = 4
    dynamics_time_every: int = 2
    dynamics_mlp_ratio: float = 4.0
    packing_factor: int = 4
    n_register: int = 2
    n_agent: int = 2
    k_max: int = 4

    # Task heads.
    reward_horizon: int = 8
    reward_bins: int = 255
    reward_log_low: float = -10.0
    reward_log_high: float = 10.0
    continuation_horizon: int = 8

    # Independently switchable research factors.
    temporal_backend: str = "transformer"  # transformer | mamba2
    representation_objective: str = "base"  # base | cdp

    # Official Mamba-2 constructor parameters.
    mamba_d_state: int = 16
    mamba_headdim: int = 32
    mamba_d_conv: int = 4
    # expand=1 gives 15,014 parameters at d_model=64 versus 16,644 for the
    # upstream temporal attention module. expand=2 would confound the backend
    # comparison with a 67% larger temporal module.
    mamba_expand: int = 1

    # CDP-shaped auxiliary.
    cdp_hidden_ratio: float = 1.0
    cdp_weight: float = 1.0
    reconstruction_anchor_weight: float = 1.0
    encoder_lr_ratio: float = 0.3

    def __post_init__(self) -> None:
        if self.temporal_backend not in {"transformer", "mamba2"}:
            raise ValueError(f"unsupported temporal_backend={self.temporal_backend!r}")
        if self.representation_objective not in {"base", "cdp"}:
            raise ValueError(
                f"unsupported representation_objective="
                f"{self.representation_objective!r}"
            )
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.n_latents % self.packing_factor:
            raise ValueError("n_latents must be divisible by packing_factor")
        if self.dynamics_d_model % self.dynamics_heads:
            raise ValueError("dynamics_d_model must be divisible by dynamics_heads")
        if self.tokenizer_d_model % self.tokenizer_heads:
            raise ValueError("tokenizer_d_model must be divisible by tokenizer_heads")
        if self.dynamics_d_model % self.mamba_headdim:
            raise ValueError("dynamics_d_model must be divisible by mamba_headdim")
        if self.k_max < 1 or self.k_max & (self.k_max - 1):
            raise ValueError("k_max must be a positive power of two")
        if not 0.0 <= self.mae_p_min <= self.mae_p_max <= 1.0:
            raise ValueError("MAE probabilities must satisfy 0 <= min <= max <= 1")
        if not 0.0 < self.encoder_lr_ratio <= 1.0:
            raise ValueError("encoder_lr_ratio must be in (0, 1]")
        if self.cdp_weight < 0.0 or self.reconstruction_anchor_weight < 0.0:
            raise ValueError("CDP and reconstruction weights must be non-negative")

    @property
    def n_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch_size * self.patch_size

    @property
    def n_spatial(self) -> int:
        return self.n_latents // self.packing_factor

    @property
    def d_spatial(self) -> int:
        return self.d_bottleneck * self.packing_factor

    @property
    def max_step_index(self) -> int:
        return int(math.log2(self.k_max))

    @property
    def arm_id(self) -> str:
        temporal = "T" if self.temporal_backend == "transformer" else "M"
        objective = "BASE" if self.representation_objective == "base" else "CDP"
        return f"{temporal}-{objective}"

    def to_dict(self) -> dict:
        return asdict(self)
