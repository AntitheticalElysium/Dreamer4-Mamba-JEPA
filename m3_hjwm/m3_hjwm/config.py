from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    # Observation / representation.
    image_size: int = 64
    in_channels: int = 3
    patch_size: int = 8                  # 64x64 -> 8x8 tokens.
    token_dim: int = 128
    num_registers: int = 4
    encoder_depth: int = 4

    # Spatial processing.
    spatial_depth: int = 2
    spatial_heads: int = 4
    spatial_window: int = 4

    # Temporal processing.
    temporal_backend: Literal["auto", "mamba3", "mamba2", "gru"] = "auto"
    temporal_depth: int = 2
    temporal_state_dim: int = 128
    mamba_d_state: int = 64
    mamba_headdim: int = 64
    action_dim: int = 17

    # Future predictor.
    predictor: Literal["deterministic", "mixture"] = "mixture"
    num_modes: int = 4
    predictor_depth: int = 3
    horizon_bins: int = 16

    # Task heads.
    reward_bins: int = 255
    reward_low: float = -20.0
    reward_high: float = 20.0

    # JEPA target.
    ema_decay: float = 0.996
    mask_ratio: float = 0.60
    target_blocks: int = 4

    # Reliability.
    reliability_hidden: int = 128
    value_ensemble: int = 3

    def validate(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        grid = self.image_size // self.patch_size
        if grid % self.spatial_window:
            raise ValueError("token grid must be divisible by spatial_window")
        if self.predictor == "mixture" and self.num_modes < 2:
            raise ValueError("mixture predictor needs at least two modes")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    sequence_length: int = 32
    imagination_horizon: int = 15

    world_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    grad_clip: float = 100.0

    gamma: float = 0.997
    lambda_: float = 0.95
    entropy_coef: float = 3e-4

    # Objective weights. Keep these explicit; do not silently add losses.
    jepa_weight: float = 1.0
    mode_commitment_weight: float = 0.05
    mode_balance_weight: float = 0.01
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    reliability_weight: float = 0.1

    # Reliability stays shadow-only until calibrated.
    reliability_shadow_only: bool = True
    reliability_temperature: float = 1.0

    device: str = "cuda"
