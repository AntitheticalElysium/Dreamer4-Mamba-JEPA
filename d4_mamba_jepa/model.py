"""D4-lite model assembly around the unchanged MMBench2 implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .config import D4LiteConfig
from .source import load_mmbench2_model
from .temporal import replace_dynamics_time_attention


class DiscreteActionEncoder(nn.Module):
    """Crafter action encoder with the upstream one-token interface.

    Index ``-1`` is the dedicated start/unlabelled action. Real Crafter action
    IDs ``0..n_actions-1`` are shifted by one. A shared base token plus
    small-initialized action deltas mirrors the upstream continuous
    ``ActionEncoder`` initialization.
    """

    def __init__(self, d_model: int, n_actions: int):
        super().__init__()
        self.d_model = int(d_model)
        self.n_actions = int(n_actions)
        self.base = nn.Parameter(torch.empty(self.d_model))
        self.action_delta = nn.Embedding(self.n_actions + 1, self.d_model)
        nn.init.normal_(self.base, std=0.02)
        nn.init.normal_(self.action_delta.weight, std=1e-3)

    def forward(
        self,
        actions: Optional[Tensor],
        *,
        batch_time_shape: Optional[tuple[int, int]] = None,
        act_mask: Optional[Tensor] = None,
    ) -> Tensor:
        del act_mask  # Categorical actions have no per-dimension validity mask.
        if actions is None:
            if batch_time_shape is None:
                raise ValueError("batch_time_shape is required when actions=None")
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
            return out[:, :, None, :]

        if actions.ndim == 3 and actions.shape[-1] == 1:
            actions = actions[..., 0]
        if actions.ndim != 2:
            raise ValueError("discrete actions must have shape [B,T] or [B,T,1]")
        ids = actions.to(torch.long)
        if bool((ids < -1).any()) or bool((ids >= self.n_actions).any()):
            raise ValueError(
                f"actions must lie in [-1,{self.n_actions - 1}]"
            )
        out = self.base.view(1, 1, -1) + self.action_delta(ids + 1)
        return out[:, :, None, :]


class ContinuationHeadMTP(nn.Module):
    """Multi-token continuation logits from post-transition agent tokens."""

    def __init__(self, d_model: int, horizon: int):
        super().__init__()
        upstream = load_mmbench2_model()
        self.horizon = int(horizon)
        self.projector = upstream.MLP(d_model=d_model, mlp_ratio=2.0, dropout=0.0)
        self.out = nn.Linear(d_model, self.horizon)
        # Neutral p(continue)=0.5 initialization; calibration is measured rather
        # than encoded as a sparse-terminal prior.
        nn.init.zeros_(self.out.bias)

    def forward(self, agent_tokens: Tensor) -> Tensor:
        if agent_tokens.ndim == 4:
            pooled = agent_tokens.mean(dim=2)
        elif agent_tokens.ndim == 3:
            pooled = agent_tokens
        else:
            raise ValueError("agent_tokens must have shape [B,T,N,D] or [B,T,D]")
        return self.out(self.projector(pooled))


class CDPPredictor(nn.Module):
    """Dreamer-CDP-shaped predictor from causal state and next action."""

    def __init__(
        self,
        *,
        d_model: int,
        n_spatial: int,
        d_spatial: int,
        hidden_ratio: float,
    ):
        super().__init__()
        hidden = max(d_model, int(d_model * hidden_ratio))
        self.n_spatial = int(n_spatial)
        self.d_spatial = int(d_spatial)
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_spatial * self.d_spatial),
        )

    def forward(self, agent_tokens: Tensor, next_action_tokens: Tensor) -> Tensor:
        if agent_tokens.ndim != 4:
            raise ValueError("agent_tokens must have shape [B,T,N,D]")
        if next_action_tokens.ndim != 3:
            raise ValueError("next_action_tokens must have shape [B,T,D]")
        context = agent_tokens.mean(dim=2)
        if context.shape[:2] != next_action_tokens.shape[:2]:
            raise ValueError("context and next-action time axes differ")
        prediction = self.net(torch.cat([context, next_action_tokens], dim=-1))
        return prediction.view(
            *prediction.shape[:2], self.n_spatial, self.d_spatial
        )


def build_tokenizer(cfg: D4LiteConfig, *, training_mask: bool = True) -> nn.Module:
    """Construct the upstream tokenizer at the registered D4-lite scale."""
    upstream = load_mmbench2_model()
    mae_min = cfg.mae_p_min if training_mask else 0.0
    mae_max = cfg.mae_p_max if training_mask else 0.0
    encoder = upstream.Encoder(
        patch_dim=cfg.patch_dim,
        d_model=cfg.tokenizer_d_model,
        n_latents=cfg.n_latents,
        n_patches=cfg.n_patches,
        n_heads=cfg.tokenizer_heads,
        depth=cfg.tokenizer_depth,
        d_bottleneck=cfg.d_bottleneck,
        dropout=0.0,
        mlp_ratio=cfg.tokenizer_mlp_ratio,
        time_every=cfg.tokenizer_time_every,
        latents_only_time=True,
        mae_p_min=mae_min,
        mae_p_max=mae_max,
    )
    decoder = upstream.Decoder(
        d_bottleneck=cfg.d_bottleneck,
        d_model=cfg.tokenizer_d_model,
        n_heads=cfg.tokenizer_heads,
        depth=cfg.tokenizer_depth,
        n_latents=cfg.n_latents,
        n_patches=cfg.n_patches,
        d_patch=cfg.patch_dim,
        dropout=0.0,
        mlp_ratio=cfg.tokenizer_mlp_ratio,
        time_every=cfg.tokenizer_time_every,
        latents_only_time=True,
    )
    return upstream.Tokenizer(encoder, decoder)


@dataclass(frozen=True)
class EncodedSequence:
    bottleneck: Tensor  # [B,T,L,d_bottleneck]
    packed: Tensor  # [B,T,n_spatial,d_spatial]


class D4LiteWorld(nn.Module):
    """Frozen-tokenizer D4-style dynamics and task heads.

    The initial implementation supports the unchanged Transformer baseline.
    Mamba and CDP are installed only through their separately tested modules.
    """

    def __init__(self, cfg: D4LiteConfig):
        super().__init__()
        self.cfg = cfg
        upstream = load_mmbench2_model()
        tokenizer = build_tokenizer(cfg, training_mask=False)
        self.encoder = tokenizer.encoder
        self.decoder = tokenizer.decoder

        self.dynamics = upstream.Dynamics(
            d_model=cfg.dynamics_d_model,
            d_bottleneck=cfg.d_bottleneck,
            d_spatial=cfg.d_spatial,
            n_spatial=cfg.n_spatial,
            n_register=cfg.n_register,
            n_agent=cfg.n_agent,
            n_heads=cfg.dynamics_heads,
            depth=cfg.dynamics_depth,
            k_max=cfg.k_max,
            dropout=0.0,
            mlp_ratio=cfg.dynamics_mlp_ratio,
            time_every=cfg.dynamics_time_every,
            lang_dim=0,
        )
        self.dynamics.action_encoder = DiscreteActionEncoder(
            cfg.dynamics_d_model, cfg.n_actions
        )

        self.reward_head = upstream.RewardHeadMTP(
            d_model=cfg.dynamics_d_model,
            L=cfg.reward_horizon,
            num_bins=cfg.reward_bins,
            mlp_ratio=2.0,
            dropout=0.0,
            log_low=cfg.reward_log_low,
            log_high=cfg.reward_log_high,
            pool_agent="attn",
        )
        self.continuation_head = ContinuationHeadMTP(
            cfg.dynamics_d_model, cfg.continuation_horizon
        )

        if cfg.temporal_backend == "mamba2":
            replace_dynamics_time_attention(self.dynamics, cfg)
        if cfg.representation_objective == "cdp":
            self.cdp_predictor = CDPPredictor(
                d_model=cfg.dynamics_d_model,
                n_spatial=cfg.n_spatial,
                d_spatial=cfg.d_spatial,
                hidden_ratio=cfg.cdp_hidden_ratio,
            )
            # The pretrained decoder is an invariant reconstruction anchor. It
            # propagates loss gradients to encoder inputs but is not updated.
            self.decoder.eval()
            for parameter in self.decoder.parameters():
                parameter.requires_grad_(False)
        else:
            self.cdp_predictor = None
            self.freeze_tokenizer()

    def freeze_tokenizer(self) -> None:
        self.encoder.eval()
        self.decoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

    def encode_frames(self, frames: Tensor, *, frozen: bool = True) -> EncodedSequence:
        """Encode ``[B,T,C,H,W]`` frames with upstream patchification."""
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B,T,C,H,W]")
        if frames.shape[-2:] != (self.cfg.image_size, self.cfg.image_size):
            raise ValueError(
                f"expected {self.cfg.image_size}x{self.cfg.image_size} frames"
            )
        upstream = load_mmbench2_model()
        pixels = frames.float()
        if frames.dtype == torch.uint8:
            pixels = pixels / 255.0
        patches = upstream.temporal_patchify(pixels, self.cfg.patch_size)
        context = torch.no_grad() if frozen else torch.enable_grad()
        with context:
            bottleneck, _ = self.encoder(patches)
        packed = upstream.pack_bottleneck_to_spatial(
            bottleneck, n_spatial=self.cfg.n_spatial, k=self.cfg.packing_factor
        )
        return EncodedSequence(bottleneck=bottleneck, packed=packed)

    def forward_dynamics(
        self,
        packed: Tensor,
        led_to_actions: Tensor,
        step_indices: Tensor,
        signal_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.dynamics(
            led_to_actions,
            step_indices,
            signal_indices,
            packed,
            act_mask=None,
            agent_tokens=None,
            lang_emb=None,
        )

    def forward_task_heads(self, agent_tokens: Tensor) -> dict[str, Tensor]:
        reward_logits, reward_centers = self.reward_head(agent_tokens)
        continue_logits = self.continuation_head(agent_tokens)
        return {
            "reward_logits": reward_logits,
            "reward_centers": reward_centers,
            "continue_logits": continue_logits,
        }

    def predict_cdp(self, clean: Tensor, led_to_actions: Tensor) -> Tensor:
        """Predict ``z[t+1]`` from clean causal state ``t`` and action ``t``."""
        if self.cdp_predictor is None:
            raise RuntimeError("CDP predictor is disabled in this arm")
        if clean.shape[1] < 2:
            raise ValueError("CDP requires at least two timesteps")
        B, T = clean.shape[:2]
        steps = torch.full(
            (B, T),
            self.cfg.max_step_index,
            device=clean.device,
            dtype=torch.long,
        )
        signals = torch.full(
            (B, T),
            self.cfg.k_max,
            device=clean.device,
            dtype=torch.long,
        )
        _, agent_tokens = self.forward_dynamics(
            clean, led_to_actions, steps, signals
        )
        # led_to_actions[t+1] is action_t: the action to apply after context t.
        next_action_tokens = self.dynamics.action_encoder(
            led_to_actions[:, 1:],
            batch_time_shape=(B, T - 1),
            act_mask=None,
        )[:, :, 0]
        return self.cdp_predictor(
            agent_tokens[:, :-1], next_action_tokens
        )
