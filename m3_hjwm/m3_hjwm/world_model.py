from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .encoder import DenseVisualEncoder, JEPATargetEncoder, MaskTokenInjector
from .masking import multi_block_mask
from .spatial import SpatialMixer
from .temporal import TemporalModel, TemporalState
from .predictor import DeterministicPredictor, HardModeMixturePredictor, PredictorOutput
from .heads import RewardHead, ContinueHead
from .reliability import (
    TargetManifoldProjector, CompatibilityEnergy, ReliabilityPredictor,
    ReliabilitySignals, mode_dispersion,
)
from .utils import effective_rank


@dataclass
class WorldModelState:
    temporal: TemporalState
    tokens: Tensor                 # [B,S,D], latest emitted state


@dataclass
class WorldModelOutput:
    context_tokens: Tensor         # [B,T,S,D]
    target_tokens: Tensor          # [B,T,S,D]
    predictions: PredictorOutput
    reward_logits: Tensor          # [B,T-1,bins]
    continue_logits: Tensor        # [B,T-1]
    loss: Tensor
    metrics: dict[str, Tensor]


class M3HJWM(nn.Module):
    """Full representation + dynamics model.

    Transition convention:
        state_t, action_t -> state_{t+1}, reward_{t+1}, continuation_{t+1}

    For sequence batches:
        observations: [B,T,C,H,W]
        actions:      [B,T-1]
        rewards:      [B,T-1]   (reward caused by actions[:, t])
        continues:    [B,T-1]
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.encoder = DenseVisualEncoder(cfg.in_channels, cfg.token_dim, cfg.patch_size, cfg.encoder_depth)
        self.target_encoder = JEPATargetEncoder(self.encoder, cfg.ema_decay)
        self.mask_injector = MaskTokenInjector(cfg.token_dim)
        self.spatial = SpatialMixer(cfg.token_dim, cfg.spatial_heads, cfg.spatial_depth, cfg.num_registers)
        self.action_input = nn.Embedding(cfg.action_dim, cfg.token_dim)
        self.temporal = TemporalModel(
            cfg.token_dim, cfg.temporal_depth, cfg.temporal_backend,
            cfg.mamba_d_state, cfg.mamba_headdim,
        )
        if cfg.predictor == "mixture":
            self.predictor = HardModeMixturePredictor(
                cfg.token_dim, cfg.action_dim, cfg.predictor_depth,
                cfg.horizon_bins, cfg.num_modes,
            )
        else:
            self.predictor = DeterministicPredictor(
                cfg.token_dim, cfg.action_dim, cfg.predictor_depth, cfg.horizon_bins
            )
        self.reward = RewardHead(cfg.token_dim, cfg.reward_bins, cfg.reward_low, cfg.reward_high)
        self.continue_head = ContinueHead(cfg.token_dim)
        self.manifold = TargetManifoldProjector(cfg.token_dim)
        self.energy = CompatibilityEnergy(cfg.token_dim, cfg.action_dim)
        self.reliability = ReliabilityPredictor(cfg.reliability_hidden)

    @property
    def streams(self) -> int:
        g = self.cfg.image_size // self.cfg.patch_size
        return g * g + self.cfg.num_registers

    def encode_frame(self, obs: Tensor, target_mask: Tensor | None = None) -> Tensor:
        tokens, _ = self.encoder(obs)
        if target_mask is not None:
            tokens = self.mask_injector(tokens, target_mask)
        return self.spatial(tokens)

    @torch.no_grad()
    def target_frame(self, obs: Tensor) -> Tensor:
        tokens, _ = self.target_encoder(obs)
        return self.spatial(tokens)  # shared spatial mixer remains online/trainable by design.

    def _pool(self, tokens: Tensor) -> Tensor:
        # Register mean; local-token mean fallback when registers=0.
        if self.cfg.num_registers:
            return tokens[..., :self.cfg.num_registers, :].mean(-2)
        return tokens.mean(-2)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> WorldModelState:
        temporal = self.temporal.init_state(batch, self.streams, device, dtype)
        return WorldModelState(temporal, temporal.output)

    def observe_step(
        self, obs: Tensor, prev_action: Tensor, state: WorldModelState, reset: Tensor | None = None
    ) -> WorldModelState:
        tokens = self.encode_frame(obs)
        x = tokens + self.action_input(prev_action)[:, None]
        out, temporal = self.temporal.step(x, state.temporal, reset)
        return WorldModelState(temporal, out)

    def imagine_step(
        self, state: WorldModelState, action: Tensor, deterministic_mode: bool = False
    ) -> tuple[WorldModelState, Tensor, Tensor, PredictorOutput]:
        b = action.shape[0]
        horizon = torch.ones(b, dtype=torch.long, device=action.device)
        if hasattr(self.predictor, "_all_modes"):
            modes, logits = self.predictor._all_modes(state.tokens, action, horizon)  # isolated internal use
            idx = logits.argmax(-1) if deterministic_mode else torch.distributions.Categorical(logits=logits).sample()
            generated = modes[torch.arange(b, device=action.device), idx]
            zero = generated.new_zeros(b)
            pred_out = PredictorOutput(generated, modes, idx, logits, zero, zero.mean(), zero.mean())
        else:
            generated = self.predictor.sample(state.tokens, action, horizon)
            zero = generated.new_zeros(b)
            pred_out = PredictorOutput(generated, None, None, None, zero, zero.mean(), zero.mean())

        # The generated target representation is the next observation-like token state.
        # The action that caused it is consumed exactly once.
        x = generated + self.action_input(action)[:, None]
        out, temporal = self.temporal.step(x, state.temporal)
        next_state = WorldModelState(temporal, out)
        control = self._pool(out)
        reward_logits = self.reward(control)
        continue_logits = self.continue_head(control)
        return next_state, reward_logits, continue_logits, pred_out

    def forward(self, batch: dict[str, Tensor], train_cfg: TrainConfig) -> WorldModelOutput:
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"].float()
        continues = batch["continues"].float()
        resets = batch.get("resets")
        b, t = obs.shape[:2]
        if actions.shape != (b, t - 1):
            raise ValueError(f"actions must be [B,T-1], got {tuple(actions.shape)}")
        if rewards.shape != actions.shape or continues.shape != actions.shape:
            raise ValueError("rewards/continues must align one-to-one with actions")

        grid = self.cfg.image_size // self.cfg.patch_size
        masks = multi_block_mask(
            b * t, grid, grid, self.cfg.mask_ratio, self.cfg.target_blocks,
            obs.device,
        )
        flat_obs = obs.reshape(b * t, *obs.shape[2:])
        ctx_local, _ = self.encoder(flat_obs)
        ctx_local = self.mask_injector(ctx_local, masks)
        ctx = self.spatial(ctx_local).reshape(b, t, self.streams, self.cfg.token_dim)
        with torch.no_grad():
            tgt_local, _ = self.target_encoder(flat_obs)
        target = self.spatial(tgt_local).reshape(b, t, self.streams, self.cfg.token_dim)

        # Input at time t contains previous action a_{t-1}; time 0 gets a null action 0.
        prev_actions = torch.zeros(b, t, dtype=torch.long, device=obs.device)
        prev_actions[:, 1:] = actions
        temporal_in = ctx + self.action_input(prev_actions)[:, :, None, :]
        context, _ = self.temporal.forward_sequence(temporal_in, resets)

        # Predict target at t+1 from context state t and action_t.
        horizon = torch.ones(b * (t - 1), dtype=torch.long, device=obs.device)
        pred = self.predictor(
            context[:, :-1].reshape(-1, self.streams, self.cfg.token_dim),
            actions.reshape(-1),
            target[:, 1:].reshape(-1, self.streams, self.cfg.token_dim),
            horizon,
        )

        # Task heads consume the post-transition state at t+1, never the pre-action state.
        next_control = self._pool(context[:, 1:])
        reward_logits = self.reward(next_control)
        continue_logits = self.continue_head(next_control)
        reward_loss = self.reward.loss(reward_logits, rewards).mean()
        continue_loss = self.continue_head.loss(continue_logits, continues).mean()

        jepa_loss = pred.per_sample_loss.mean()
        manifold_loss = self.manifold.training_loss(target[:, 1:].detach())
        energy_loss = self.energy.contrastive_loss(
            context[:, :-1].reshape(-1, self.streams, self.cfg.token_dim),
            actions.reshape(-1),
            target[:, 1:].reshape(-1, self.streams, self.cfg.token_dim),
        )
        loss = (
            train_cfg.jepa_weight * jepa_loss
            + train_cfg.mode_commitment_weight * pred.commitment_loss
            + train_cfg.mode_balance_weight * pred.balance_loss
            + train_cfg.reward_weight * reward_loss
            + train_cfg.continue_weight * continue_loss
            + 0.05 * manifold_loss
            + 0.05 * energy_loss
        )
        metrics = {
            "loss": loss.detach(),
            "jepa_loss": jepa_loss.detach(),
            "mode_commitment": pred.commitment_loss.detach(),
            "mode_balance": pred.balance_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "continue_loss": continue_loss.detach(),
            "manifold_loss": manifold_loss.detach(),
            "energy_loss": energy_loss.detach(),
            "target_effective_rank": effective_rank(target.detach()),
        }
        return WorldModelOutput(context, target, pred, reward_logits, continue_logits, loss, metrics)

    @torch.no_grad()
    def update_target(self) -> None:
        self.target_encoder.update(self.encoder)
