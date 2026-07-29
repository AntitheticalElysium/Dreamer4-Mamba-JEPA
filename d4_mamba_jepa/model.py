"""D4-lite model assembly around the unchanged MMBench2 implementation."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .config import D4LiteConfig
from .source import load_lejepa_sigreg, load_mmbench2_model
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
        context_mode: str = "pooled_agent",
        n_agent: int = 2,
    ):
        super().__init__()
        self.n_spatial = int(n_spatial)
        self.d_spatial = int(d_spatial)
        self.context_mode = str(context_mode)
        self.n_agent = int(n_agent)
        # How much of the post-dynamics state reaches the predictor:
        #   pooled_agent  = mean over agent tokens            -> d_model
        #   concat_agent  = all agent tokens                  -> n_agent*d_model
        #   spatial_agent = spatial stream + agent tokens     -> +n_spatial*d_spatial
        # `pooled_agent` is the default and reproduces the original module
        # exactly; Dreamer-CDP by contrast predicts from a 4096-d deterministic
        # state (`rssm.py:140`), so the pooled 64-d channel is a local narrowing.
        if self.context_mode == "pooled_agent":
            context_dim = d_model
        elif self.context_mode == "concat_agent":
            context_dim = self.n_agent * d_model
        elif self.context_mode == "spatial_agent":
            context_dim = self.n_agent * d_model + self.n_spatial * self.d_spatial
        else:
            raise ValueError(f"unsupported context_mode={context_mode!r}")
        hidden = max(d_model, int(d_model * hidden_ratio))
        if self.context_mode != "pooled_agent":
            # A wider context through an unchanged 64-unit hidden layer would
            # simply move the bottleneck; scale the hidden width with it.
            hidden = max(hidden, context_dim)
        self.context_dim = context_dim
        self.net = nn.Sequential(
            nn.Linear(context_dim + d_model, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_spatial * self.d_spatial),
        )

    def forward(
        self,
        agent_tokens: Tensor,
        next_action_tokens: Tensor,
        spatial_tokens: Tensor | None = None,
    ) -> Tensor:
        if agent_tokens.ndim != 4:
            raise ValueError("agent_tokens must have shape [B,T,N,D]")
        if next_action_tokens.ndim != 3:
            raise ValueError("next_action_tokens must have shape [B,T,D]")
        if self.context_mode == "pooled_agent":
            context = agent_tokens.mean(dim=2)
        elif self.context_mode == "concat_agent":
            context = agent_tokens.flatten(2)
        else:
            if spatial_tokens is None:
                raise ValueError(
                    "context_mode='spatial_agent' requires spatial_tokens"
                )
            if spatial_tokens.ndim != 4:
                raise ValueError("spatial_tokens must have shape [B,T,S,D]")
            context = torch.cat(
                [spatial_tokens.flatten(2), agent_tokens.flatten(2)], dim=-1
            )
        if context.shape[:2] != next_action_tokens.shape[:2]:
            raise ValueError("context and next-action time axes differ")
        prediction = self.net(torch.cat([context, next_action_tokens], dim=-1))
        return prediction.view(
            *prediction.shape[:2], self.n_spatial, self.d_spatial
        )


class JepaProjector(nn.Module):
    """SPR/BYOL global projection or prediction MLP.

    Faithful to ``mila-iqia/spr`` commit ``0b9dd4e7`` ``src/models.py``
    ``global_classifier`` (``Linear -> BatchNorm1d -> ReLU -> Linear``). The
    batch-norm is the documented SPR/BYOL anti-collapse component; it is applied
    over the folded ``[B*T]`` batch, matching SPR's flattened application.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("projector input must be [B,T,F]")
        B, T, F = x.shape
        y = self.net(x.reshape(B * T, F))
        return y.reshape(B, T, -1)


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

        # NOTE: the Mamba-2 substitution is deliberately deferred to the END of
        # construction. It used to run here, before the JEPA predictor and the
        # three projection MLPs were built, so it consumed RNG and every
        # `mamba2` world drew DIFFERENT weights for those four shared modules
        # than the `transformer` world at the same seed. That silently broke the
        # "temporal operator is the single moved axis" contract of D037: 16
        # shared tensors differed at initialization. Building the backend last
        # makes every non-temporal module bit-identical across backends.
        # Attributes present in every arm for state-dict/introspection symmetry.
        self.cdp_predictor = None
        self.jepa_predictor = None
        self.target_encoder = None
        self.jepa_projection = None
        self.jepa_prediction = None
        self.jepa_target_projection = None
        self.sigreg_test = None
        self.jepa_sigreg_projector = None
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
        elif cfg.representation_objective == "jepa":
            self._build_jepa()
        else:
            self.freeze_tokenizer()

        # Backend substitution LAST, so it cannot perturb the initialization of
        # any module shared with the transformer arm (see the note above).
        if cfg.temporal_backend == "mamba2":
            replace_dynamics_time_attention(self.dynamics, cfg)

    def _build_jepa(self) -> None:
        """Non-generative SPR/BYOL arm: drop the decoder, keep the online
        encoder trainable, add a stop-gradient EMA target encoder, a
        deterministic action-conditioned next-embedding predictor (the rollout),
        and asymmetric projection/prediction heads."""
        cfg = self.cfg
        self.decoder = None  # non-generative: no pixel decoder
        # The deterministic action-conditioned predictor (the rollout) is shared
        # by both anti-collapse mechanisms.
        self.jepa_predictor = CDPPredictor(
            d_model=cfg.dynamics_d_model,
            n_spatial=cfg.n_spatial,
            d_spatial=cfg.d_spatial,
            hidden_ratio=cfg.jepa_predictor_hidden_ratio,
            context_mode=cfg.jepa_predictor_context,
            n_agent=cfg.n_agent,
        )
        flat = cfg.n_spatial * cfg.d_spatial
        if cfg.jepa_anticollapse == "ema":
            # SPR/BYOL: stop-grad EMA target encoder + asymmetric heads.
            self.target_encoder = copy.deepcopy(self.encoder)
            # The EMA target must stay in TRAINING mode during JEPA optimization
            # so its BatchNorm normalizes with current-batch statistics, exactly
            # as pinned SPR does (``do_spr_loss`` uses ``no_grad``, never
            # ``eval()``). Forcing ``eval()`` here -- or in any diagnostic that
            # then fails to restore modes -- makes the target use stale running
            # statistics and silently collapses the representation while the
            # training cosine looks high (a causally-validated bug). The
            # stop-gradient is enforced ONLY by ``requires_grad_(False)`` below;
            # the module mode is left at its default (train). See
            # tests/test_jepa_arm.py::test_jepa_target_stays_in_training_mode.
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.jepa_projection = JepaProjector(flat, cfg.jepa_projection_dim)
            self.jepa_prediction = JepaProjector(
                cfg.jepa_projection_dim, cfg.jepa_projection_dim
            )
            self.jepa_target_projection = copy.deepcopy(self.jepa_projection)
            # As above (see the target encoder): freeze by requires_grad only
            # and keep BatchNorm in training mode so targets use current-batch
            # statistics. This projection is the one whose BatchNorm the bug
            # actually corrupted.
            for parameter in self.jepa_target_projection.parameters():
                parameter.requires_grad_(False)
        else:
            # LeJEPA: no EMA target and no asymmetric prediction heuristics.
            # Both sides use the non-EMA online encoder with gradients; SIGReg
            # on its projected embeddings supplies the anti-collapse pressure.
            SlicingUnivariateTest, EppsPulley = load_lejepa_sigreg()
            self.sigreg_test = SlicingUnivariateTest(
                univariate_test=EppsPulley(n_points=cfg.jepa_sigreg_points),
                num_slices=cfg.jepa_sigreg_slices,
                reduction="mean",
            )
            # LeJEPA keeps a projector (config projector_arch="MLP"). SIGReg and
            # the prediction/invariance loss it regularizes act in this
            # throwaway projected space, so the raw downstream encoder latent is
            # free to keep task structure. Applied per spatial token.
            proj = cfg.jepa_projection_dim
            self.jepa_sigreg_projector = nn.Sequential(
                nn.Linear(cfg.d_spatial, 2 * proj),
                nn.GELU(),
                nn.Linear(2 * proj, proj),
            )
        # Online encoder is trained jointly by self-prediction.
        self.encoder.train()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)

    def freeze_tokenizer(self) -> None:
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        if self.decoder is not None:
            self.decoder.eval()
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

    # --- Non-generative JEPA arm -------------------------------------------
    def predict_next_jepa(self, clean: Tensor, led_to_actions: Tensor) -> Tensor:
        """Deterministic z[t+1] from clean causal state t and action_t.

        No denoising: one dynamics pass over clean latents, then the
        action-conditioned predictor. Returns [B, T-1, n_spatial, d_spatial].
        """
        if self.jepa_predictor is None:
            raise RuntimeError("JEPA predictor disabled in this arm")
        if clean.shape[1] < 2:
            raise ValueError("JEPA prediction requires at least two timesteps")
        B, T = clean.shape[:2]
        steps = torch.full(
            (B, T), self.cfg.max_step_index, device=clean.device, dtype=torch.long
        )
        signals = torch.full(
            (B, T), self.cfg.k_max, device=clean.device, dtype=torch.long
        )
        spatial_tokens, agent_tokens = self.forward_dynamics(
            clean, led_to_actions, steps, signals
        )
        next_action_tokens = self.dynamics.action_encoder(
            led_to_actions[:, 1:], batch_time_shape=(B, T - 1), act_mask=None
        )[:, :, 0]
        return self.jepa_predictor(
            agent_tokens[:, :-1], next_action_tokens, spatial_tokens[:, :-1]
        )

    def encode_frames_target(self, frames: Tensor) -> Tensor:
        """Encode frames with the stop-gradient EMA target encoder -> packed."""
        if self.target_encoder is None:
            raise RuntimeError("JEPA target encoder disabled in this arm")
        upstream = load_mmbench2_model()
        pixels = frames.float()
        if frames.dtype == torch.uint8:
            pixels = pixels / 255.0
        patches = upstream.temporal_patchify(pixels, self.cfg.patch_size)
        with torch.no_grad():
            bottleneck, _ = self.target_encoder(patches)
        return upstream.pack_bottleneck_to_spatial(
            bottleneck, n_spatial=self.cfg.n_spatial, k=self.cfg.packing_factor
        )

    def jepa_online_project(self, latent: Tensor) -> Tensor:
        """Online path: projection then prediction head (BYOL predictor)."""
        B, T = latent.shape[:2]
        flat = latent.reshape(B, T, -1)
        return self.jepa_prediction(self.jepa_projection(flat))

    def jepa_target_project(self, latent: Tensor) -> Tensor:
        """Target path: EMA projection only, stop-gradient."""
        B, T = latent.shape[:2]
        flat = latent.reshape(B, T, -1)
        with torch.no_grad():
            return self.jepa_target_projection(flat)

    @torch.no_grad()
    def update_jepa_target(self, tau: float) -> None:
        """EMA update of the target encoder and target projection.

        ``target = tau*target + (1-tau)*online`` with the momentum ``tau``
        ramped toward 1 over training, following the I-JEPA (``52c1ae95``) and
        V-JEPA-2 (``204698b4``) target-encoder momentum schedule cited for the
        target-encoder mechanics. This is the same EMA as SPR's
        ``update_state_dict`` (rlpyt) up to the reciprocal tau naming. Buffers
        (BatchNorm running statistics) are copied directly from the online net.
        """
        if self.target_encoder is None:
            raise RuntimeError("JEPA target encoder disabled in this arm")
        pairs = (
            (self.target_encoder, self.encoder),
            (self.jepa_target_projection, self.jepa_projection),
        )
        for target, online in pairs:
            for tp, sp in zip(target.parameters(), online.parameters()):
                tp.mul_(tau).add_(sp.detach(), alpha=1.0 - tau)
            for tb, sb in zip(target.buffers(), online.buffers()):
                tb.copy_(sb)
