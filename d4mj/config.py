from dataclasses import dataclass
from typing import Literal

Transition = Literal["flow", "direct"]
TimeMixer = Literal["attention", "mamba"]


@dataclass(frozen=True)
class Config:
    """Every constant in the system. The four Stage-A arms are four Config values.

    Fields that define Z* -- patch, window, n_latents, d_bottleneck, packing --
    must be frozen before the final encoder trains: changing one changes every
    latent. Capacity fields are shape-legal placeholders until the 6 GB probe.
    """

    transition: Transition = "flow"
    time_mixer: TimeMixer = "attention"

    # Environment. Craftax-Classic renders 9x9 tiles of BLOCK_PIXEL_SIZE_AGENT=7.
    resolution: int = 63
    channels: int = 3
    patch: int = 7
    n_actions: int = 17

    # Z*. window is the bounded causal context: z_t = Z*(x_{t-window+1..t}).
    window: int = 16
    n_latents: int = 32
    d_bottleneck: int = 16
    packing: int = 2
    mae_p_max: float = 0.9
    lpips_weight: float = 0.2

    d_model_encoder: int = 256
    depth_encoder: int = 8
    n_heads_encoder: int = 4

    d_model: int = 256
    depth: int = 8
    n_heads: int = 4
    mlp_ratio: float = 4.0
    time_every: int = 4
    n_register: int = 4
    n_agent: int = 2

    # d_state is the primary parameter-matching knob.
    mamba_d_state: int = 64
    mamba_headdim: int = 64
    mamba_expand: int = 1
    mamba_d_conv: int = 4

    # k_max is the training noise grid; rungs is the generation ladder length.
    k_max: int = 8
    rungs: int = 4
    tau_ctx_noise: float = 0.1

    mtp_leads: int = 8
    bins: int = 255
    symlog_limit: float = 20.0

    horizon: int = 8
    gamma: float = 0.997
    lam: float = 0.95
    pmpo_alpha: float = 0.5
    prior_beta: float = 0.3

    # Dreamer 4 reports C = 3*T_short and T_long = 4*T_short in all three of its
    # Appendix A configurations. A scaling rule it reports, not one it claims.
    sequence: int = 16
    sequence_long: int = 64
    dynamics_context: int = 48
    long_batch_every: int = 4
    separate_image_fraction: float = 0.3
    batch: int = 8
    rms_decay: float = 0.99
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    warmup: int = 1000
    ema_momentum: tuple[float, float] = (0.996, 1.0)

    seed: int = 20260731

    def __post_init__(self) -> None:
        assert self.resolution % self.patch == 0
        assert self.n_latents == self.n_spatial * self.packing
        assert self.depth % self.time_every == 0
        assert self.depth // self.time_every >= 2
        assert self.k_max >= 8 and self.k_max & (self.k_max - 1) == 0
        assert self.rungs <= self.k_max
        assert self.tau_ctx_index < self.k_max
        assert self.dynamics_context == 3 * self.sequence
        assert self.sequence_long == 4 * self.sequence
        assert self.sequence_long > self.dynamics_context

    @property
    def n_patches(self) -> int:
        return (self.resolution // self.patch) ** 2

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch * self.patch

    @property
    def n_spatial(self) -> int:
        return self.n_latents // self.packing

    @property
    def d_spatial(self) -> int:
        return self.d_bottleneck * self.packing

    @property
    def receptive_field(self) -> int:
        """`window` bounds each time layer's state; influence still travels one
        window per time layer, so a latent depends on this many frames."""
        return self.window * (self.depth_encoder // self.time_every)

    @property
    def burn_in(self) -> int:
        return self.receptive_field - 1

    @property
    def n_signal_bins(self) -> int:
        """Exactly k_max: the top row of a k_max+1 table is unreachable by training."""
        return self.k_max

    @property
    def n_step_bins(self) -> int:
        return self.k_max.bit_length()

    @property
    def tau_ctx_index(self) -> int:
        """Signal bin of every committed block, clamped below the untrained top row."""
        return min(round((1.0 - self.tau_ctx_noise) * self.k_max), self.k_max - 1)

    @property
    def tau_ctx_signal(self) -> float:
        """The signal the committed tensor is actually mixed at. Derived from the bin
        so content and label cannot disagree: mixing at 0.9 while labelling bin 7/8
        mislabels every observation, commit and deployment prefix."""
        return self.tau_ctx_index / self.k_max

    @property
    def step_index(self) -> int:
        """Committed and observed blocks carry the finest step size d_min."""
        return self.n_step_bins - 1
