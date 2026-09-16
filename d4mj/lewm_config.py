"""Explicit, versioned settings for the joint LeWM family (legacy Config is untouched)."""

from dataclasses import asdict, dataclass, field, fields
import math


@dataclass(frozen=True)
class EncoderSettings:
    resolution: int = 63
    channels: int = 3
    patch: int = 7
    width: int = 192
    depth: int = 12
    heads: int = 3
    mlp_ratio: int = 4
    latent_dim: int = 192
    projector_hidden: int = 2048
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    qkv_bias: bool = True
    dropout: float = 0.0
    attention_backend: str = "sdpa"
    pixel_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    pixel_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1
    checkpoint_blocks: bool = True


@dataclass(frozen=True)
class DynamicsSettings:
    width: int = 256
    depth: int = 6
    n_actions: int = 17
    action_dim: int = 64
    d_state: int = 64
    headdim: int = 64
    expand: int = 1
    d_conv: int = 4
    chunk_size: int = 256
    norm_eps: float = 1e-5
    backend: str = "triton"
    # Remaining construction arguments are explicit, not installed-package defaults.
    ngroups: int = 1
    A_init_range: tuple[float, float] = (1.0, 16.0)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: tuple[float, str] = (0.0, "inf")
    bias: bool = False
    conv_bias: bool = True
    norm_before_gate: bool = False


@dataclass(frozen=True)
class JointSettings:
    frames: int = 4
    # Native environment steps between retained frames.  1 reproduces the v1
    # recipe exactly; 4 is TC-LeWM's frame skip, which makes a four-frame
    # centering window span 13 native steps instead of 4.
    stride: int = 1
    # Spacing of the SIGReg centering set, independent of the prediction pairs.
    # >1 widens the centering window in physical time while dynamics stay
    # one-step, which is the window-only ablation.  The encoded window is then
    # the union of the prediction frames and the centering frames, so both
    # `centering` settings see identical encoder inputs and identical projector
    # BatchNorm batches: only the index set entering SIGReg differs.
    centering_stride: int = 1
    centering: str = "consecutive"
    batch: int = 128
    sigreg_weight: float = 0.09
    projections: int = 1024
    knots: int = 17
    steps: int = 10000
    screen_step: int = 2000
    learning_rate: float = 5e-5
    min_learning_rate: float = 5e-6
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    grad_clip: float = 1.0
    warmup: int = 500
    checkpoint_every: int = 500


def pair_axis(left: dict, right: dict) -> str:
    """The single declared axis on which a paired run's two arms may differ.

    A pair isolates one variable: the regularizer target (raw vs TC), or the
    centering index set for the window ablation, where both arms are TC and the
    arm directories keep their raw/tc names for tooling.
    """
    def leaves(value, prefix=""):
        flat = {}
        for key, item in value.items():
            if isinstance(item, dict):
                flat.update(leaves(item, prefix + key + "."))
            else:
                flat[prefix + key] = item
        return flat
    a, b = leaves(left), leaves(right)
    axis = {key for key in set(a) | set(b) if a.get(key) != b.get(key)}
    if axis not in ({"variant"}, {"joint.centering"}):
        raise ValueError("a paired run differs in exactly one declared axis: variant or joint.centering")
    return next(iter(axis))


def window_layout(j: JointSettings) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Native offsets of the encoded frames, and positions of the prediction and
    centering sets *within that window* -- not native offsets themselves.

    At ``frames=4, centering_stride=4`` the window holds native offsets
    (0,1,2,3,4,8,12); the strided centering set is native (0,4,8,12), which is
    positions (0,4,5,6) in the window. Callers index tensors, so positions are
    what is returned.

    The encoded window is the union of the two sets, so a `consecutive` and a
    `strided` recipe with the same `centering_stride` encode exactly the same
    frames and differ only in which latents SIGReg centers over.  At the
    defaults this is the four consecutive frames of the v1 recipe.
    """
    base = [k * j.stride for k in range(j.frames)]
    centre = [k * j.centering_stride * j.stride for k in range(j.frames)]
    offsets = sorted(set(base) | set(centre))
    index = {offset: position for position, offset in enumerate(offsets)}
    selected = centre if j.centering == "strided" else base
    return tuple(offsets), tuple(index[o] for o in base), tuple(index[o] for o in selected)


@dataclass(frozen=True)
class RuntimeSettings:
    device: str = "cuda"
    precision: str = "bf16"
    # Tests use the same implementation with small dimensions, never a research run ID.
    purpose: str = "research"
    cache_chunk: int = 128
    cache_dtype: str = "float32"
    memory_budget_bytes: int = 6 * 1024**3


@dataclass(frozen=True)
class LeWMConfig:
    schema: str = "d4mj_lewm_recipe_v1"
    family: str = "lewm_mamba"
    variant: str = "tc"
    seed: int = 20260731
    encoder: EncoderSettings = field(default_factory=EncoderSettings)
    dynamics: DynamicsSettings = field(default_factory=DynamicsSettings)
    joint: JointSettings = field(default_factory=JointSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    def __post_init__(self):
        validate_recipe(self)


@dataclass(frozen=True)
class ScreenConfig:
    """Evaluation-only G1 choices, sealed separately from the model recipe."""

    schema: str = "d4mj_joint_screen_v1"
    seed: int = 20260906
    train_episodes: int = 256
    dev_episodes: int = 128
    windows_per_episode: int = 4
    encode_batch: int = 32
    probe_hidden: int = 128
    probe_steps: int = 200
    probe_batch: int = 256
    probe_lr: float = 1e-3
    probe_decay: float = 1e-2
    minimum_positive: int = 20
    minimum_negative: int = 20
    bootstrap_draws: int = 1000
    auc_margin: float = .03
    variance_floor: float = 1e-10

    def __post_init__(self):
        if self.schema != "d4mj_joint_screen_v1":
            raise ValueError("unsupported joint screen schema")
        for item in fields(self):
            value = getattr(self,item.name)
            if item.type is int and type(value) is not int:
                raise ValueError(f"screen count {item.name} must be an integer")
            if item.type is float and (type(value) not in (int,float) or not math.isfinite(value)):
                raise ValueError(f"screen value {item.name} must be finite numeric")
        for name, value in asdict(self).items():
            if name in ("schema", "seed"):
                continue
            if isinstance(value, float) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid screen field {name}")
            if isinstance(value, int) and (type(value) is not int or value < 1):
                raise ValueError(f"invalid screen count {name}")
        if self.dev_episodes < 2 or self.train_episodes < 2 or not 0 < self.auc_margin < 1:
            raise ValueError("screen needs multiple episodes and a valid retention margin")
        if self.probe_lr <= 0 or self.variance_floor <= 0:
            raise ValueError("screen learning rate and variance floor must be positive")


def validate_recipe(c: LeWMConfig) -> None:
    if c.schema not in ("d4mj_lewm_recipe_v1", "d4mj_lewm_recipe_v2") or c.family != "lewm_mamba":
        raise ValueError("unsupported LeWM recipe schema/family")
    if type(c.joint.stride) is not int or c.joint.stride < 1:
        raise ValueError("joint stride must be a positive integer")
    if c.joint.stride != 1 and c.schema != "d4mj_lewm_recipe_v2":
        raise ValueError("a strided joint window requires recipe schema v2")
    if type(c.joint.centering_stride) is not int or c.joint.centering_stride < 1:
        raise ValueError("centering stride must be a positive integer")
    if c.joint.centering not in ("consecutive", "strided"):
        raise ValueError("centering selects the consecutive or the strided index set")
    if c.joint.centering_stride != 1:
        if c.schema != "d4mj_lewm_recipe_v2":
            raise ValueError("a widened centering window requires recipe schema v2")
        if c.joint.stride != 1:
            raise ValueError("the centering ablation holds dynamics one-step: stride must be 1")
    elif c.joint.centering != "consecutive":
        raise ValueError("a strided centering set needs centering_stride > 1")
    if c.variant not in ("raw", "tc"):
        raise ValueError("regularizer target must be raw or tc")
    e, d, j, r = c.encoder, c.dynamics, c.joint, c.runtime
    for group in (e, d, j, r):
        for name, value in asdict(group).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"nonfinite recipe field {name}")
    for value in (e.resolution, e.patch, e.width, e.depth, e.heads, e.mlp_ratio,
                  e.latent_dim, e.projector_hidden, d.width, d.depth, d.n_actions,
                  d.action_dim, d.d_state, d.headdim, d.expand, d.d_conv, d.chunk_size):
        if type(value) is not int or value <= 0:
            raise ValueError("model dimensions must be positive integers")
    if e.resolution % e.patch or e.width % e.heads or d.width * d.expand % d.headdim:
        raise ValueError("incompatible patch/head geometry")
    if e.channels != 3 or len(e.pixel_mean) != 3 or len(e.pixel_std) != 3 or min(e.pixel_std) <= 0:
        raise ValueError("encoder requires an explicit RGB normalization")
    if e.dropout != 0 or e.attention_backend not in ("sdpa", "eager"):
        raise ValueError("M0-M3 supports the declared dropout-free ViT recipe")
    if min(e.layer_norm_eps, e.bn_eps, d.norm_eps) <= 0 or not 0 < e.bn_momentum <= 1:
        raise ValueError("invalid normalization settings")
    if d.backend not in ("reference", "triton") or d.ngroups != 1:
        raise ValueError("supported recurrence: reference/triton, ngroups=1")
    if tuple(d.dt_limit) != (0.0, "inf") or d.norm_before_gate:
        raise ValueError("this source-audited path requires unbounded dt and gate before norm")
    if not 0 < d.dt_min <= d.dt_max or d.A_init_range[0] <= 0 or d.A_init_range[1] < d.A_init_range[0]:
        raise ValueError("invalid Mamba initialization")
    if j.frames != 4 or j.batch < 2 or j.projections < 1 or j.knots < 2:
        raise ValueError("joint loss needs four frames, B>=2 and valid SIGReg dimensions")
    if j.sigreg_weight <= 0 or not 0 < j.min_learning_rate <= j.learning_rate:
        raise ValueError("invalid joint loss/learning rate")
    if not 0 <= j.warmup < j.steps or not 0 < j.screen_step <= j.steps:
        raise ValueError("invalid fixed training schedule")
    if j.checkpoint_every <= 0 or j.grad_clip <= 0 or j.weight_decay < 0 or j.optimizer_eps <= 0:
        raise ValueError("invalid optimizer/checkpoint settings")
    if len(j.betas) != 2 or not all(0 <= b < 1 for b in j.betas):
        raise ValueError("invalid AdamW betas")
    if r.device not in ("cpu", "cuda") or r.precision not in ("fp32", "bf16"):
        raise ValueError("unsupported device/precision")
    if r.purpose not in ("research", "verification") or r.cache_chunk < 1 or r.cache_dtype != "float32":
        raise ValueError("invalid runtime/cache contract")
    if r.purpose == "research" and (j.batch != 128 or j.projections != 1024 or j.knots != 17):
        raise ValueError("research recipe requires actual B128 / J1024 / 17 knots; no microbatch substitute")








def _settings(cls, values):
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    values = dict(values)
    for f in fields(cls):
        if f.name in values and isinstance(f.default, tuple):
            values[f.name] = tuple(values[f.name])
    return cls(**values)


def config_from_dict(values: dict) -> LeWMConfig:
    values = dict(values)
    for name, cls in (("encoder", EncoderSettings), ("dynamics", DynamicsSettings),
                      ("joint", JointSettings), ("runtime", RuntimeSettings)):
        if name in values:
            values[name] = _settings(cls, values[name])
    return _settings(LeWMConfig, values)
