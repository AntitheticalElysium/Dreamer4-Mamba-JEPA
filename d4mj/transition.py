import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import AGENT, REGISTER, SPATIAL, Backbone, Layout
from .config import Config
from .representation import Encoder, pack
from .state import Memory, RealState, WorldState


class World(nn.Module):
    """The action-conditioned dynamics. `forward` evaluates one block; `predict`
    turns its world features into a latent, and is the only thing the two arms do
    differently.

    Flow reads its clean-latent estimate straight off the block it corrupted.
    Direct reads the *next* latent from the current block's features plus the
    action about to be taken -- a separate head, as in V-JEPA 2-AC and DINO-WM,
    rather than an in-block query. A query token carries no information, so a
    position holding one cannot also serve as context for later positions; filling
    every position with a query, which is the only way to train that in one pass,
    leaves the prediction a function of the action history alone.

    The head reads spatial and register features only. Those are agent-free by
    induction over depth under the dynamics mask, so the task cannot reach world
    prediction -- pooling the agent slot instead is what the firewall forbids.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layout = Layout.dynamics(config)
        self.spatial = self.layout.span(SPATIAL)
        self.register = self.layout.span(REGISTER)
        self.agent = self.layout.span(AGENT)

        self.action_embed = nn.Embedding(config.n_actions + 1, config.d_model)
        self.agent_tokens = nn.Parameter(torch.randn(config.n_agent, config.d_model) * 0.02)
        self.project = nn.Linear(config.d_spatial, config.d_model)
        self.registers = nn.Parameter(torch.randn(config.n_register, config.d_model) * 0.02)
        self.backbone = Backbone(
            config,
            self.layout,
            "dynamics",
            config.d_model,
            config.n_heads,
            config.depth,
            config.dynamics_context,
        )

        if config.transition == "flow":
            self.signal_embed = nn.Embedding(config.n_signal_bins, config.d_model // 2)
            self.step_embed = nn.Embedding(config.n_step_bins, config.d_model // 2)
            self.readout = nn.Linear(config.d_model, config.d_spatial)
        else:
            self.condition_embed = nn.Embedding(1, config.d_model)
            width = config.n_spatial + config.n_register
            self.readout = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.SiLU(),
                nn.Linear(config.d_model, config.d_spatial),
            )
            self.pool = nn.Linear(width, config.n_spatial)

    def condition(self, indices: Tensor) -> Tensor:
        if self.config.transition == "flow":
            halves = (self.signal_embed(indices[..., 0]), self.step_embed(indices[..., 1]))
            return torch.cat(halves, dim=-1)[:, :, None]
        return self.condition_embed(torch.zeros_like(indices[..., 0]))[:, :, None]

    def forward(
        self,
        memory: Memory | None,
        led_to_action: Tensor,
        latent: Tensor,
        conditioning: Tensor,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor, Memory]:
        b, t = led_to_action.shape
        blocks = torch.cat(
            [
                self.action_embed(led_to_action)[:, :, None],
                self.condition(conditioning),
                self.project(latent),
                self.registers.expand(b, t, -1, -1),
                self.agent_tokens.expand(b, t, -1, -1),
            ],
            dim=2,
        )
        out, memory = self.backbone(blocks, memory, offset)
        return out, out[:, :, self.agent], memory

    def predict(self, features: Tensor, action: Tensor | None = None) -> Tensor:
        """Flow: the clean latent of *this* block. Direct: the latent of the next
        block, given the action about to be taken. Direct's output is tanh-bounded
        here, so training and rollout share one codomain -- squashing at rollout
        only would decay a correct 0.900 to 0.431 over six recursive steps."""
        if self.config.transition == "flow":
            return self.readout(features[:, :, self.spatial])
        world = torch.cat([features[:, :, self.spatial], features[:, :, self.register]], dim=2)
        pooled = self.pool(world.transpose(2, 3)).transpose(2, 3)
        context = self.action_embed(action)[:, :, None].expand_as(pooled)
        return torch.tanh(self.readout(torch.cat([pooled, context], dim=-1)))


def flow_conditioning(rng: torch.Generator, shape: tuple[int, int], config: Config, device) -> Tensor:
    """Diffusion forcing: an independent step size and signal level per position.

    The signal grid tops out at 1 - d, so tau = 1 is never trained and no path may
    ever present a fully clean latent to the flow arm.
    """
    step = torch.randint(config.n_step_bins, shape, generator=rng, device=device)
    rungs = 2**step
    index = (torch.rand(shape, generator=rng, device=device) * rungs).floor().long()
    return torch.stack([index * (config.k_max // rungs), step], dim=-1)


def signal_level(conditioning: Tensor, config: Config) -> Tensor:
    return conditioning[..., 0].float() / config.k_max


def initial(
    world: World,
    latent: Tensor,
    led_to_action: Tensor,
    rng: torch.Generator,
    config: Config,
    memory: Memory | None = None,
    offset: int = 0,
) -> tuple[WorldState, Tensor]:
    """Commit a known latent and return the state it produces.

    A rollout state cannot be advanced until its starting observation has been
    committed: the direct arm predicts from the committed block's features, and
    `advance` reads memory rather than latent, so starting from an uncommitted
    state silently predicts from nothing.
    """
    committed, conditioning = commit_inputs(latent, rng, config)
    features, agent, memory = world(memory, led_to_action, committed, conditioning, offset)
    return WorldState(latent, memory, offset + latent.shape[1], features), agent


def observe(
    world: World,
    encoder: Encoder,
    state: RealState | None,
    led_to_action: Tensor,
    patches: Tensor,
    rng: torch.Generator,
    config: Config,
) -> tuple[RealState, Tensor]:
    """The one path a real frame takes to become (e_t, z_t, m_t, h_t).

    The encoder's bounded-window memory and the dynamics memory are different
    objects with different lifetimes; carrying them in one field is how a rollout
    silently starts from zero memory.
    """
    encoder_memory = state.encoder_memory if state is not None else None
    world_memory = state.world.memory if state is not None else None
    step = state.world.step if state is not None else 0

    z, encoder_memory, _ = encoder(patches, encoder_memory, offset=step)
    latent = pack(z, config)
    world_state, agent = initial(world, latent, led_to_action, rng, config, world_memory, step)
    return RealState(encoder_memory, world_state), agent


def advance(
    world: World, state: WorldState, action: Tensor, rng: torch.Generator, config: Config
) -> tuple[WorldState, Tensor]:
    """One semantic transition. Flow runs its four read-only rungs then commits;
    direct predicts from the committed block's features and commits once. A
    candidate's memory is always discarded, so the prefix only ever ingests a
    latent the model will condition on later.

    The state must span exactly one block. `initial` over a window returns one
    spanning T, and passing that here would broadcast a single action across every
    block in the direct arm while raising a shape error in flow -- one type meaning
    two things, held together by caller discipline.
    """
    assert state.latent.shape[1] == 1, "advance steps one block; slice the state first"
    if config.transition == "flow":
        accepted = _flow_candidate(world, state, action, rng, config)
    else:
        accepted = world.predict(state.features, action)

    committed, conditioning = commit_inputs(accepted, rng, config)
    features, agent, memory = world(state.memory, action, committed, conditioning, state.step)
    return WorldState(accepted, memory, state.step + 1, features), agent


def transition_loss(
    world: World, batch, rng: torch.Generator, config: Config, return_agent: bool = False
):
    """Teacher-forced over the window, on real committed latents in both arms.

    Phase 2 asks for the agent readout from this same pass rather than running a
    second one: Dreamer 4 fits the heads in the pretraining setting, so they must
    see the signal range the transition loss sampled, not a uniform condition.
    """
    loss, agent = (_direct_loss if config.transition == "direct" else _shortcut_loss)(
        world, batch, rng, config
    )
    return (loss, agent) if return_agent else loss


def commit_inputs(latent: Tensor, rng: torch.Generator, config: Config):
    """Flow commits at the signal its own conditioning bin names, so the tensor and
    its label cannot disagree; direct has no noise mechanism and commits clean."""
    shape = latent.shape[:2]
    label = torch.stack(
        [
            torch.full(shape, config.tau_ctx_index, device=latent.device),
            torch.full(shape, config.step_index, device=latent.device),
        ],
        dim=-1,
    )
    if config.transition == "direct":
        return latent, label
    signal = config.tau_ctx_signal
    noise = torch.randn(latent.shape, generator=rng, device=latent.device, dtype=latent.dtype)
    return signal * latent + (1.0 - signal) * noise, label


def _direct_loss(world: World, batch, rng: torch.Generator, config: Config) -> Tensor:
    """Teacher forcing plus a generated-prefix rollout, as V-JEPA 2-AC does.

    The rollout term runs the *actual* `advance` transaction from a real prefix:
    commit, predict, commit the prediction, predict again. Stacking independently
    teacher-forced predictions into a fresh sequence is not that -- each was
    conditioned on a different real prefix, so they never form one trajectory, and
    measured, that construction differs from the runtime path by 6.5e-2.

    Direct has no corruption channel to make it tolerate an imperfect prefix, so
    this is its only such training. Gradient flows through one recurrent step,
    matching `auto_steps: 2`. The readout handed to the heads carries the
    generated-prefix block, so they are fitted where imagination reads them.

    The action taken at block t is `led_to_action[t + 1]` under the led-to
    convention -- the same shift the policy target uses.
    """
    committed, conditioning = commit_inputs(batch.latents, rng, config)
    features, agent, memory = world(None, batch.led_to_action, committed, conditioning)
    taken = batch.led_to_action[:, 1:]
    predicted = world.predict(features[:, :-1], taken)
    teacher = (predicted - batch.latents[:, 1:]).pow(2).mean()

    length = batch.latents.shape[1]
    if length < 3:
        return teacher, agent

    prefix, _, memory = world(
        None, batch.led_to_action[:, :-2], committed[:, :-2], conditioning[:, :-2]
    )
    state = WorldState(batch.latents[:, -3:-2], memory, length - 2, prefix[:, -1:])
    first, rolled = advance(world, state, batch.led_to_action[:, -2:-1], rng, config)
    second = world.predict(first.features, batch.led_to_action[:, -1:])
    rollout = (first.latent - batch.latents[:, -2:-1]).pow(2).mean()
    rollout = rollout + (second - batch.latents[:, -1:]).pow(2).mean()
    return teacher + rollout, torch.cat([agent[:, :-1], rolled], dim=1)


def _shortcut_loss(world: World, batch, rng: torch.Generator, config: Config) -> Tensor:
    """Equation 7. At the finest step size the target is the clean latent; at every
    larger step it is the stop-gradient average of two half-steps, which is what
    teaches the step token to mean anything and what makes a four-rung sampler
    work. Without it the arm is the paper's own diffusion-forcing ablation.
    """
    target = batch.latents
    conditioning = flow_conditioning(rng, target.shape[:2], config, target.device)
    tau = signal_level(conditioning, config)[..., None, None]
    noise = torch.randn(target.shape, generator=rng, device=target.device, dtype=target.dtype)
    corrupted = tau * target + (1.0 - tau) * noise
    features, agent, _ = world(None, batch.led_to_action, corrupted, conditioning)
    predicted = world.predict(features)

    finest = conditioning[..., 1] == config.step_index
    weight = 0.9 * tau + 0.1
    flow = (predicted - target).pow(2).mean(dim=(2, 3))

    with torch.no_grad():
        bootstrap = _bootstrap_target(world, batch, corrupted, conditioning, tau, config)
    velocity = (predicted - corrupted) / (1.0 - tau)
    self_loss = ((1.0 - tau) ** 2 * (velocity - bootstrap).pow(2)).mean(dim=(2, 3))

    combined = weight.squeeze(-1).squeeze(-1) * torch.where(finest, flow, self_loss)
    return combined.mean(), agent


def _bootstrap_target(world: World, batch, corrupted, conditioning, tau, config: Config) -> Tensor:
    """Two half-steps, stop-gradient, per Equation 7: b' from tau at d/2, then b''
    from the point b' reaches. Their average is what a single step of size d must
    reproduce."""
    half = torch.stack([conditioning[..., 0], (conditioning[..., 1] + 1).clamp(max=config.step_index)], -1)
    step = (2.0 ** -conditioning[..., 1].float())[..., None, None]

    first, _, _ = world(None, batch.led_to_action, corrupted, half)
    b_first = (world.predict(first) - corrupted) / (1.0 - tau)
    moved = corrupted + b_first * step / 2

    midpoint = ((tau + step / 2).squeeze(-1).squeeze(-1) * config.k_max).round().long()
    shifted = torch.stack([midpoint.clamp(max=config.k_max - 1), half[..., 1]], -1)
    second, _, _ = world(None, batch.led_to_action, moved, shifted)
    b_second = (world.predict(second) - moved) / (1.0 - tau - step / 2)
    return (b_first + b_second) / 2


def _flow_candidate(
    world: World, state: WorldState, action: Tensor, rng: torch.Generator, config: Config
) -> Tensor:
    """The shortcut ladder. At the final rung tau = 1 - d, so the Euler coefficient
    is exactly one and the iterate lands on the predicted clean latent."""
    shape = action.shape
    z = torch.randn(
        (*shape, config.n_spatial, config.d_spatial),
        generator=rng,
        device=action.device,
        dtype=state.latent.dtype,
    )
    scale, step = config.k_max // config.rungs, config.rungs.bit_length() - 1
    for rung in range(config.rungs):
        tau = rung / config.rungs
        indices = torch.stack(
            [
                torch.full(shape, rung * scale, device=z.device),
                torch.full(shape, step, device=z.device),
            ],
            dim=-1,
        )
        features, _, _ = world(state.memory, action, z, indices, state.step)
        z = z + (world.predict(features) - z) / (1.0 - tau) / config.rungs
    return z
