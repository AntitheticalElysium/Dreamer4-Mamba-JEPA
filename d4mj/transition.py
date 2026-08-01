import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import AGENT, REGISTER, SPATIAL, Backbone, Layout
from .config import Config
from .representation import Encoder, pack
from .state import Memory, RealState, WorldState


class World(nn.Module):
    """The action-conditioned dynamics. `forward` evaluates one block; `predict`
    turns its features into a latent and is the only thing the arms do differently.
    Direct uses an external head over committed features (S34), reading spatial and
    register slots only, which the dynamics mask keeps agent-free."""

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


def flow_conditioning(rng: torch.Generator, shape: tuple[int, int], config: Config, device):
    """Diffusion forcing: an independent step size and signal level per position,
    plus the mask of positions the loss may score.

    `commit_prefix_fraction` of rows instead carry the rollout prefix -- every block
    but the last at the commit condition -- which independent draws make
    vanishingly rare (S41). Only that row's last block is scored; scoring the prefix
    too would push the finest-step share from 25% to 43%."""
    step = torch.randint(config.n_step_bins, shape, generator=rng, device=device)
    rungs = 2**step
    index = (torch.rand(shape, generator=rng, device=device) * rungs).floor().long()
    conditioning = torch.stack([index * (config.k_max // rungs), step], dim=-1)
    scored = torch.ones(shape, device=device)

    rows = int(config.commit_prefix_fraction * shape[0])
    if rows:
        picked = torch.randperm(shape[0], generator=rng, device=device)[:rows]
        conditioning[picked, :-1, 0] = config.tau_ctx_index
        conditioning[picked, :-1, 1] = config.step_index
        scored[picked, :-1] = 0.0
    return conditioning, scored


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
    """One semantic transition: flow runs its read-only rungs then commits, direct
    predicts and commits once. Must span exactly one block. Incoming memory is
    detached to equalise the two time mixers (S55); gradient still flows through the
    accepted latent."""
    assert state.latent.shape[1] == 1, "advance steps one block; slice the state first"
    if config.transition == "flow":
        accepted = _flow_candidate(world, state, action, rng, config)
    else:
        accepted = world.predict(state.features, action)

    memory = None if state.memory is None else tuple(
        tuple(tensor.detach() for tensor in pair) for pair in state.memory
    )
    committed, conditioning = commit_inputs(accepted, rng, config)
    features, agent, memory = world(memory, action, committed, conditioning, state.step)
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
    """Teacher forcing plus a two-step generated-prefix rollout, after V-JEPA 2-AC
    (S55). Both generated states are committed through `advance`, and both readouts
    replace the real ones at their own indices, so the heads see every state Phase 3
    will read them at. The two rollout terms are averaged, matching the source's
    `jloss + sloss` of two means; squared error is a declared deviation from its L1.
    """
    committed, conditioning = commit_inputs(batch.latents, rng, config)
    features, agent, memory = world(None, batch.led_to_action, committed, conditioning)
    taken = batch.led_to_action[:, 1:]
    predicted = world.predict(features[:, :-1], taken)
    teacher = (predicted - batch.latents[:, 1:]).pow(2).mean(dim=(1, 2, 3))

    length = batch.latents.shape[1]
    if length < 3:
        return _uniform_mean(teacher, batch), agent

    prefix, _, memory = world(
        None, batch.led_to_action[:, :-2], committed[:, :-2], conditioning[:, :-2]
    )
    state = WorldState(batch.latents[:, -3:-2], memory, length - 2, prefix[:, -1:])
    first, rolled = advance(world, state, batch.led_to_action[:, -2:-1], rng, config)
    second, rolled_again = advance(world, first, batch.led_to_action[:, -1:], rng, config)
    rollout = (first.latent - batch.latents[:, -2:-1]).pow(2).mean(dim=(1, 2, 3))
    rollout = rollout + (second.latent - batch.latents[:, -1:]).pow(2).mean(dim=(1, 2, 3))
    readout = torch.cat([agent[:, :-2], rolled, rolled_again], dim=1)
    return _uniform_mean(teacher + rollout / 2, batch), readout


def _shortcut_loss(world: World, batch, rng: torch.Generator, config: Config) -> Tensor:
    """Equation 7. At the finest step size the target is the clean latent; at every
    larger step it is the stop-gradient average of two half-steps, which is what
    teaches the step token to mean anything and what makes a four-rung sampler
    work. Without it the arm is the paper's own diffusion-forcing ablation.
    """
    target = batch.latents
    conditioning, scored = flow_conditioning(rng, target.shape[:2], config, target.device)
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

    combined = weight.squeeze(-1).squeeze(-1) * torch.where(finest, flow, self_loss) * scored
    return _uniform_mean(combined.sum(dim=1) / scored.sum(dim=1), batch), agent


def _uniform_mean(per_row: Tensor, batch) -> Tensor:
    """Every row during pretraining, the uniform half during finetuning, never a
    support row. D4 pretrains on the whole corpus and then restricts the continued
    dynamics loss to uniform rows "to avoid optimistic generations" (§4.1)."""
    mask = batch.rows("dynamics").to(per_row.device).float()
    return (per_row * mask).sum() / mask.sum().clamp(min=1.0)


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
