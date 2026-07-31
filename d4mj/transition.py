import torch
from torch import Tensor, nn

from .backbone import AGENT, SPATIAL, Backbone, Layout
from .config import Config
from .representation import Encoder, pack
from .state import Memory, RealState, WorldState

CANDIDATE, COMMIT = 0, 1


class World(nn.Module):
    """The action-conditioned dynamics. One `forward` is one block evaluation, and
    every path differs only in what occupies the spatial slots and what the
    conditioning slot says.

    The conditioning slot exists in both arms: flow puts its signal and step
    embeddings there, direct a two-row candidate/commit table. Both rows are
    reachable at every step, so no row can go untrained -- the failure the upstream
    signal table ships by sizing itself k_max + 1.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layout = Layout.dynamics(config)
        self.spatial = self.layout.span(SPATIAL)
        self.agent = self.layout.span(AGENT)

        self.action_embed = nn.Embedding(config.n_actions + 1, config.d_model)
        if config.transition == "flow":
            self.signal_embed = nn.Embedding(config.n_signal_bins, config.d_model // 2)
            self.step_embed = nn.Embedding(config.n_step_bins, config.d_model // 2)
        else:
            self.family_embed = nn.Embedding(2, config.d_model)
            self.query = nn.Parameter(torch.randn(config.n_spatial, config.d_spatial) * 0.02)

        self.project = nn.Linear(config.d_spatial, config.d_model)
        self.registers = nn.Parameter(torch.randn(config.n_register, config.d_model) * 0.02)
        self.backbone = Backbone(
            config, self.layout, "dynamics", config.d_model, config.n_heads, config.depth
        )
        self.readout = nn.Linear(config.d_model, config.d_spatial)

    def condition(self, indices: Tensor) -> Tensor:
        if self.config.transition == "flow":
            halves = (self.signal_embed(indices[..., 0]), self.step_embed(indices[..., 1]))
            return torch.cat(halves, dim=-1)[:, :, None]
        return self.family_embed(indices[..., 0])[:, :, None]

    def forward(
        self, memory: Memory | None, led_to_action: Tensor, latent: Tensor, conditioning: Tensor
    ) -> tuple[Tensor, Tensor, Memory]:
        b, t = led_to_action.shape
        blocks = torch.cat(
            [
                self.action_embed(led_to_action)[:, :, None],
                self.condition(conditioning),
                self.project(latent),
                self.registers.expand(b, t, -1, -1),
                torch.zeros(b, t, self.config.n_agent, self.config.d_model, device=latent.device),
            ],
            dim=2,
        )
        out, memory = self.backbone(blocks, memory)
        return self.readout(out[:, :, self.spatial]), out[:, :, self.agent], memory


def flow_conditioning(rng: torch.Generator, shape: tuple[int, int], config: Config, device) -> Tensor:
    """Diffusion forcing: an independent step size and signal level per position.

    The signal grid tops out at 1 - d, so tau = 1 is never trained and no path may
    ever present a fully clean latent to the flow arm.
    """
    step = torch.randint(config.n_step_bins, shape, generator=rng, device=device)
    rungs = 2**step
    offset = torch.rand(shape, generator=rng, device=device)
    index = (offset * rungs).floor().long()
    return torch.stack([index * (config.k_max // rungs), step], dim=-1)


def signal_level(conditioning: Tensor, config: Config) -> Tensor:
    return conditioning[..., 0].float() / config.k_max


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
    silently starts from zero memory. Flow commits the latent corrupted to tau_ctx,
    because its training never presents an uncorrupted one; direct commits it
    clean, having no noise mechanism at all.
    """
    encoder_memory = state.encoder_memory if state is not None else None
    world_memory = state.world.memory if state is not None else None
    step = state.world.step if state is not None else 0

    z, encoder_memory, _ = encoder(patches, encoder_memory)
    latent = pack(z, config)
    committed, conditioning = _commit_inputs(latent, rng, config)
    _, agent, world_memory = world(world_memory, led_to_action, committed, conditioning)
    return RealState(encoder_memory, WorldState(latent, world_memory, step + patches.shape[1])), agent


def advance(
    world: World, state: WorldState, led_to_action: Tensor, rng: torch.Generator, config: Config
) -> tuple[WorldState, Tensor]:
    """One semantic transition: N read-only candidate evaluations, then exactly one
    commit. Flow N = 4 rungs, direct N = 1 query pass. A candidate's memory is
    always discarded, so the committed prefix only ever holds a latent the model
    will condition on later -- never a placeholder, never a mid-ladder iterate.
    """
    accepted = (_flow_candidate if config.transition == "flow" else _direct_candidate)(
        world, state, led_to_action, rng, config
    )
    committed, conditioning = _commit_inputs(accepted, rng, config)
    _, agent, memory = world(state.memory, led_to_action, committed, conditioning)
    return WorldState(accepted, memory, state.step + 1), agent


def transition_loss(world: World, batch, rng: torch.Generator, config: Config) -> Tensor:
    """Teacher-forced over the window. Flow trains shortcut forcing in x-space with
    the ramp weight; direct predicts the frozen target from a candidate query."""
    target = batch.latents
    if config.transition == "direct":
        query = world.query.expand(*target.shape[:2], -1, -1)
        conditioning = torch.full((*target.shape[:2], 2), CANDIDATE, device=target.device)
        predicted, _, _ = world(None, batch.led_to_action, query, conditioning)
        return (predicted - target).pow(2).mean()

    conditioning = flow_conditioning(rng, target.shape[:2], config, target.device)
    tau = signal_level(conditioning, config)[..., None, None]
    corrupted = (1 - tau) * torch.randn_like(target) + tau * target
    predicted, _, _ = world(None, batch.led_to_action, corrupted, conditioning)
    weight = 0.9 * tau + 0.1
    return (weight * (predicted - target).pow(2)).mean()


def _commit_inputs(latent: Tensor, rng: torch.Generator, config: Config):
    shape = latent.shape[:2]
    if config.transition == "direct":
        indices = torch.full((*shape, 2), COMMIT, device=latent.device)
        return latent, indices
    noise = config.tau_ctx_noise
    corrupted = (1 - noise) * latent + noise * torch.randn(
        latent.shape, generator=rng, device=latent.device
    )
    indices = torch.stack(
        [
            torch.full(shape, config.tau_ctx_index, device=latent.device),
            torch.full(shape, config.step_index, device=latent.device),
        ],
        dim=-1,
    )
    return corrupted, indices


def _flow_candidate(
    world: World, state: WorldState, led_to_action: Tensor, rng: torch.Generator, config: Config
) -> Tensor:
    """The shortcut ladder. At the final rung tau = 1 - d, so the Euler coefficient
    is exactly one and the iterate lands on the predicted clean latent."""
    shape = led_to_action.shape
    z = torch.randn(
        (*shape, config.n_spatial, config.d_spatial), generator=rng, device=led_to_action.device
    )
    scale = config.k_max // config.rungs
    step = config.rungs.bit_length() - 1
    for rung in range(config.rungs):
        tau = rung / config.rungs
        indices = torch.stack(
            [
                torch.full(shape, rung * scale, device=z.device),
                torch.full(shape, step, device=z.device),
            ],
            dim=-1,
        )
        predicted, _, _ = world(state.memory, led_to_action, z, indices)
        z = z + (predicted - z) / (1.0 - tau) / config.rungs
    return z


def _direct_candidate(
    world: World, state: WorldState, led_to_action: Tensor, rng: torch.Generator, config: Config
) -> Tensor:
    query = world.query.expand(*led_to_action.shape, -1, -1)
    indices = torch.full((*led_to_action.shape, 2), CANDIDATE, device=led_to_action.device)
    predicted, _, _ = world(state.memory, led_to_action, query, indices)
    return torch.tanh(predicted)
