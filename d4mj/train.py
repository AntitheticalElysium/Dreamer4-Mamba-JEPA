import copy
from dataclasses import replace

import torch
from torch import nn

from .actor_critic import actor_loss, critic_loss, lambda_returns
from .agent import Heads, head_loss, head_targets
from .config import Config
from .data import Batch, Episode, patchify, sample_batch
from .imagination import imagine
from .representation import Decoder, Encoder, pack, reconstruction_loss
from .sources import source_digests
from .state import WorldState
from .transition import World, commit_inputs, transition_loss


def optimizer(modules: list[nn.Module], config: Config) -> torch.optim.AdamW:
    """The only place parameter groups are built, and the only place Mamba-2's
    no-weight-decay contract is honoured.

    Upstream marks A_log, D and dt_bias exempt. Decaying them is an asymmetry
    applied to exactly one arm of the single comparison the project exists to
    make, which is why it cannot live at a call site.
    """
    decayed, exempt = [], []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                (exempt if getattr(parameter, "_no_weight_decay", False) else decayed).append(parameter)
    groups = [{"params": decayed, "weight_decay": config.weight_decay}]
    if exempt:
        groups.append({"params": exempt, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def train_representation(
    episodes: list[Episode], steps: int, config: Config
) -> tuple[Encoder, Decoder, list[Episode]]:
    """Phase 1A, returning the frozen encoder, its decoder, and the latent cache.

    Windows carry a burn-in that updates encoder memory and scores nothing, so
    every scored latent has the history it will have at deployment. The cache is
    then written under that same bounded window -- scanning whole episodes
    unbounded would give the same frame a different latent than deployment does.
    """
    import lpips

    device = _device(config)
    torch.manual_seed(config.seed)
    encoder, decoder = Encoder(config).to(device), Decoder(config).to(device)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)

    optimiser = optimizer([encoder, decoder], config)
    rng = torch.Generator(device=device).manual_seed(config.seed)

    for step in range(steps):
        batch = _to(sample_batch(episodes, _cpu(rng), config, step), device)
        z, _, masked = encoder(batch.patches, p_mask=config.mae_p_max, rng=rng)
        predicted, _ = decoder(z)
        scored = slice(batch.burn_in, None)
        loss = reconstruction_loss(
            predicted[:, scored], batch.patches[:, scored], masked[:, scored], perceptual, config
        )
        _update(optimiser, loss, [encoder, decoder], config, step)

    encoder.eval()
    return encoder, decoder.eval(), cache_latents(encoder, episodes, config)


@torch.no_grad()
def cache_latents(encoder: Encoder, episodes: list[Episode], config: Config) -> list[Episode]:
    """Scan each episode once under the declared window, at mask probability zero.

    One scan per episode rather than a frame-by-frame walk: the windowed mask makes
    a scan and the deployed recurrence produce the same latent, which
    `scan_step_parity` asserts, so the cheap path is also the faithful one.
    """
    device, digest = _device(config), _cache_digest(config)
    cached = []
    for episode in episodes:
        frames = patchify(episode.observations[None], config.patch).to(device)
        z, _, _ = encoder(frames)
        cached.append(replace(episode, latents=pack(z, config)[0].cpu(), latent_digest=digest))
    return cached


def train_dynamics(episodes: list[Episode], steps: int, config: Config) -> World:
    """Phase 1B, on the frozen cache. Agent slots are already present and masked,
    so no state shape changes at the Phase 2 boundary."""
    device = _device(config)
    torch.manual_seed(config.seed + 1)
    world = World(config).to(device)
    optimiser = optimizer([world], config)
    rng = torch.Generator(device=device).manual_seed(config.seed + 1)

    for step in range(steps):
        batch = _to(sample_batch(episodes, _cpu(rng), config, step), device)
        _update(optimiser, transition_loss(world, batch, rng, config), [world], config, step)
    return world


def train_agent(episodes: list[Episode], world: World, steps: int, config: Config) -> Heads:
    """Phase 2. The dynamics objective continues alongside the head losses, which
    keeps the world model from drifting while the heads fit it.

    Heads read the readout from the *same* diffusion-forced pass the transition
    loss uses, not a separately committed one: Dreamer 4 reuses the pretraining
    setting so the heads are fitted across the sampled signal range rather than at
    one uniform condition they will never see again.
    """
    device = _device(config)
    torch.manual_seed(config.seed + 2)
    heads = Heads(config).to(device)
    optimiser = optimizer([world, heads], config)
    rng = torch.Generator(device=device).manual_seed(config.seed + 2)
    balance: dict[str, float] = {}

    for step in range(steps):
        batch = _to(sample_batch(episodes, _cpu(rng), config, step), device)
        dynamics, agent = transition_loss(world, batch, rng, config, return_agent=True)
        readout = heads(agent) | {"centers": heads.centers}
        losses = {"dynamics": dynamics, "heads": head_loss(readout, head_targets(batch, config), config)}
        _update(optimiser, _balance(losses, balance, config), [world, heads], config, step)
    return heads


def train_actor(
    episodes: list[Episode], world: World, heads: Heads, steps: int, config: Config
) -> Heads:
    """Phase 3. The world is frozen and the behaviour-cloned policy is copied and
    frozen as the prior, so the actor cannot improve by reshaping the model it is
    being scored inside."""
    device = _device(config)
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    prior = copy.deepcopy(heads).eval()
    optimiser = optimizer([heads], config)
    rng = torch.Generator(device=device).manual_seed(config.seed + 3)
    balance: dict[str, float] = {}

    for step in range(steps):
        batch = _to(sample_batch(episodes, _cpu(rng), config, step), device)
        with torch.no_grad():
            committed, conditioning = commit_inputs(batch.latents, rng, config)
            features, agent, memory = world(None, batch.led_to_action, committed, conditioning)
        start = WorldState(batch.latents[:, -1:], memory, batch.latents.shape[1], features[:, -1:])
        trajectory = imagine(world, heads, start, agent[:, -1:], rng, config)

        returns = lambda_returns(trajectory, config)
        with torch.no_grad():
            reference = prior(trajectory.agent[:, :-1])["policy"][:, :, 0]
        losses = {
            "actor": actor_loss(trajectory, returns, reference, config),
            "critic": critic_loss(heads(trajectory.agent[:, :-1])["value"], returns, heads.centers),
        }
        _update(optimiser, _balance(losses, balance, config), [heads], config, step)
    return heads


def _balance(losses: dict[str, torch.Tensor], state: dict[str, float], config: Config) -> torch.Tensor:
    """Dreamer 4 normalises concurrent losses by running RMS estimates, which makes
    a coefficient a relative weight rather than a scale accident. `state` is a plain
    dict so it checkpoints with everything else."""
    total = 0.0
    for name, value in losses.items():
        squared = float(value.detach().pow(2))
        state[name] = config.rms_decay * state.get(name, squared) + (1 - config.rms_decay) * squared
        total = total + value / max(state[name] ** 0.5, 1e-8)
    return total


def _cache_digest(config: Config) -> str:
    import hashlib

    parts = (config.patch, config.window, config.n_latents, config.d_bottleneck, config.packing)
    return hashlib.sha256(repr((parts, source_digests(config))).encode()).hexdigest()[:16]


def _device(config: Config) -> str:
    return "cuda" if config.time_mixer == "mamba" else "cpu"


def _cpu(rng: torch.Generator) -> torch.Generator:
    """The sampler indexes on CPU while the model draws noise on device. One seed,
    two generators, so a paired arm is reproducible on either backend."""
    return torch.Generator().manual_seed(int(torch.randint(2**31, (1,), generator=rng)))


def _to(batch: Batch, device: str) -> Batch:
    moved = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in vars(batch).items()
    }
    return Batch(**moved)


def _update(optimiser, loss, modules, config: Config, step: int) -> None:
    for group in optimiser.param_groups:
        group["lr"] = config.learning_rate * min(1.0, (step + 1) / config.warmup)
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for module in modules for p in module.parameters()], config.grad_clip
    )
    optimiser.step()
