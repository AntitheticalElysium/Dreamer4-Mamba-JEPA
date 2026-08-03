import copy
from dataclasses import replace

import torch
from torch import nn

from .actor_critic import actor_loss, critic_loss, lambda_returns
from .agent import Heads, head_loss, head_targets, terminal_loss
from .config import Config
from .data import FORMAT, Batch, Episode, patchify, sample_batch, sample_terminal_batch
from .imagination import imagine
from .checkpoint import load, save
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
    episodes: list[Episode], steps: int, config: Config, checkpoint=None
) -> tuple[Encoder, Decoder, list[Episode]]:
    """Phase 1A, returning the frozen encoder, its decoder, and the latent cache.

    `batch.scored` decides which blocks the loss uses, per row: a block counts once
    it holds a full receptive field, and unconditionally in a window that starts at
    the episode start, where nothing earlier is missing. The cache is written under
    the same bounded window -- scanning whole episodes unbounded would give the same
    frame a different latent than deployment does.
    """
    import lpips

    device = config.device
    torch.manual_seed(config.seed)
    encoder, decoder = Encoder(config).to(device), Decoder(config).to(device)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)

    optimiser = optimizer([encoder, decoder], config)
    sampler, rng = _generators(config, 0)
    balance: dict[str, float] = {}
    bundle, streams = [encoder, decoder, optimiser], {"sampler": sampler, "model": rng}
    resume = _checkpoint(checkpoint, config, bundle, balance, streams, contract=f"1A:{steps}")

    for step in range(resume, steps):
        batch = _to(sample_batch(episodes, sampler, config, step, steps), device)
        z, _, masked = encoder(batch.patches, p_mask=config.mae_p_max, rng=rng)
        predicted, _ = decoder(z)
        losses = reconstruction_loss(
            predicted, batch.patches, masked, batch.scored, perceptual, config
        )
        weights = {"lpips": config.lpips_weight}
        loss = _balance(losses, balance, config, weights)
        _update(optimiser, loss, [encoder, decoder], config, step)
        if checkpoint is not None and ((step + 1) % config.checkpoint_every == 0 or step + 1 == steps):
            _checkpoint(checkpoint, config, bundle, balance, streams, step + 1, f"1A:{steps}")

    encoder.eval()
    return encoder, decoder.eval(), cache_latents(encoder, episodes, config)


@torch.no_grad()
def cache_latents(encoder: Encoder, episodes: list[Episode], config: Config) -> list[Episode]:
    """Scan each episode once under the declared window, at mask probability zero.

    Chunked with memory carried, not one dense call: an episode runs to thousands
    of frames and a single scan builds an attention problem that size. The windowed
    mask makes chunked and fully recurrent encoding produce the same latent, which
    `scan_step_parity` asserts, so the cheap path is also the faithful one.
    """
    device, digest = config.device, _cache_digest(encoder, config)
    cached = []
    for episode in episodes:
        frames = patchify(episode.observations[None], config.patch).to(device)
        latents, memory = [], None
        for start in range(0, frames.shape[1], config.sequence_long):
            chunk = frames[:, start : start + config.sequence_long]
            z, memory, _ = encoder(chunk, memory, offset=start)
            latents.append(z)
        packed = pack(torch.cat(latents, dim=1), config)[0].cpu()
        cached.append(replace(episode, latents=packed, latent_digest=digest))
    return cached


def train_dynamics(episodes: list[Episode], steps: int, config: Config, checkpoint=None) -> World:
    """Phase 1B, on the frozen cache. Agent slots are already present and masked,
    so no state shape changes at the Phase 2 boundary."""
    device = config.device
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(device)
    optimiser = optimizer([world], config)
    balance: dict[str, float] = {}
    sampler, rng = _generators(config, 1)
    streams = {"sampler": sampler, "model": rng}
    resume = _checkpoint(checkpoint, config, [world, optimiser], balance, streams, contract=f"1B:{steps}")

    for step in range(resume, steps):
        batch = _to(sample_batch(episodes, sampler, config, step, steps), device)
        dynamics = transition_loss(world, batch, rng, config, step=step)
        loss = _balance({"dynamics": dynamics}, balance, config)
        _update(optimiser, loss, [world], config, step)
        if checkpoint is not None and ((step + 1) % config.checkpoint_every == 0 or step + 1 == steps):
            _checkpoint(checkpoint, config, [world, optimiser], balance, streams, step + 1, f"1B:{steps}")
    return world


def train_agent(
    episodes: list[Episode],
    world: World,
    steps: int,
    config: Config,
    checkpoint=None,
    world_steps: int = 0,
) -> Heads:
    """Phase 2. The dynamics objective continues alongside the head losses, which
    keeps the world model from drifting while the heads fit it.

    Heads read the readout from the *same* diffusion-forced pass the transition
    loss uses, not a separately committed one: Dreamer 4 reuses the pretraining
    setting so the heads are fitted across the sampled signal range rather than at
    one uniform condition they will never see again.

    `world_steps` is how far Phase 1B already trained the world, so the shortcut
    bootstrap clock (S67) continues rather than restarting -- otherwise Phase 2
    would spend its first `bootstrap_start` steps back on the supervised objective.
    """
    device = config.device
    torch.manual_seed(config.seed + 2)
    heads = Heads(config).to(device)
    optimiser = optimizer([world, heads], config)
    sampler, rng = _generators(config, 2)
    balance: dict[str, float] = {}
    bundle, streams = [world, heads, optimiser], {"sampler": sampler, "model": rng}
    contract = f"2:{world_steps}:{steps}"
    resume = _checkpoint(checkpoint, config, bundle, balance, streams, contract=contract)

    for step in range(resume, steps):
        batch = _to(sample_batch(episodes, sampler, config, step, steps, mixture=True), device)
        dynamics, agent = transition_loss(
            world, batch, rng, config, return_agent=True, step=world_steps + step
        )
        readout = heads(agent) | {"centers": heads.centers}
        losses = {"dynamics": dynamics} | head_loss(readout, head_targets(batch, config), config)

        terminal = _to(sample_terminal_batch(episodes, sampler, config, step, steps), device)
        committed, conditioning = commit_inputs(terminal.latents, rng, config)
        _, terminal_agent, _ = world(
            None, terminal.led_to_action, committed, conditioning
        )
        terminal_readout = heads(terminal_agent) | {"centers": heads.centers}
        losses["continuation"] = (
            (1.0 - config.terminal_loss_mass) * losses["continuation"]
            + config.terminal_loss_mass
            * terminal_loss(terminal_readout, head_targets(terminal, config))
        )
        _update(optimiser, _balance(losses, balance, config), [world, heads], config, step)
        if checkpoint is not None and ((step + 1) % config.checkpoint_every == 0 or step + 1 == steps):
            _checkpoint(checkpoint, config, bundle, balance, streams, step + 1, contract)
    return heads


def train_actor(
    episodes: list[Episode],
    world: World,
    heads: Heads,
    steps: int,
    config: Config,
    checkpoint=None,
) -> Heads:
    """Phase 3. The world is frozen and the behaviour-cloned policy is copied and
    frozen as the prior, so the actor cannot improve by reshaping the model it is
    being scored inside. The reward and continuation body is frozen with it.

    Starting contexts use the §4.1 mixture. D4 applies it to "behavioral cloning,
    reward modeling, and reinforcement learning", so imagining only from uniformly
    drawn contexts starts RL away from the events that carry the sparse reward."""
    device = config.device
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    prior = copy.deepcopy(heads).eval()
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    for parameter in heads.actor_parameters():
        parameter.requires_grad_(True)
    optimiser = optimizer([heads], config)
    sampler, rng = _generators(config, 3)
    policy_rng = torch.Generator(device=device).manual_seed(config.seed + 2**20)
    balance: dict[str, float] = {}
    streams = {"sampler": sampler, "model": rng, "policy": policy_rng}
    frozen = f"3:{steps}:{config.actor_batch}:{_identity(world, prior)}"
    resume = _checkpoint(checkpoint, config, [heads, optimiser], balance, streams, contract=frozen)
    sampling = replace(config, batch=config.actor_batch)

    for step in range(resume, steps):
        batch = _to(sample_batch(episodes, sampler, sampling, step, steps, mixture=True), device)
        with torch.no_grad():
            committed, conditioning = commit_inputs(batch.latents, rng, config)
            features, agent, memory = world(None, batch.led_to_action, committed, conditioning)
        begin = WorldState(batch.latents[:, -1:], memory, batch.latents.shape[1], features[:, -1:])
        trajectory = imagine(world, heads, begin, agent[:, -1:], rng, policy_rng, config)

        returns = lambda_returns(trajectory, config)
        with torch.no_grad():
            reference = prior(trajectory.agent[:, :-1])["policy"][:, :, 0]
        losses = {
            "actor": actor_loss(trajectory, returns, reference, config),
            "critic": critic_loss(heads(trajectory.agent[:, :-1])["value"], returns, heads.centers),
        }
        _update(optimiser, _balance(losses, balance, config), [heads], config, step)
        if checkpoint is not None and ((step + 1) % config.checkpoint_every == 0 or step + 1 == steps):
            _checkpoint(
                checkpoint, config, [heads, optimiser], balance, streams, step + 1, frozen
            )
    return heads


def _checkpoint(
    path, config: Config, modules: list, balance: dict, streams: dict, step=None, contract: str = ""
) -> int:
    """Both directions of a mid-phase resume: with `step` it saves, without it
    restores and returns the step to continue from.

    Optimizer state, the running-RMS normalisers and every generator stream travel;
    restoring weights alone is not a resume. `contract` fixes what `Config` cannot:
    the planned phase length, which sets the short/long schedule, and the frozen
    world and prior a phase is trained against.
    """
    named = {f"part{index}": module for index, module in enumerate(modules)}
    if step is not None:
        save(path, config, step=step, balance=balance, contract=contract,
             generators=generator_state(**streams), **named)
        return step
    if path is None or not path.exists():
        return 0
    stored = load(path, config, balance=balance, **named)["modules"]
    if stored.get("contract", "") != contract:
        raise ValueError(
            f"checkpoint contract {stored.get('contract', '')!r} does not match {contract!r}: "
            "the phase length or the frozen model it was trained against has changed"
        )
    for name, generator in streams.items():
        generator.set_state(stored["generators"][name])
    return int(stored["step"])


def _identity(*modules: nn.Module) -> str:
    """A digest of frozen modules a phase is scored against, so a resume cannot
    silently swap them. `Config` matching is not enough: two worlds with the same
    config are different learned environments."""
    import hashlib

    digest = hashlib.sha256()
    for module in modules:
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def _share_initialisation(world: World, config: Config) -> World:
    """Same starting weights wherever the arms share a parameter (S45). `manual_seed`
    alone does not achieve it: the mixers consume different numbers of draws, so
    everything built after the first time layer would differ."""
    if config.time_mixer == "attention":
        return world
    torch.manual_seed(config.seed + 1)
    reference = World(replace(config, time_mixer="attention")).state_dict()
    current = world.state_dict()
    shared = {
        name: tensor
        for name, tensor in reference.items()
        if name in current and current[name].shape == tensor.shape
    }
    world.load_state_dict(shared, strict=False)
    return world


def _balance(
    losses: dict[str, torch.Tensor],
    state: dict[str, float],
    config: Config,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Running-RMS normalisation, so a coefficient is a relative weight rather than
    a scale accident. `weights` apply *after* it: folded in first they are divided
    straight back out, and 0.2 and 5.0 measure identically."""
    total = 0.0
    for name, value in losses.items():
        squared = float(value.detach().pow(2))
        state[name] = config.rms_decay * state.get(name, squared) + (1 - config.rms_decay) * squared
        scale = (weights or {}).get(name, 1.0)
        total = total + scale * value / max(state[name] ** 0.5, 1e-8)
    return total


def _cache_digest(encoder: Encoder, config: Config) -> str:
    """Identity of the latent cache: the whole latent function, not just its weights.

    Every field the encoder's forward pass depends on is included. Weights alone are
    not identity -- two encoders with identical parameters but different resolution
    and patch layout produced the same digest, and the cache from one would load
    against the other. The time mixer is excluded because the tokenizer is shared
    and always attention.
    """
    import hashlib

    shape = (
        config.patch,
        config.resolution,
        config.channels,
        config.n_patches,
        config.window,
        config.n_latents,
        config.d_bottleneck,
        config.packing,
        config.d_model_encoder,
        config.depth_encoder,
        config.n_heads_encoder,
        config.time_every,
        config.receptive_field,
        FORMAT,
    )
    weights = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        weights.update(name.encode())
        weights.update(tensor.detach().cpu().numpy().tobytes())
    visual = source_digests(replace(config, time_mixer="attention"))
    return hashlib.sha256(repr((shape, visual, weights.hexdigest())).encode()).hexdigest()[:16]




def _generators(config: Config, phase: int) -> tuple[torch.Generator, torch.Generator]:
    """A CPU generator for the sampler and a device generator for model noise, each
    seeded independently. Drawing one seed from the other fails outright when the
    device generator is CUDA, and couples two streams that should be separable."""
    sampler = torch.Generator().manual_seed(config.seed + phase)
    model = torch.Generator(device=config.device).manual_seed(config.seed + 1000 + phase)
    return sampler, model


def generator_state(**streams: torch.Generator) -> dict:
    """Every stream a phase draws from, in a form `checkpoint.save` stores.
    `torch.get_rng_state()` captures the global stream, which nothing here draws
    from -- saving it and calling that resumable would replay every window and every
    noise draw from step zero."""
    return {name: generator.get_state() for name, generator in streams.items()}


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
