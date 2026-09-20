import copy
from contextlib import nullcontext
import math
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .actor_critic import actor_loss, critic_loss, lambda_returns
from .agent import Heads, _distribution_loss, head_loss, head_targets, paired_terminal_loss
from .config import Config
from .lewm_config import LeWMConfig
from .cache import cache_latents, cache_latents_to_store, _cache_digest
from .data import (
    Batch,
    BridgeBatch,
    JointSampler,
    Episode,
    EpisodeCorpus,
    sample_batch,
    sample_bridge_batch,
    sample_bridge_terminals,
    sample_terminal_batch,
    to_head_batch,
)
from .imagination import imagine
from .checkpoint import (load, save, restore_lewm_bundle, save_lewm_bundle,
                         publish_lewm_latest, save_lewm_bridge, restore_lewm_bridge,
                         save_lewm_actor, restore_lewm_actor)
from .representation import Decoder, Encoder, reconstruction_loss
from .sources import tensor_state_digest
from .state import PredictiveState, WorldState, repeat_memory as _repeat_memory
from .mamba_recurrence import MambaCarry
from .lewm import SIGReg, joint_loss
from .world_api import ModelBundle
from .transition import advance, World, commit_inputs, transition_loss


def optimizer(modules: list[nn.Module], config, *, exclude_vectors: bool = False) -> torch.optim.AdamW:
    """One AdamW grouping implementation, with explicitly different recipe policies.

    Legacy phases retain their historical decay of ordinary vectors. Joint LeWM
    exempts vectors/biases as declared in TC-12. Both honor upstream Mamba's
    `_no_weight_decay`; order is preserved for optimizer state and numerical parity.
    """
    decayed, exempt = [], []
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                no_decay = getattr(parameter, "_no_weight_decay", False) or (
                    exclude_vectors and (parameter.ndim < 2 or name.endswith("bias")))
                (exempt if no_decay else decayed).append(parameter)
    groups = [{"params": decayed, "weight_decay": config.weight_decay}]
    if exempt:
        groups.append({"params": exempt, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=config.learning_rate,
                             betas=getattr(config, "betas", (0.9,0.999)),
                             eps=getattr(config, "optimizer_eps", 1e-8))


def optimizer_step(optimiser, loss, parameters, *, learning_rate: float, grad_clip: float,
                   strict: bool = False, zero_grad: bool = True):
    """Shared backward/clipping/update; callers own objectives and schedules."""
    for group in optimiser.param_groups:
        group["lr"] = learning_rate
    if zero_grad:
        optimiser.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=strict)
    optimiser.step()
    return norm


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


def _counterfactual_path(world: World, heads: Heads, batch: dict, rng, config: Config):
    """All 17 actions from one state, plus one further step where the branch survived.

    Dynamics on both generated latents, and reward/continuation heads on the readouts
    those generations produce -- supervising dynamics alone would leave the outcome heads
    on logged trajectories, the coverage that made death prediction fail. No policy loss:
    these actions were never chosen by a policy.
    """
    spatial, d = config.n_spatial, config.d_spatial
    z, actions = batch["history"], batch["actions"]
    n, t = z.shape[0], z.shape[1]
    committed, conditioning = commit_inputs(z.view(n, t, spatial, d), rng, config)
    features, _, memory = world(None, batch["led"], committed, conditioning)

    a = actions.shape[1]
    flat = actions.reshape(-1, 1)
    state = WorldState(z[:, -1:].view(n, 1, spatial, d).repeat_interleave(a, 0),
                       _repeat_memory(memory, n, a), t,
                       features[:, -1:].repeat_interleave(a, dim=0))
    first, agent = advance(world, state, flat, rng, config)
    dynamics = (first.latent.flatten(2)[:, 0] - batch["branch"]).pow(2).mean()

    readout = heads(agent)
    centers = heads.centers
    reward = _distribution_loss(readout["reward"][:, :, 0], batch["reward"][:, None],
                                centers).mean()
    continuation = F.binary_cross_entropy_with_logits(
        readout["continuation"][:, 0], batch["continuation"][:, None]).mean()

    keep = batch["valid"]
    if bool(keep.any()):
        noop = torch.zeros_like(flat)
        step_two, agent_two = advance(world, first, noop, rng, config)
        error = (step_two.latent.flatten(2)[:, 0] - batch["second"]).pow(2).mean(-1)
        # `_direct_loss` averages its two rollout terms; the second is scored only where
        # the first action left the branch alive, and the invalid targets are zeros that
        # must never enter the loss
        dynamics = dynamics + 0.5 * (error * keep).sum() / keep.sum()
        second_readout = heads(agent_two)
        reward = reward + 0.5 * (
            _distribution_loss(second_readout["reward"][:, :, 0],
                               batch["second_reward"][:, None], centers)[:, 0] * keep
        ).sum() / keep.sum()
        continuation = continuation + 0.5 * (
            F.binary_cross_entropy_with_logits(
                second_readout["continuation"][:, 0, 0],
                batch["second_continuation"], reduction="none") * keep
        ).sum() / keep.sum()
    return {"dynamics": dynamics, "reward": reward, "continuation": continuation}


def train_agent(
    episodes: list[Episode],
    world: World,
    steps: int,
    config: Config,
    checkpoint=None,
    world_steps: int = 0,
    terminal_dynamics_mass: float = 0.0,
    counterfactual=None,
    counterfactual_mass: float = 0.0,
    paired_semantic: bool = False,
    rollout_only: bool = False,
    freeze_world: bool = False,
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

    `terminal_dynamics_mass` is an explicit ablation. At zero, terminal tails only
    supervise continuation as before. Above zero, the same ordinary dynamics loss
    also scores those realized tail transitions; BC remains excluded.
    """
    if not 0.0 <= terminal_dynamics_mass < 1.0:
        raise ValueError("terminal_dynamics_mass must be in [0, 1)")
    if not 0.0 <= counterfactual_mass < 1.0:
        raise ValueError("counterfactual_mass must be in [0, 1)")
    if counterfactual is not None and counterfactual_mass == 0.0:
        raise ValueError("a counterfactual corpus with zero mass would never be read")
    device = config.device
    torch.manual_seed(config.seed + 2)
    heads = Heads(config).to(device)
    # `freeze_world` turns this into a head-extraction ceiling: the readouts are whatever
    # the given world already produces, so a difference between two arms cannot come from
    # the world or from its interaction with the heads. The dynamics loss is still
    # computed -- the readouts come from that same pass -- but carries no gradient, and
    # `_balance` normalises each loss by its own RMS, so it cannot reweight the heads.
    if freeze_world:
        for parameter in world.parameters():
            parameter.requires_grad_(False)
    optimiser = optimizer([heads] if freeze_world else [world, heads], config)
    sampler, rng = _generators(config, 2)
    balance: dict[str, float] = {}
    bundle, streams = [world, heads, optimiser], {"sampler": sampler, "model": rng}
    contract = f"2:{world_steps}:{steps}"
    contract += ":continuation=paired-v2"
    # Both change what the run optimises, so a checkpoint must not resume under the
    # opposite setting.
    if paired_semantic:
        contract += ":paired_semantic"
    if rollout_only:
        contract += ":rollout_only"
    if freeze_world:
        contract += ":frozen_world"
    if counterfactual is not None:
        contract += f":counterfactual={counterfactual_mass:.17g}"
    if terminal_dynamics_mass:
        contract += f":terminal_dynamics={terminal_dynamics_mass:.17g}"
    resume = _checkpoint(checkpoint, config, bundle, balance, streams, contract=contract)

    for step in range(resume, steps):
        batch = _to(sample_batch(episodes, sampler, config, step, steps, mixture=True), device)
        dynamics, agent, observed = transition_loss(
            world, batch, rng, config, return_agent=True, return_observed=True,
            step=world_steps + step,
        )
        targets = head_targets(batch, config)
        # The generated readouts are the last `direct_rollout` blocks and nothing else,
        # so scoring every block compares two objectives that are the same tensor almost
        # everywhere: measured, 2 of 16 blocks differ on short rows and 2 of 64 on long
        # ones. `rollout_only` confines both arms to the blocks where they can differ.
        window = None
        if rollout_only:
            window = torch.zeros(agent.shape[1], device=device)
            window[-config.direct_rollout:] = 1.0
        readout = heads(agent) | {"centers": heads.centers}
        losses = {"dynamics": dynamics} | head_loss(readout, targets, config, window)
        if paired_semantic and observed is not agent:
            # The generated and the observed readout of the same state, judged against
            # the same real targets. Raw readout MSE in Phase 1B was satisfiable by
            # collapsing both sides into a shared low-rank subspace; ground-truth actions
            # and outcomes are what stop that, because a degenerate readout cannot
            # predict them. Averaged, so total head-loss mass is unchanged and no new
            # weight is introduced. The flow arm returns one readout for both and is
            # skipped by identity, as the terminal path already does.
            paired = head_loss(
                heads(observed) | {"centers": heads.centers}, targets, config, window)
            losses = {name: 0.5 * (value + paired[name]) if name in paired else value
                      for name, value in losses.items()}

        terminal = _to(sample_terminal_batch(episodes, sampler, config, step, steps), device)
        terminal_dynamics, terminal_agent, terminal_observed = _terminal_path(
            world,
            terminal,
            rng,
            config,
            world_steps + step,
            score_dynamics=bool(terminal_dynamics_mass),
            return_observed=True,
        )
        if terminal_dynamics_mass:
            losses["dynamics"] = (
                (1.0 - terminal_dynamics_mass) * losses["dynamics"]
                + terminal_dynamics_mass * terminal_dynamics
            )
        terminal_readout = heads(terminal_agent) | {"centers": heads.centers}
        observed_readout = (
            heads(terminal_observed) | {"centers": heads.centers}
            if terminal_observed is not terminal_agent
            else terminal_readout
        )
        terminal_objective = paired_terminal_loss(
            terminal_readout, observed_readout, head_targets(terminal, config)
        )
        losses["continuation"] = (
            (1.0 - config.terminal_loss_mass) * losses["continuation"]
            + config.terminal_loss_mass
            * terminal_objective
        )
        if counterfactual is not None:
            # blended before `_balance`, the way `terminal_dynamics_mass` is: a hand-rolled
            # weighted sum would bypass the running-RMS normalisation and leave the arms
            # unmatched
            extra = _counterfactual_path(world, heads, counterfactual(step), rng, config)
            for name, value in extra.items():
                losses[name] = ((1.0 - counterfactual_mass) * losses[name]
                                + counterfactual_mass * value)
        _update(optimiser, _balance(losses, balance, config), [world, heads], config, step)
        if checkpoint is not None and ((step + 1) % config.checkpoint_every == 0 or step + 1 == steps):
            _checkpoint(checkpoint, config, bundle, balance, streams, step + 1, contract)
    return heads


def _terminal_path(
    world: World,
    batch: Batch,
    rng: torch.Generator,
    config: Config,
    step: int,
    *,
    score_dynamics: bool,
    return_observed: bool = False,
):
    """Evaluate a tail once; optionally admit it to the dynamics stratum."""
    routed = replace(batch, support=None) if score_dynamics else batch
    return transition_loss(
        world,
        routed,
        rng,
        config,
        return_agent=True,
        return_observed=return_observed,
        step=step,
    )


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
    drawn contexts starts RL away from the events that carry the sparse reward.

    S68 is enforced here rather than at each call site. Direct trains exactly
    `direct_rollout` generated states, so an actor imagining past that optimises against
    transitions and heads on inputs they never saw. Ten scripts were each re-deriving the
    cap by hand. It is checked here and not in `imagine`, which is a mechanism the tests
    legitimately exercise at other horizons."""
    if config.transition == "direct" and config.horizon > config.direct_rollout:
        raise ValueError(
            f"horizon {config.horizon} exceeds the {config.direct_rollout} generated "
            "states Direct trains (S68); raise direct_rollout and retrain the world first"
        )
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
    optimizer_step(optimiser, loss, [p for module in modules for p in module.parameters()],
                   learning_rate=config.learning_rate * min(1.0, (step + 1) / config.warmup),
                   grad_clip=config.grad_clip)


def set_phase_mode(bundle: ModelBundle, phase: str):
    if phase == "joint":
        if bundle.encoder._frozen:
            raise ValueError("phase_handoff: cannot silently unfreeze an exported encoder")
        bundle.encoder.train().requires_grad_(True)
        bundle.world.train().requires_grad_(True)
        bundle.world.agent_readout.requires_grad_(False)
    elif phase == "export":
        bundle.encoder.freeze()
        bundle.world.eval()
    elif phase == "bridge":
        if bundle.config.agent is None:
            raise ValueError("phase_gate: bridge requires an M4 recipe")
        bundle.encoder.freeze()
        bundle.world.train().requires_grad_(True)
        bundle.world.agent_readout.requires_grad_(True)
        # TC-15: affine parameters keep gradients, running statistics do not move.
        for module in bundle.world.predictor_projector.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
    elif phase == "actor":
        if bundle.config.agent is None:
            raise ValueError("phase_gate: actor requires an M4 recipe")
        bundle.encoder.freeze()
        bundle.world.eval().requires_grad_(False)
    else:
        raise ValueError(f"phase_gate: unknown LeWM phase {phase}")


def joint_optimizer(bundle: ModelBundle):
    return optimizer([bundle.encoder,bundle.world], bundle.config.joint, exclude_vectors=True)


def learning_rate(config: LeWMConfig, update: int) -> float:
    """Zero-based optimizer update; pausing never changes the original schedule."""
    j = config.joint
    if not 0 <= update < j.steps:
        raise ValueError("update is outside the sealed schedule")
    if update < j.warmup:
        return j.learning_rate * (update + 1) / j.warmup
    progress = (update - j.warmup) / max(1, j.steps - j.warmup - 1)
    return j.min_learning_rate + 0.5 * (j.learning_rate-j.min_learning_rate) * (1 + math.cos(math.pi*progress))


def autocast_context(config: LeWMConfig):
    return (torch.autocast(device_type=config.runtime.device, dtype=torch.bfloat16)
            if config.runtime.precision == "bf16" else nullcontext())


def _seed_dropout_stream(config) -> None:
    """Fix the global streams a dropout-bearing world will draw from.

    The Mamba world has no dropout, so its trajectory never depended on ambient RNG and
    this must not touch it. The source predictor does: without an explicit reset, a Raw and
    a TC run would see different dropout masks purely because preflight ran in a different
    order, and the pair would stop being a single-variable comparison.
    """
    if getattr(config, "family", "lewm_mamba") != "lewm_transformer":
        return
    torch.manual_seed(config.seed + 5003)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + 5003)


def _joint_components(episodes, config, bundle=None):
    _seed_dropout_stream(config)
    bundle = ModelBundle.create(config) if bundle is None else bundle
    if bundle.config != config:
        raise ValueError("bundle recipe differs from training recipe")
    set_phase_mode(bundle, "joint")
    optimiser = joint_optimizer(bundle)
    sampler = JointSampler(episodes, config, torch.Generator().manual_seed(config.seed + 1))
    projection_rng = torch.Generator(device=config.runtime.device).manual_seed(config.seed + 1001)
    identity = {name: tensor_state_digest(getattr(bundle,name).state_dict()) for name in ("encoder","world")}
    return bundle, optimiser, sampler, projection_rng, identity


def initialize_joint(episodes, config, output, *, dataset_contract, gate_report):
    """Persist the exact initial weights and streams as an immutable resume parent."""
    from .gates import require_joint_gates
    require_joint_gates(gate_report,config,dataset_contract)
    bundle, optimiser, sampler, rng, identity = _joint_components(episodes,config)
    path = Path(output)/"step-000000.pt"
    save_lewm_bundle(path,bundle,step=0,dataset_contract=dataset_contract,initial_identity=identity,
                     optimizer=optimiser,sampler=sampler,projection_rng=rng,gate_report=gate_report)
    publish_lewm_latest(path)
    return path, identity


def train_joint(episodes, config: LeWMConfig, output: str | Path, *, dataset_contract: dict,
                gate_report: dict, stop_at: int | None = None, resume: str | Path | None = None,
                bundle: ModelBundle | None = None, screen_report: dict | None = None):
    """Canonical joint LeWM objective; TC-17 keeps counterfactual forks evaluation-only."""
    from .gates import require_joint_gates, require_joint_screen, ComponentGateError

    require_joint_gates(gate_report, config, dataset_contract)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and ((output / "latest.pt").exists() or (output / "metrics.jsonl").exists()):
        raise ValueError("run_output: existing training output requires explicit resume")
    default_end = config.joint.screen_step if config.runtime.purpose == "research" else config.joint.steps
    end = default_end if stop_at is None else stop_at
    if not 0 < end <= config.joint.steps:
        raise ValueError("stop_at pauses within the full declared schedule; it cannot extend the budget")
    if config.runtime.purpose == "research" and end > config.joint.screen_step:
        require_joint_screen(screen_report,config,dataset_contract,resume)
    if screen_report is not None:
        gate_report = gate_report | {"joint_screen":screen_report}
    bundle, optimizer, sampler, projection_rng, initial_identity = _joint_components(episodes,config,bundle)
    begin = 0
    if resume is not None:
        stored = restore_lewm_bundle(resume, bundle, dataset_contract=dataset_contract,
                                     optimizer=optimizer, sampler=sampler, projection_rng=projection_rng)
        begin, initial_identity = stored["step"], stored["initial_identity"]
    if end <= begin:
        raise ValueError("stop_at must be after the restored optimizer step")
    log = output / "metrics.jsonl"
    if log.exists():
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        if rows and rows[-1]["update"] != begin:
            raise ValueError("resume_history: resume at the last logged update or use a fresh output directory")
    if any(int(p.stem.split("-")[-1]) > begin for p in output.glob("step-*.pt")):
        raise ValueError("resume_history: newer immutable snapshots exist; use a fresh output directory")
    regularizer = SIGReg(config.joint.knots, config.joint.projections).to(config.runtime.device)
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    metrics = []
    for update in range(begin, end):
        batch = sampler.sample().to(config.runtime.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(config):
            loss = joint_loss(bundle.encoder, bundle.world, batch.frames, batch.actions,
                              regularizer, projection_rng, config)
        if not bool(torch.isfinite(loss.total)):
            raise RuntimeError(f"joint_objective: nonfinite loss at update {update}; stop this component")
        total, supplement = loss.total, {}
        lr = learning_rate(config, update)
        norm = optimizer_step(optimizer, total, parameters, learning_rate=lr,
                              grad_clip=config.joint.grad_clip, strict=True, zero_grad=False)
        with torch.no_grad():
            z = loss.latent.float()[:, :, 0]
            residual = z-z.mean(1, keepdim=True)
            row = {"update": update+1, "prediction": float(loss.prediction),
                   "regularization": float(loss.regularization), "loss": float(total),
                   "joint_loss": float(loss.total), **supplement,
                   "gradient_norm": float(norm), "learning_rate": lr,
                   "latent_mean": float(z.mean()), "latent_std": float(z.std()),
                   "residual_std": float(residual.std()), "actual_batch": len(batch.episode_ids),
                   "unique_episodes": len(set(batch.episode_ids)),
                   "windows": list(zip(batch.episode_ids, batch.starts.tolist()))}
        metrics.append(row)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if (update+1) % 100 == 0:
            print(json.dumps({"variant":config.variant,"update":update+1,"target":end,
                              "loss":row["loss"],"prediction":row["prediction"]}),flush=True)
        if ((update+1) % config.joint.checkpoint_every == 0
                or update+1 in (config.joint.screen_step, end)):
            snapshot = output / f"step-{update+1:06d}.pt"
            save_lewm_bundle(snapshot, bundle, step=update+1,
                             dataset_contract=dataset_contract, initial_identity=initial_identity,
                             optimizer=optimizer, sampler=sampler, projection_rng=projection_rng,
                             gate_report=gate_report)
            publish_lewm_latest(snapshot)
    return bundle, metrics


def freeze_encoder(bundle: ModelBundle):
    set_phase_mode(bundle, "export")
    return bundle.encoder


# ---- Canonical LeWM M4 ---------------------------------------------------------------

_BRIDGE_GROUPS = ("dynamics", "policy", "reward", "continuation")


def phase_optimizer(modules: list[nn.Module], config: LeWMConfig) -> torch.optim.AdamW:
    """Fresh Phase-2/3 AdamW, retaining upstream parameter no-decay annotations."""
    if config.agent is None:
        raise ValueError("phase_gate: an M4 optimizer requires agent settings")
    return optimizer(modules, config.agent, exclude_vectors=False)


def _phase_lr(config: LeWMConfig, update: int) -> float:
    settings = config.agent
    if settings is None:
        raise ValueError("phase_gate: an M4 schedule requires agent settings")
    return settings.learning_rate * min(1.0, (update + 1) / settings.warmup)


def _phase_balance(losses: dict[str, torch.Tensor], state: dict[str, float], decay: float):
    total = 0.0
    for name, value in losses.items():
        squared = float(value.detach().float().square())
        state[name] = decay * state.get(name, squared) + (1.0 - decay) * squared
        total = total + value / max(state[name] ** 0.5, 1e-8)
    return total


def _outgoing_actions(batch: Batch, n_actions: int) -> torch.Tensor:
    actions = batch.led_to_action[:, 1:]
    if actions.dtype != torch.long or bool(((actions < 0) | (actions >= n_actions)).any()):
        raise ValueError("bridge_data: BOS/padding reached an outgoing action")
    return actions


@torch.no_grad()
def _bridge_initial_state(bundle: ModelBundle, batch: BridgeBatch) -> PredictiveState:
    """Consume each ragged prefix and detach exactly once at the main-window boundary."""
    if bundle.config.family != "lewm_mamba":
        raise ValueError("phase_gate: the canonical M4 bridge is the Mamba thesis architecture")
    rows = []
    for latents, actions in zip(batch.burn_latents, batch.burn_actions):
        state = bundle.world.teacher(latents.unsqueeze(0), actions.unsqueeze(0)).state
        state = bundle.detach_state(state)
        # The main window supplies a new relative clock; absolute archive positions are metadata.
        rows.append(replace(state, step=0))
    return PredictiveState(
        latent=torch.cat([state.latent for state in rows], 0),
        memory=tuple(
            MambaCarry(
                torch.cat([state.memory[layer].conv for state in rows], 0),
                torch.cat([state.memory[layer].ssm for state in rows], 0),
            )
            for layer in range(len(rows[0].memory))
        ),
        history=torch.cat([state.history for state in rows], 0),
        step=0,
    )


def bridge_rollout(bundle: ModelBundle, z: torch.Tensor, actions: torch.Tensor, depth: int,
                   initial: PredictiveState):
    """Teacher path plus exactly ``depth`` recursively generated factual successors."""
    teacher = bundle.world.teacher(z, actions, state=initial)
    blocks = z.shape[1]
    if not 0 < depth < blocks:
        raise ValueError("bridge_depth: generated suffix must leave an observed anchor")
    anchor = blocks - 1 - depth
    prefix = bundle.world.teacher(z[:, :anchor + 1], actions[:, :anchor], state=initial).state
    generated, features = [], []
    for offset in range(depth):
        prefix, feature = bundle.advance(prefix, actions[:, anchor + offset:anchor + offset + 1])
        generated.append(prefix.latent)
        features.append(feature)
    return teacher, torch.cat(generated, 1), torch.cat(features, 1), anchor


def _bridge_dynamics_loss(teacher, generated, z, rows, anchor):
    mask = rows.to(z.dtype).view(-1, 1, 1, 1)

    def mean_squared(prediction, target):
        error = (prediction.float() - target.float()).square() * mask
        return error.sum() / mask.expand_as(error).sum().clamp(min=1.0)

    return (mean_squared(teacher.predicted, z[:, 1:])
            + mean_squared(generated, z[:, anchor + 1:anchor + 1 + generated.shape[1]]))


def _bridge_head_losses(heads: Heads, observed: torch.Tensor, generated: torch.Tensor,
                        targets: dict, config: LeWMConfig, anchor: int):
    blocks = observed.shape[1]
    prefix = torch.arange(blocks, device=observed.device) <= anchor
    suffix = ~prefix
    observed_readout = heads(observed) | {"centers": heads.centers}
    prefix_losses = head_loss(observed_readout, targets, config, prefix)
    observed_suffix = head_loss(observed_readout, targets, config, suffix)
    merged = torch.cat((observed[:, :anchor + 1], generated), 1)
    generated_suffix = head_loss(heads(merged) | {"centers": heads.centers}, targets,
                                 config, suffix)
    return {
        name: 0.5 * prefix_losses[name]
              + 0.25 * observed_suffix[name]
              + 0.25 * generated_suffix[name]
        for name in prefix_losses
    }


def bridge_losses(bundle: ModelBundle, heads: Heads, main: BridgeBatch,
                  terminal: BridgeBatch, depth: int) -> dict[str, torch.Tensor]:
    """TC-16/17 objective with fixed observed/generated and terminal strata."""
    batch = to_head_batch(main)
    initial = _bridge_initial_state(bundle, main)
    actions = _outgoing_actions(batch, bundle.n_actions)
    teacher, generated, generated_features, anchor = bridge_rollout(
        bundle, batch.latents, actions, depth, initial
    )
    targets = head_targets(batch, bundle.config)
    losses = _bridge_head_losses(heads, teacher.features, generated_features, targets,
                                 bundle.config, anchor)
    losses["dynamics"] = _bridge_dynamics_loss(
        teacher, generated, batch.latents, batch.rows("dynamics").to(batch.latents.device), anchor
    )

    support = to_head_batch(terminal)
    support_initial = _bridge_initial_state(bundle, terminal)
    support_teacher, _, support_features, support_anchor = bridge_rollout(
        bundle, support.latents, _outgoing_actions(support, bundle.n_actions), depth, support_initial
    )
    support_targets = head_targets(support, bundle.config)
    observed = heads(support_teacher.features) | {"centers": heads.centers}
    combined = torch.cat((support_teacher.features[:, :support_anchor + 1], support_features), 1)
    recursive = heads(combined) | {"centers": heads.centers}
    losses["continuation"] = (
        0.8 * losses["continuation"]
        + 0.2 * paired_terminal_loss(recursive, observed, support_targets)
    )
    return losses


def train_bridge(cache: EpisodeCorpus, bundle: ModelBundle, parent: dict, output: str | Path,
                 *, parent_path: str | Path, cache_contract: dict, stop_after: str = "h2",
                 resume: str | Path | None = None, gate_report: dict | None = None):
    """Canonical Phase 2: H2 pause, external gate, then H16 on the same world.

    Research budgets can be paused only at declared stage boundaries.  Small plumbing runs use a
    verification recipe with small budgets; command-line step overrides cannot mint research
    checkpoints that look complete.
    """
    from .gates import require_bridge_gate

    config, settings = bundle.config, bundle.config.agent
    if settings is None or not parent.get("capabilities", {}).get("joint_complete"):
        raise ValueError("phase_gate: bridge requires a completed joint M4 parent")
    if stop_after not in ("h2", "h16"):
        raise ValueError("bridge_schedule: stop_after must be h2 or h16")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    set_phase_mode(bundle, "bridge")
    torch.manual_seed(config.seed + 2)
    heads = Heads(config).to(config.runtime.device)
    head_initial_identity = tensor_state_digest(heads.state_dict())
    optimiser = phase_optimizer([bundle.world, heads], config)
    sampler = torch.Generator().manual_seed(config.seed + 31)
    balance: dict[str, float] = {}
    begin = 0
    gate_identity = None
    if resume is not None:
        restored = restore_lewm_bridge(
            resume, bundle, heads, optimizer=optimiser, sampler=sampler,
            parent_path=parent_path, cache_contract=cache_contract, balance=balance,
        )
        if restored.get("head_initial_identity") != head_initial_identity:
            raise ValueError("checkpoint_initialization: bridge heads do not share the seeded start")
        begin = restored["step"]
    elif (output / "metrics.jsonl").exists() or (output / "latest.pt").exists():
        raise ValueError("run_output: existing bridge output requires explicit resume")

    h2, total = settings.h2_steps, settings.h2_steps + settings.h16_steps
    end = h2 if stop_after == "h2" else total
    if begin < h2 < end:
        raise ValueError("bridge_gate: stop at H2; H16 requires the identity-bound H2 gate")
    if begin == h2 and end > begin:
        gate_identity = require_bridge_gate(
            gate_report, checkpoint=Path(resume), config=config, cache_contract=cache_contract,
            stage="h2", minimum_depth=settings.recursive_depth,
        )
    elif begin > h2 and end > begin:
        # The accepted H2 report is bound to the H2 checkpoint, not to a later H16
        # recovery snapshot.  Requiring it against the latter makes an interrupted H16
        # run impossible to resume.  The immutable intermediate carries that already
        # verified identity and full report forward.
        gate_identity = restored.get("gate_identity")
        recorded_report = restored.get("gate")
        if (not isinstance(gate_identity, dict) or not isinstance(recorded_report, dict)
                or recorded_report.get("report_id") != gate_identity.get("report_id")):
            raise ValueError("checkpoint_gate: resumed H16 state lacks accepted H2 lineage")
        if gate_report is not None and gate_report.get("report_id") != gate_identity.get("report_id"):
            raise ValueError("checkpoint_gate: bridge resume changed its accepted H2 report")
        gate_report = recorded_report
    if end <= begin:
        raise ValueError("bridge_schedule: target stage is not after the restored update")
    _check_resume_history(output, begin, "update")

    train = cache.subset(i for i, episode in enumerate(cache) if episode.split == "train")
    if not train:
        raise ValueError("bridge_data: latent cache has no TRAIN episodes")
    bn_start = tensor_state_digest({
        name: tensor for name, tensor in bundle.world.predictor_projector.state_dict().items()
        if "running_" in name or "num_batches_tracked" in name
    })
    started = __import__("time").time()
    for update in range(begin, end):
        depth = settings.recursive_depth if update < h2 else settings.recursive_depth_final
        main = sample_bridge_batch(train, sampler, config, update).to(config.runtime.device)
        terminal = sample_bridge_terminals(train, sampler, config, update).to(config.runtime.device)
        optimiser.zero_grad(set_to_none=True)
        with autocast_context(config):
            losses = bridge_losses(bundle, heads, main, terminal, depth)
            objective = _phase_balance(
                {name: losses[name] for name in _BRIDGE_GROUPS}, balance, settings.rms_decay
            )
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"bridge_objective: nonfinite loss at update {update}")
        parameters = [parameter for group in optimiser.param_groups for parameter in group["params"]]
        norm = optimizer_step(optimiser, objective, parameters,
                              learning_rate=_phase_lr(config, update),
                              grad_clip=settings.grad_clip, strict=True, zero_grad=False)
        row = {
            "update": update + 1, "stage": "h2" if update < h2 else "h16", "depth": depth,
            "frames": main.main.latents.shape[1], "burn_lengths": main.burn_lengths.cpu().tolist(),
            "episode_ids": list(main.episode_ids), "starts": main.starts.cpu().tolist(),
            "loss": float(objective.detach()), "gradient_norm": float(norm),
            "learning_rate": _phase_lr(config, update),
            "seconds": round(__import__("time").time() - started, 2),
            **{name: float(value.detach()) for name, value in losses.items()},
        }
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if (update + 1) % 100 == 0:
            print(json.dumps(row), flush=True)
        if (update + 1) % settings.checkpoint_every == 0 or update + 1 == end:
            current_bn = tensor_state_digest({
                name: tensor for name, tensor in bundle.world.predictor_projector.state_dict().items()
                if "running_" in name or "num_batches_tracked" in name
            })
            if current_bn != bn_start:
                raise RuntimeError("predictor_normalization: Phase 2 changed frozen BN buffers")
            snapshot = output / f"step-{update + 1:06d}.pt"
            save_lewm_bridge(
                snapshot, bundle, heads, step=update + 1, parent_path=parent_path,
                cache_contract=cache_contract, optimizer=optimiser, sampler=sampler,
                balance=balance, bn_identity=bn_start, gate_report=gate_report,
                gate_identity=gate_identity, head_initial_identity=head_initial_identity,
            )
            publish_lewm_latest(snapshot)
    return bundle, heads


def _check_resume_history(output: Path, begin: int, key: str) -> None:
    log = output / "metrics.jsonl"
    if log.exists():
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        if rows and rows[-1][key] != begin:
            raise ValueError("resume_history: resume at the last logged update or use a fresh output")
    if any(int(path.stem.split("-")[-1]) > begin for path in output.glob("step-*.pt")):
        raise ValueError("resume_history: newer immutable snapshots exist; use a fresh output")


def train_actor_lewm(cache: EpisodeCorpus, bundle: ModelBundle, heads: Heads, bridge: dict,
                     output: str | Path, *, bridge_path: str | Path, cache_contract: dict,
                     bridge_gate: dict, stop_after: str = "screen", resume: str | Path | None = None,
                     actor_gate: dict | None = None):
    """Canonical Phase 3 with immutable BC prior and a real 500-update stop."""
    from .gates import require_actor_gate, require_bridge_gate

    config, settings = bundle.config, bundle.config.agent
    if settings is None:
        raise ValueError("phase_gate: actor requires an M4 recipe")
    if settings.actor_batch != settings.batch:
        raise ValueError("actor_data: canonical Phase 3 and bridge batches are both 16")
    if (bridge.get("cache") != cache_contract
            or bridge.get("step") != settings.h2_steps + settings.h16_steps):
        raise ValueError("checkpoint_parent: actor requires the completed bridge and its exact cache")
    accepted_bridge_gate = require_bridge_gate(
        bridge_gate, checkpoint=Path(bridge_path), config=config, cache_contract=cache_contract,
        stage="h16", minimum_depth=settings.horizon,
    )
    if stop_after not in ("screen", "budget"):
        raise ValueError("actor_schedule: stop_after must be screen or budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    set_phase_mode(bundle, "actor")
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    for parameter in heads.actor_parameters():
        parameter.requires_grad_(True)
    prior = copy.deepcopy(heads).eval().requires_grad_(False)
    optimiser = phase_optimizer([heads], config)
    sampler = torch.Generator().manual_seed(config.seed + 3)
    policy_rng = torch.Generator(device=config.runtime.device).manual_seed(config.seed + 2**20)
    balance: dict[str, float] = {}
    begin = 0
    actor_gate_identity = None
    if resume is not None:
        restored = restore_lewm_actor(
            resume, bundle, heads, prior, optimizer=optimiser, sampler=sampler,
            policy_rng=policy_rng, bridge_path=bridge_path, cache_contract=cache_contract,
            balance=balance,
        )
        begin = restored["step"]
    elif (output / "metrics.jsonl").exists() or (output / "latest.pt").exists():
        raise ValueError("run_output: existing actor output requires explicit resume")
    screen, total = settings.actor_screen_steps, settings.actor_steps
    end = screen if stop_after == "screen" else total
    if begin < screen < end:
        raise ValueError("actor_gate: stop at 500 updates before the full actor budget")
    if begin == screen and end > begin:
        actor_gate_identity = require_actor_gate(
            actor_gate, checkpoint=Path(resume), config=config, cache_contract=cache_contract
        )
    elif begin > screen and end > begin:
        actor_gate_identity = restored.get("actor_gate_identity")
        recorded_report = restored.get("actor_gate")
        if (not isinstance(actor_gate_identity, dict) or not isinstance(recorded_report, dict)
                or recorded_report.get("report_id") != actor_gate_identity.get("report_id")):
            raise ValueError("checkpoint_gate: resumed actor lacks accepted screen lineage")
        if actor_gate is not None and actor_gate.get("report_id") != actor_gate_identity.get("report_id"):
            raise ValueError("checkpoint_gate: actor resume changed its accepted screen report")
        actor_gate = recorded_report
    if resume is not None:
        recorded_bridge = restored.get("bridge_gate", {})
        if recorded_bridge.get("report_id") != accepted_bridge_gate["report_id"]:
            raise ValueError("checkpoint_gate: actor resume changed its accepted H16 report")
    if end <= begin:
        raise ValueError("actor_schedule: target stage is not after the restored update")
    _check_resume_history(output, begin, "update")

    train = cache.subset(i for i, episode in enumerate(cache) if episode.split == "train")
    bundle.capabilities = {
        "readout_trained": True,
        "trained_recursive_depth": settings.recursive_depth_final,
        "validated_recursive_depth": settings.recursive_depth_final,
        "m4_authorized": False,
    }
    frozen = {
        "encoder": tensor_state_digest(bundle.encoder.state_dict()),
        "world": tensor_state_digest(bundle.world.state_dict()),
        "prior": tensor_state_digest(prior.state_dict()),
        "model_heads": tensor_state_digest({
            **{f"model_body.{k}": v for k, v in heads.model_body.state_dict().items()},
            **{f"reward.{k}": v for k, v in heads.reward.state_dict().items()},
            **{f"continuation.{k}": v for k, v in heads.continuation.state_dict().items()},
        }),
    }
    started = __import__("time").time()
    for update in range(begin, end):
        batch = sample_bridge_batch(train, sampler, config, update).to(config.runtime.device)
        main = batch.main
        with torch.no_grad():
            initial = _bridge_initial_state(bundle, batch)
            observed = bundle.world.teacher(
                main.latents, _outgoing_actions(main, bundle.n_actions), state=initial
            )
        trajectory = imagine(bundle, heads, observed.state, observed.features[:, -1:],
                             None, policy_rng, config)
        returns = lambda_returns(trajectory, config)
        with torch.no_grad():
            reference = prior(trajectory.agent[:, :-1])["policy"][:, :, 0]
        losses = {
            "actor": actor_loss(trajectory, returns, reference, config),
            "critic": critic_loss(heads(trajectory.agent[:, :-1])["value"], returns, heads.centers),
        }
        objective = _phase_balance(losses, balance, settings.rms_decay)
        parameters = [parameter for group in optimiser.param_groups for parameter in group["params"]]
        norm = optimizer_step(optimiser, objective, parameters,
                              learning_rate=_phase_lr(config, update),
                              grad_clip=settings.grad_clip, strict=True)
        row = {
            "update": update + 1, "horizon": settings.horizon,
            "actor": float(losses["actor"].detach()), "critic": float(losses["critic"].detach()),
            "loss": float(objective.detach()), "gradient_norm": float(norm),
            "learning_rate": _phase_lr(config, update),
            "imagined_reward": float(trajectory.reward.mean()),
            "imagined_continuation": float(trajectory.continuation.mean()),
            "seconds": round(__import__("time").time() - started, 2),
        }
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if (update + 1) % 100 == 0:
            print(json.dumps(row), flush=True)
        if (update + 1) % settings.checkpoint_every == 0 or update + 1 == end:
            now = {
                "encoder": tensor_state_digest(bundle.encoder.state_dict()),
                "world": tensor_state_digest(bundle.world.state_dict()),
                "prior": tensor_state_digest(prior.state_dict()),
                "model_heads": tensor_state_digest({
                    **{f"model_body.{k}": v for k, v in heads.model_body.state_dict().items()},
                    **{f"reward.{k}": v for k, v in heads.reward.state_dict().items()},
                    **{f"continuation.{k}": v for k, v in heads.continuation.state_dict().items()},
                }),
            }
            if now != frozen:
                raise RuntimeError("actor_freeze: Phase 3 changed its world, prior, encoder or model heads")
            snapshot = output / f"step-{update + 1:06d}.pt"
            save_lewm_actor(
                snapshot, bundle, heads, prior, step=update + 1, bridge_path=bridge_path,
                cache_contract=cache_contract, optimizer=optimiser, sampler=sampler,
                policy_rng=policy_rng, balance=balance, frozen_identity=frozen,
                bridge_gate=bridge_gate, actor_gate=actor_gate,
                actor_gate_identity=actor_gate_identity,
            )
            publish_lewm_latest(snapshot)
    return heads, prior
