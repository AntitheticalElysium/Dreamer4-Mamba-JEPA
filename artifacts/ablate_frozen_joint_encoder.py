"""H2: does letting the MAE encoder adapt during Phase 2 rescue action consequence?

Two arms differing in exactly one fact -- whether the encoder receives gradients.
Both start from the same Phase-1A encoder, the same Phase-1B Direct world, and
identically initialised fresh Phase-2 heads.

Nothing in `d4mj/` is modified. The production samplers are called unchanged; a
diagnostic-local wrapper around `d4mj.data._window` observes the `(episode, start,
length)` they already selected, so raw frames for the same window can be fetched
and encoded online. Episodes are resolved by canonical content fingerprint, never
by position: latent caching replaces observations with latents and leaves every
trajectory field below untouched, so those fields identify an episode across both
corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

import d4mj.data as data_module
from artifacts.phase1b_diagnostic_common import cached_train, file_digest, implementation_digests
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import Batch, Episode, patchify, sample_batch, sample_terminal_batch
from d4mj.representation import Decoder, Encoder, pack, reconstruction_loss
from d4mj.state import WorldState
from d4mj.train import _balance, _generators, _to, optimizer
from d4mj.transition import World, advance, commit_inputs

VERSION = "frozen-joint-encoder-ablation-v1"

FINGERPRINT_FIELDS = (
    "actions_taken",
    "rewards",
    "terminated",
    "truncated",
    "events",
)
FINGERPRINT_FLAGS = (
    "uniform_eligible",
    "bc_eligible",
    "epsilon",
    "split",
    "terminal_cause",
)


def _tensor_bytes(value: Tensor | None) -> bytes:
    """Canonical: dtype and shape then contiguous CPU bytes, not a serialisation."""
    if value is None:
        return b"none"
    tensor = value.detach().cpu().contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    return header + tensor.numpy().tobytes()


def fingerprint(episode: Episode) -> str:
    """Identity from the trajectory fields latent caching carries through unchanged.

    `observations`, `latents` and `latent_digest` are deliberately excluded: those
    are exactly what differs between the cached and raw corpora.
    """
    digest = hashlib.sha256()
    digest.update(f"len={len(episode)}|".encode())
    for field in FINGERPRINT_FIELDS:
        digest.update(field.encode() + b"=" + _tensor_bytes(getattr(episode, field)))
    for field in FINGERPRINT_FLAGS:
        digest.update(f"{field}={getattr(episode, field)!r}|".encode())
    if episode.episode_id is not None:
        digest.update(f"episode_id={episode.episode_id}|".encode())
    return digest.hexdigest()


def build_resolver(cached: list[Episode], raw: list[Episode]) -> dict[str, Episode]:
    """One raw episode per cached episode, or refuse to start.

    Resolved globally rather than lazily, so an ambiguous or missing episode is a
    startup failure instead of a mid-training surprise on a rare draw.
    """
    table: dict[str, list[Episode]] = {}
    for episode in raw:
        table.setdefault(fingerprint(episode), []).append(episode)
    resolver, missing, ambiguous = {}, [], []
    for episode in cached:
        key = fingerprint(episode)
        candidates = table.get(key, [])
        if not candidates:
            missing.append(episode.episode_id or key[:16])
        elif len(candidates) > 1:
            ambiguous.append(episode.episode_id or key[:16])
        else:
            resolver[key] = candidates[0]
    if missing or ambiguous:
        raise ValueError(
            f"raw resolution failed: {len(missing)} missing, {len(ambiguous)} ambiguous; "
            f"first missing={missing[:3]} first ambiguous={ambiguous[:3]}"
        )
    return resolver


def assert_pair(cached: Episode, raw: Episode) -> None:
    """Re-check the resolved pair on every draw, not only at startup."""
    for field in FINGERPRINT_FIELDS:
        left, right = getattr(cached, field), getattr(raw, field)
        if (left is None) != (right is None):
            raise AssertionError(f"resolved pair disagrees on {field} presence")
        if left is not None and not torch.equal(left, right):
            raise AssertionError(f"resolved pair disagrees on {field}")
    for field in FINGERPRINT_FLAGS:
        if getattr(cached, field) != getattr(raw, field):
            raise AssertionError(f"resolved pair disagrees on {field}")
    if cached.episode_id is not None and cached.episode_id != raw.episode_id:
        raise AssertionError("content-resolved episode has a different episode_id")
    if raw.observations is None:
        raise AssertionError("resolved raw episode carries no observations")


@contextmanager
def record_windows():
    """Observe what the production sampler already chose. No RNG, no decisions."""
    original, seen = data_module._window, []
    def capture(episode, start, length, config):
        seen.append((episode, int(start), int(length)))
        return original(episode, start, length, config)
    data_module._window = capture
    try:
        yield seen
    finally:
        data_module._window = original


def online_latents(
    selections: list[tuple[Episode, int, int]],
    resolver: dict[str, Episode],
    encoder: Encoder,
    config: Config,
) -> Tensor:
    """Encode each selected window online, with causal context, keeping the target.

    The encoder is causal and bounded by `W`, so a window starting mid-episode
    needs its preceding frames or `z_t` is not the cached `Z*`. Only the final
    `length` latents are retained, which is what the cached window holds.
    """
    rows, targets = [], []
    for episode, start, length in selections:
        raw = resolver[fingerprint(episode)]
        assert_pair(episode, raw)
        context = min(config.burn_in, start)
        frames = raw.observations[start - context : start + length]
        patches = patchify(frames[None], config.patch).to(config.device)
        z, _, _ = encoder(patches, None, p_mask=0.0, offset=start - context)
        rows.append(pack(z, config)[0, context:])
        targets.append(patches[0, context:])
    return torch.stack(rows), torch.stack(targets)


def paired_batch(
    episodes,
    sampler: torch.Generator,
    config: Config,
    step: int,
    total: int,
    resolver: dict[str, Episode],
    encoder: Encoder,
    *,
    terminal: bool,
    mixture: bool = False,
) -> Batch:
    """A production cached batch with only its latents replaced by online ones."""
    with record_windows() as selections:
        if terminal:
            batch = sample_terminal_batch(episodes, sampler, config, step, total)
        else:
            batch = sample_batch(episodes, sampler, config, step, total, mixture=mixture)
    expected = config.terminal_batch if terminal else config.batch
    if len(selections) != expected:
        raise AssertionError(f"captured {len(selections)} selections, expected {expected}")
    latents, patches = online_latents(selections, resolver, encoder, config)
    cached = _to(batch, config.device).latents
    if latents.shape != cached.shape:
        raise AssertionError(f"online latents {tuple(latents.shape)} != cached {tuple(cached.shape)}")
    return replace(_to(batch, config.device), latents=latents), cached, patches


def direct_loss_detached(world: World, batch: Batch, rng: torch.Generator, config: Config, *, step: int):
    """`transition.transition_loss` for Direct, with the regression targets detached.

    Production never needed this: cached `Z*` carried no gradient. Once the encoder
    is online the target would otherwise be pulled toward the prediction, which is
    not the experiment. Inputs and history stay differentiable.
    """
    committed, conditioning = commit_inputs(batch.latents, rng, config)
    features, agent, memory = world(None, batch.led_to_action, committed, conditioning)
    taken = batch.led_to_action[:, 1:]
    predicted = world.predict(features[:, :-1], taken)
    teacher = (predicted - batch.latents[:, 1:].detach()).pow(2).mean(dim=(1, 2, 3))

    length = batch.latents.shape[1]
    mask = batch.rows("dynamics").to(teacher.device).float()
    if length < 3:
        return (teacher * mask).sum() / mask.sum().clamp(min=1.0), agent, agent

    prefix, _, memory = world(
        None, batch.led_to_action[:, :-2], committed[:, :-2], conditioning[:, :-2]
    )
    state = WorldState(batch.latents[:, -3:-2], memory, length - 2, prefix[:, -1:])
    first, rolled = advance(world, state, batch.led_to_action[:, -2:-1], rng, config)
    second, rolled_again = advance(world, first, batch.led_to_action[:, -1:], rng, config)
    rollout = (first.latent - batch.latents[:, -2:-1].detach()).pow(2).mean(dim=(1, 2, 3))
    rollout = rollout + (second.latent - batch.latents[:, -1:].detach()).pow(2).mean(dim=(1, 2, 3))
    readout = torch.cat([agent[:, :-2], rolled, rolled_again], dim=1)
    per_row = teacher + rollout / 2
    return (per_row * mask).sum() / mask.sum().clamp(min=1.0), readout, agent


def scaled_update(optimiser, modules, config: Config, step: int) -> dict[str, float]:
    """Warmup applied to each group's own base LR.

    `train._update` writes `config.learning_rate` into every group, which would
    silently raise the encoder from 6e-6 to 1e-4 -- a 17x encoder-LR ablation
    wearing this experiment's name.
    """
    factor = min(1.0, (step + 1) / config.warmup)
    realised = {}
    for group in optimiser.param_groups:
        group["lr"] = group["base_lr"] * factor
        realised[group["name"]] = group["lr"]
    torch.nn.utils.clip_grad_norm_(
        [p for module in modules for p in module.parameters()], config.grad_clip
    )
    optimiser.step()
    return realised


def build_optimizer(world, heads, encoder, config: Config, encoder_lr: float, joint: bool):
    """World and heads at the production LR; the encoder at its own, when joint.

    Mirrors `train.optimizer`'s weight-decay split so the only difference from
    production is the encoder group and its base LR.
    """
    members = [("world", world, config.learning_rate), ("heads", heads, config.learning_rate)]
    if joint:
        members.append(("encoder", encoder, encoder_lr))
    groups = []
    for name, module, lr in members:
        decayed, exempt = [], []
        for parameter in module.parameters():
            if parameter.requires_grad:
                (exempt if getattr(parameter, "_no_weight_decay", False) else decayed).append(parameter)
        if decayed:
            groups.append({"name": name, "params": decayed, "weight_decay": config.weight_decay, "base_lr": lr, "lr": lr})
        if exempt:
            groups.append({"name": f"{name}:no_decay", "params": exempt, "weight_decay": 0.0, "base_lr": lr, "lr": lr})
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def phase2_step(
    episodes,
    resolver,
    encoder: Encoder,
    decoder: Decoder,
    perceptual,
    world: World,
    heads,
    optimiser,
    balances: dict,
    sampler: torch.Generator,
    rng: torch.Generator,
    config: Config,
    step: int,
    total: int,
    world_steps: int,
    *,
    joint: bool,
) -> dict:
    """One matched update. Two backwards, so the MAE and Phase-2 activation graphs
    never coexist; the encoder accumulates gradient from both before one step."""
    from d4mj.agent import head_loss, head_targets, paired_terminal_loss

    optimiser.zero_grad(set_to_none=True)
    report: dict[str, float] = {}

    main, cached_main, main_patches = paired_batch(
        episodes, sampler, config, step, total, resolver, encoder, terminal=False, mixture=True
    )
    terminal, _, _ = paired_batch(
        episodes, sampler, config, step, total, resolver, encoder, terminal=True
    )

    if joint:
        z, _, masked = encoder(main_patches, p_mask=config.mae_p_max, rng=rng)
        predicted, _ = decoder(z)
        mae = reconstruction_loss(predicted, main_patches, masked, main.scored, perceptual, config)
        mae_loss = _balance(mae, balances["mae"], config, {"lpips": config.lpips_weight})
        mae_loss.backward()
        report["mae"] = float(mae_loss.detach())

    dynamics, agent, _ = direct_loss_detached(world, main, rng, config, step=world_steps + step)
    readout = heads(agent) | {"centers": heads.centers}
    losses = {"dynamics": dynamics} | head_loss(readout, head_targets(main, config), config)

    _, terminal_agent, terminal_observed = direct_loss_detached(
        world, terminal, rng, config, step=world_steps + step
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
        + config.terminal_loss_mass * terminal_objective
    )
    phase2 = _balance(losses, balances["phase2"], config)
    phase2.backward()
    report["phase2"] = float(phase2.detach())
    report["dynamics"] = float(dynamics.detach())
    report["terminal"] = float(terminal_objective.detach())

    modules = [world, heads] + ([encoder] if joint else [])
    report["lr"] = scaled_update(optimiser, modules, config, step)
    report["cache_parity"] = float((main.latents.detach() - cached_main).abs().max())
    return report


def build_arm(phase1a: Path, phase1b: Path, config: Config, base: Config, *, joint: bool, encoder_lr: float):
    """Same encoder checkpoint, same Phase-1B world, identically seeded fresh heads."""
    import lpips
    from d4mj.agent import Heads

    encoder, decoder = Encoder(base).to(base.device), Decoder(base).to(base.device)
    load(phase1a, base, part0=encoder, part1=decoder)
    world = World(config).to(config.device)
    load(phase1b, config, part0=world)
    torch.manual_seed(config.seed + 2)
    heads = Heads(config).to(config.device)

    encoder.requires_grad_(joint)
    decoder.requires_grad_(joint)
    encoder.train(joint)
    decoder.train(joint)
    world.train()
    heads.train()
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(base.device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    optimiser = build_optimizer(world, heads, encoder, config, encoder_lr, joint)
    return encoder, decoder, perceptual, world, heads, optimiser


def _encoder_grad(encoder: Encoder) -> float:
    total = sum(float(p.grad.pow(2).sum()) for p in encoder.parameters() if p.grad is not None)
    return total ** 0.5


def preflight(episodes, resolver, phase1a: Path, phase1b: Path, config: Config, base: Config, encoder_lr: float) -> dict:
    """The four numbers, measured, before any 20k run."""
    from d4mj.agent import head_loss, head_targets, paired_terminal_loss
    import time

    out: dict = {}
    frozen = build_arm(phase1a, phase1b, config, base, joint=False, encoder_lr=encoder_lr)
    fenc = frozen[0]
    sampler, rng = _generators(config, 2)
    with torch.no_grad():
        rows = []
        for step in (0, 1, 3, 7):
            for terminal in (False, True):
                b, cached, _ = paired_batch(
                    episodes, sampler, config, step, 20_000, resolver, fenc,
                    terminal=terminal, mixture=not terminal,
                )
                rows.append({
                    "step": step, "terminal": terminal, "length": int(b.latents.shape[1]),
                    "max_abs": float((b.latents - cached).abs().max()),
                    "rms": float((b.latents - cached).pow(2).mean().sqrt()),
                    "cached_rms": float(cached.pow(2).mean().sqrt()),
                })
    out["latent_parity"] = rows

    joint = build_arm(phase1a, phase1b, config, base, joint=True, encoder_lr=encoder_lr)
    jenc, jdec, jperc, jworld, jheads, jopt = joint
    sampler, rng = _generators(config, 2)
    main, _, main_patches = paired_batch(
        episodes, sampler, config, 0, 20_000, resolver, jenc, terminal=False, mixture=True
    )
    terminal_batch, _, _ = paired_batch(
        episodes, sampler, config, 0, 20_000, resolver, jenc, terminal=True
    )
    balances = {"mae": {}, "phase2": {}}

    jopt.zero_grad(set_to_none=True)
    z, _, masked = jenc(main_patches, p_mask=config.mae_p_max, rng=rng)
    predicted, _ = jdec(z)
    mae = reconstruction_loss(predicted, main_patches, masked, main.scored, jperc, config)
    _balance(mae, balances["mae"], config, {"lpips": config.lpips_weight}).backward()
    out["encoder_grad_mae"] = _encoder_grad(jenc)

    jopt.zero_grad(set_to_none=True)
    dynamics, agent, _ = direct_loss_detached(jworld, main, rng, config, step=0)
    readout = jheads(agent) | {"centers": jheads.centers}
    losses = {"dynamics": dynamics} | head_loss(readout, head_targets(main, config), config)
    _balance(losses, dict(balances["phase2"]), config).backward()
    out["encoder_grad_main_phase2"] = _encoder_grad(jenc)

    jopt.zero_grad(set_to_none=True)
    _, t_agent, t_obs = direct_loss_detached(jworld, terminal_batch, rng, config, step=0)
    t_read = jheads(t_agent) | {"centers": jheads.centers}
    o_read = jheads(t_obs) | {"centers": jheads.centers} if t_obs is not t_agent else t_read
    paired_terminal_loss(t_read, o_read, head_targets(terminal_batch, config)).backward()
    out["encoder_grad_terminal"] = _encoder_grad(jenc)

    out["realised_lr"] = {}
    for step in (0, config.warmup - 1, config.warmup * 2):
        factor = min(1.0, (step + 1) / config.warmup)
        out["realised_lr"][str(step)] = {
            g["name"]: g["base_lr"] * factor for g in jopt.param_groups
        }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sampler, rng = _generators(config, 2)
    balances = {"mae": {}, "phase2": {}}
    for step in range(3):
        phase2_step(episodes, resolver, jenc, jdec, jperc, jworld, jheads, jopt,
                    balances, sampler, rng, config, step, 20_000, 20_000, joint=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    N = 8
    for step in range(3, 3 + N):
        last = phase2_step(episodes, resolver, jenc, jdec, jperc, jworld, jheads, jopt,
                           balances, sampler, rng, config, step, 20_000, 20_000, joint=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out["seconds_per_step_joint"] = (time.time() - start) / N
    out["peak_vram_gib"] = (
        torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    )
    out["step_report"] = last
    return out
