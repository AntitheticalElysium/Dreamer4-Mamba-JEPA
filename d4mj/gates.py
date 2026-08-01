import torch

from .backbone import AGENT, Backbone, Layout, space_mask
from .config import Config
from .data import Episode, sample_batch
from .state import WorldState
from .transition import advance, initial


def scan_step_parity(config: Config, tolerance: float = 1e-3) -> None:
    """One batched scan must equal the same frames stepped one at a time carrying
    memory. Training runs the scan and imagination runs the steps, so a divergence
    here is a model that is correct in every loss and wrong in every rollout.

    The tolerance is relative and generous because Mamba's chunked scan and its
    sequential step are different kernels: measured drift is 2.5e-5 relative for
    Mamba and 2.5e-7 for attention, and neither grows with sequence length. A
    misalignment -- a stale offset, a dropped state -- is order one, not order
    1e-5, so the gap between the two is what the gate actually tests.

    The sequence runs past `dynamics_context` on purpose: at a shorter length the
    windowed branch of `_decode_mask` never executes, and a regression in the
    window arithmetic would pass everything.
    """
    device = _device(config)
    layout = Layout.dynamics(config)
    backbone = Backbone(
        config, layout, "dynamics", config.d_model, config.n_heads, config.depth,
        config.dynamics_context,
    )
    backbone = backbone.to(device).eval()

    length = config.dynamics_context + 4
    frames = torch.randn(2, length, layout.size, config.d_model, device=device)
    with torch.no_grad():
        scanned, scanned_memory = backbone(frames)
        stepped, memory = [], None
        for index in range(frames.shape[1]):
            out, memory = backbone(frames[:, index : index + 1], memory, index)
            stepped.append(out)

    drift = (torch.cat(stepped, dim=1) - scanned).abs().max() / scanned.abs().max()
    assert drift < tolerance, f"scan/step drift {drift:.2e} exceeds {tolerance:.0e}"
    assert len(memory) == len(scanned_memory) == config.depth // config.time_every
    if config.time_mixer == "attention":
        assert all(pair[0].shape[2] <= config.dynamics_context for pair in scanned_memory), (
            "the scan ignores the window the cache enforces"
        )

    _world_parity(config, device, tolerance)
    _encoder_parity(config, device, tolerance)


def _world_parity(config: Config, device: str, tolerance: float) -> None:
    """The teacher-forced pass and the runtime path must agree. Only training runs
    the first and only imagination runs the second, so nothing else compares them --
    and a Direct arm whose loss never saw an observation passed every other gate."""
    from .transition import World, commit_inputs

    world = World(config).to(device).eval()
    latents = torch.randn(2, 4, config.n_spatial, config.d_spatial, device=device).tanh()
    actions = torch.randint(config.n_actions, (2, 4), device=device)

    with torch.no_grad():
        committed, conditioning = commit_inputs(
            latents, torch.Generator(device=device).manual_seed(1), config
        )
        forced, _, _ = world(None, actions, committed, conditioning)
        stepped, memory = [], None
        for index in range(4):
            block = slice(index, index + 1)
            features, _, memory = world(
                memory, actions[:, block], committed[:, block], conditioning[:, block], index
            )
            stepped.append(features)

    drift = (torch.cat(stepped, dim=1) - forced).abs().max() / forced.abs().max()
    assert drift < tolerance, f"teacher-forced vs stepped world drift {drift:.2e}"
    if config.time_mixer == "attention":
        assert all(
            pair[0].shape[2] <= config.dynamics_context for pair in memory
        ), "dynamics cache exceeds the declared context"


def _encoder_parity(config: Config, device: str, tolerance: float) -> None:
    """Batched scan, frame-by-frame recurrence and the cached Z* must agree, and
    nothing beyond the receptive field may reach a latent.

    `window` bounds each time layer's state; influence still travels one window per
    time layer, so the reach is their product. The state bound is what makes the
    cache and the deployed rollout produce the same latent for the same frame."""

    from .representation import Encoder

    encoder = Encoder(config).to(device).eval()
    reach = config.receptive_field
    frames = torch.rand(1, reach + 6, config.n_patches, config.patch_dim, device=device)

    with torch.no_grad():
        scanned, memory, _ = encoder(frames)
        stepped, carried = [], None
        for index in range(frames.shape[1]):
            z, carried, _ = encoder(frames[:, index : index + 1], carried, offset=index)
            stepped.append(z)
        distant = frames.clone()
        distant[:, : frames.shape[1] - reach] = torch.rand_like(distant[:, :-reach])
        bounded, _, _ = encoder(distant)

    drift = (torch.cat(stepped, dim=1) - scanned).abs().max() / scanned.abs().max()
    assert drift < tolerance, f"encoder scan/step drift {drift:.2e}"
    assert all(pair[0].shape[2] <= config.window for pair in memory), "encoder state exceeds the window"
    influence = (bounded[:, -1] - scanned[:, -1]).abs().max()
    assert influence == 0, f"a frame beyond the receptive field moved z by {influence:.2e}"

    inside = frames.clone()
    inside[:, -2] = torch.rand_like(inside[:, -2])
    with torch.no_grad():
        used, _, _ = encoder(inside)
    assert not torch.equal(used[:, -1], scanned[:, -1]), "the encoder ignores its history"


def alignment(config: Config) -> None:
    """The temporal contract, asserted on episodes whose every value identifies its
    own index. Any shift, any window crossing a boundary, and any start-of-episode
    action leaking into a mid-episode window shows up as a mismatch rather than as
    a plausible-looking number months later.

"""
    episodes = [_probe(index, config) for index in range(4)]
    rng = torch.Generator().manual_seed(config.seed)
    batch = sample_batch(episodes, rng, config)

    for row in range(batch.led_to_action.shape[0]):
        start = _recover_start(batch, row, config)
        steps = torch.arange(start, start + batch.led_to_action.shape[1])
        exists = steps > 0
        source = (steps - 1).clamp(min=0)

        assert torch.equal(batch.valid[row], exists)
        assert torch.equal(batch.reward[row][exists], source[exists].float())
        assert torch.equal(batch.led_to_action[row][exists], source[exists] % config.n_actions)
        assert (batch.led_to_action[row][~exists] == config.n_actions).all()
        assert not batch.valid[row][~exists].any()

    assert batch.patches.shape[1:] == (
        config.burn_in + config.sequence,
        config.n_patches,
        config.patch_dim,
    )
    _observation_dependence(config)


def _observation_dependence(config: Config) -> None:
    """The *prediction* must move when the context moves, with the action and the
    conditioning held fixed.

    Comparing losses across two batches is not this test: changing the latents also
    changes the target, so a model-free loss returning `latents.pow(2).mean()`
    passes it. Only the prediction path is evidence that observations reach the
    predictor at all. The loss is separately checked to be finite, since a division
    by a vanishing signal level surfaces as a clean-looking curve of NaNs.
    """
    from .transition import World, commit_inputs

    device = _device(config)
    world = World(config).to(device).eval()
    action = torch.zeros(2, 3, dtype=torch.long, device=device)
    shape = (2, 3, config.n_spatial, config.d_spatial)

    from .data import Batch
    from .transition import transition_loss

    probe = Batch(
        led_to_action=torch.zeros(2, 4, dtype=torch.long, device=device),
        reward=torch.zeros(2, 4, device=device),
        terminated=torch.zeros(2, 4, dtype=torch.bool, device=device),
        truncated=torch.zeros(2, 4, dtype=torch.bool, device=device),
        valid=torch.ones(2, 4, dtype=torch.bool, device=device),
        burn_in=0,
        latents=torch.randn(2, 4, config.n_spatial, config.d_spatial, device=device).tanh(),
    )
    with torch.no_grad():
        loss = transition_loss(world, probe, torch.Generator(device=device).manual_seed(5), config)
    assert torch.isfinite(loss), f"transition loss is {loss}"

    predictions = []
    for seed in (1, 2):
        context = torch.randn(
            shape, generator=torch.Generator(device=device).manual_seed(seed), device=device
        ).tanh()
        with torch.no_grad():
            committed, conditioning = commit_inputs(
                context, torch.Generator(device=device).manual_seed(4), config
            )
            features, _, _ = world(None, action, committed, conditioning)
            predictions.append(world.predict(features, action))
    assert not torch.equal(*predictions), "the prediction ignores its context"


def _conditioning_coverage(config: Config) -> None:
    """Every row of the conditioning table must be reachable by training.

    An unreachable row is the one defect the register spends a whole decision on
    (S10): the pinned source sizes its signal table `k_max + 1` and labels
    uncorrupted context with a row its own sampler can never draw.
    """
    from .data import Batch
    from .transition import World, transition_loss

    device = _device(config)
    world = World(config).to(device)
    tables = (
        [world.signal_embed, world.step_embed]
        if config.transition == "flow"
        else [world.condition_embed]
    )
    touched = [torch.zeros(t.num_embeddings, device=device) for t in tables]

    for trial in range(40):
        generator = torch.Generator(device=device).manual_seed(trial)
        batch = Batch(
            led_to_action=torch.zeros(4, 6, dtype=torch.long, device=device),
            reward=torch.zeros(4, 6, device=device),
            terminated=torch.zeros(4, 6, dtype=torch.bool, device=device),
            truncated=torch.zeros(4, 6, dtype=torch.bool, device=device),
            valid=torch.ones(4, 6, dtype=torch.bool, device=device),
            burn_in=0,
            latents=torch.randn(4, 6, config.n_spatial, config.d_spatial, device=device).tanh(),
        )
        world.zero_grad()
        transition_loss(world, batch, generator, config).backward()
        for table, seen in zip(tables, touched):
            if table.weight.grad is not None:
                seen += table.weight.grad.abs().sum(-1)

    for table, seen in zip(tables, touched):
        missing = int((seen == 0).sum())
        assert missing == 0, f"{missing} of {table.num_embeddings} conditioning rows never train"


def reset_parity(config: Config) -> None:
    """A fresh state must not remember a previous episode. Constructing rather than
    clearing makes this true by type, so the gate exists to catch a caller that
    threads memory across a boundary anyway."""
    device = _device(config)
    world = _world(config)
    rng = torch.Generator(device=device).manual_seed(0)
    latent = torch.randn(2, 1, config.n_spatial, config.d_spatial, device=device).tanh()
    action = torch.zeros(2, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        start, _ = initial(world, latent, action, rng, config)
        carried, _ = advance(world, start, action, rng, config)
        follow, _ = advance(world, carried, action, torch.Generator(device=device).manual_seed(1), config)
        blank, _ = initial(world, carried.latent, action, rng, config)
        fresh, _ = advance(world, blank, action, torch.Generator(device=device).manual_seed(1), config)
    assert not torch.allclose(follow.latent, fresh.latent, atol=1e-6), "history had no effect"


def firewall(config: Config) -> None:
    """Agent state reaches the world only through the chosen action. Asserted in
    both directions, because a mask that blocks only one is still broken and still
    passes a one-directional test."""
    layout = Layout.dynamics(config)
    kinds = layout.kinds
    agent = kinds == AGENT
    mask = space_mask(layout, "dynamics")
    assert not mask[~agent][:, agent].any(), "world reads agent keys"
    assert mask[agent].all(), "agent cannot read the world"
    assert not space_mask(layout, "dynamics", agent_active=False)[:, agent].any()


def branch_nonmutation(config: Config) -> None:
    """Candidate evaluations from one prefix must leave it untouched, so a planner
    comparing actions cannot corrupt the state it branched from. Mamba's step
    mutates in place, which is why the mixer clones rather than trusting callers."""
    device = _device(config)
    world = _world(config)
    rng = torch.Generator(device=device).manual_seed(0)
    latent = torch.randn(2, 1, config.n_spatial, config.d_spatial, device=device).tanh()
    with torch.no_grad():
        state, _ = initial(world, latent, torch.zeros(2, 1, dtype=torch.long, device=device), rng, config)
        state, _ = advance(world, state, torch.zeros(2, 1, dtype=torch.long, device=device), rng, config)
        before = tuple(tuple(t.clone() for t in pair) for pair in state.memory)
        for action in range(3):
            advance(world, state, torch.full((2, 1), action, device=device), torch.Generator(device=device).manual_seed(7), config)
    for pair, original in zip(state.memory, before):
        assert all(torch.equal(a, b) for a, b in zip(pair, original)), "branch mutated the prefix"


def recurrent_carry(config: Config) -> None:
    """The rollout must actually depend on history. A model that ignores its own
    memory passes every one-step loss and produces a constant trajectory.

    Both branches re-seed the generator identically, so a difference is history
    rather than a fresh corruption draw -- without that the flow arm's commit noise
    alone would satisfy the assertion.

    This is also the only place the flow arm's history dependence is tested:
    `_observation_dependence` compares `World.predict`, which for flow reads the
    current block's own corrupted latent and would pass with all history ignored.
    """
    device = _device(config)
    world = _world(config)
    rng = torch.Generator(device=device).manual_seed(0)
    latent = torch.randn(2, 1, config.n_spatial, config.d_spatial, device=device).tanh()
    action = torch.zeros(2, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        deep, _ = initial(world, latent, action, rng, config)
        for _ in range(4):
            deep, _ = advance(world, deep, action, torch.Generator(device=device).manual_seed(3), config)
        blank, _ = initial(world, deep.latent, action, rng, config)
        shallow, _ = advance(world, blank, action, torch.Generator(device=device).manual_seed(3), config)
        stepped, _ = advance(world, deep, action, torch.Generator(device=device).manual_seed(3), config)
    assert not torch.allclose(stepped.latent, shallow.latent, atol=1e-6), "memory is inert"
    _conditioning_coverage(config)


def _device(config: Config) -> str:
    """The configured device, the same one training uses. A gate suite that put the
    two arms on different hardware would validate a pairing the experiment never
    runs."""
    return config.device


def _world(config: Config):
    from .transition import World

    return World(config).to(_device(config)).eval()


def _probe(index: int, config: Config) -> Episode:
    """An episode where actions_taken[t] = t mod n_actions and rewards[t] = t, so a
    one-step shift is not a plausible alternative reading of any array."""
    steps = config.burn_in + config.sequence + 8 + index
    shape = (steps + 1, config.resolution, config.resolution, config.channels)
    return Episode(
        observations=torch.zeros(shape, dtype=torch.uint8),
        actions_taken=torch.arange(steps) % config.n_actions,
        rewards=torch.arange(steps).float(),
        terminated=torch.zeros(steps, dtype=torch.bool),
        truncated=torch.zeros(steps, dtype=torch.bool),
    )


def _recover_start(batch, row: int, config: Config) -> int:
    """The window's episode offset, read back out of the reward it carries rather
    than trusted from the sampler that produced it."""
    if bool(batch.valid[row][0]):
        return int(batch.reward[row][0]) + 1
    return 0
