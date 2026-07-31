import torch

from .backbone import AGENT, Backbone, Layout, space_mask
from .config import Config
from .data import Episode, sample_batch
from .state import WorldState
from .transition import advance


def scan_step_parity(config: Config, tolerance: float = 1e-3) -> None:
    """One batched scan must equal the same frames stepped one at a time carrying
    memory. Training runs the scan and imagination runs the steps, so a divergence
    here is a model that is correct in every loss and wrong in every rollout.

    The tolerance is relative and generous because Mamba's chunked scan and its
    sequential step are different kernels: measured drift is 2.5e-5 relative for
    Mamba and 2.5e-7 for attention, and neither grows with sequence length. A
    misalignment -- a stale offset, a dropped state -- is order one, not order
    1e-5, so the gap between the two is what the gate actually tests.
    """
    device = _device(config)
    layout = Layout.dynamics(config)
    backbone = Backbone(config, layout, "dynamics", config.d_model, config.n_heads, config.depth)
    backbone = backbone.to(device).eval()

    frames = torch.randn(2, 6, layout.size, config.d_model, device=device)
    with torch.no_grad():
        scanned, scanned_memory = backbone(frames)
        stepped, memory = [], None
        for index in range(frames.shape[1]):
            out, memory = backbone(frames[:, index : index + 1], memory)
            stepped.append(out)

    drift = (torch.cat(stepped, dim=1) - scanned).abs().max() / scanned.abs().max()
    assert drift < tolerance, f"scan/step drift {drift:.2e} exceeds {tolerance:.0e}"
    assert len(memory) == len(scanned_memory) == config.depth // config.time_every


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
        carried, _ = advance(world, WorldState(latent, None, 0), action, rng, config)
        follow, _ = advance(world, carried, action, torch.Generator(device=device).manual_seed(1), config)
        fresh, _ = advance(
            world, WorldState(carried.latent, None, 0), action, torch.Generator(device=device).manual_seed(1), config
        )
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
    state = WorldState(latent, None, 0)

    with torch.no_grad():
        state, _ = advance(world, state, torch.zeros(2, 1, dtype=torch.long, device=device), rng, config)
        before = tuple(tuple(t.clone() for t in pair) for pair in state.memory)
        for action in range(3):
            advance(world, state, torch.full((2, 1), action, device=device), torch.Generator(device=device).manual_seed(7), config)
    for pair, original in zip(state.memory, before):
        assert all(torch.equal(a, b) for a, b in zip(pair, original)), "branch mutated the prefix"


def recurrent_carry(config: Config) -> None:
    """The rollout must actually depend on history. A model that ignores its own
    memory passes every one-step loss and produces a constant trajectory."""
    device = _device(config)
    world = _world(config)
    rng = torch.Generator(device=device).manual_seed(0)
    latent = torch.randn(2, 1, config.n_spatial, config.d_spatial, device=device).tanh()
    action = torch.zeros(2, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        deep = WorldState(latent, None, 0)
        for _ in range(4):
            deep, _ = advance(world, deep, action, torch.Generator(device=device).manual_seed(3), config)
        shallow, _ = advance(
            world, WorldState(deep.latent, None, 0), action, torch.Generator(device=device).manual_seed(3), config
        )
        stepped, _ = advance(world, deep, action, torch.Generator(device=device).manual_seed(3), config)
    assert not torch.allclose(stepped.latent, shallow.latent, atol=1e-6), "memory is inert"


def _device(config: Config) -> str:
    """Mamba-2's kernels are CUDA-only, so a gate that silently ran the M arm on
    CPU would report a pass it never executed."""
    return "cuda" if config.time_mixer == "mamba" else "cpu"


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
