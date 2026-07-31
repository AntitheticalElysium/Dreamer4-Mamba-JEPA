import torch

from .backbone import Backbone, Layout
from .config import Config
from .data import Episode, sample_batch


def scan_step_parity(config: Config, tolerance: float = 1e-4) -> None:
    """One batched scan must equal the same frames stepped one at a time carrying
    memory. Training runs the scan and imagination runs the steps, so a divergence
    here is a model that is correct in every loss and wrong in every rollout.
    """
    device = "cuda" if config.time_mixer == "mamba" else "cpu"
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

    drift = (torch.cat(stepped, dim=1) - scanned).abs().max().item()
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
