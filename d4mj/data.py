from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .config import Config

FORMAT = "d4mj_episodes_v3"


@dataclass(frozen=True)
class Episode:
    """Unshifted storage: `actions_taken[t]` is taken at `observations[t]` and causes
    `rewards[t]`, `terminated[t]`, `truncated[t]`, arriving at `observations[t + 1]`.

    `events[t]` is what happened; `uniform_eligible` and `bc_eligible` are separate
    facts about where the rollout belongs. Degraded exploratory data is
    uniform-eligible and not BC-eligible while keeping its true events, which one
    combined flag could not express.
    """

    observations: Tensor
    actions_taken: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    latents: Tensor | None = None
    latent_digest: str | None = None
    events: Tensor | None = None
    uniform_eligible: bool = True
    bc_eligible: bool = True

    def __post_init__(self) -> None:
        steps = len(self.actions_taken)
        assert len(self.observations) == steps + 1
        assert len(self.rewards) == len(self.terminated) == len(self.truncated) == steps
        assert (self.latents is None) == (self.latent_digest is None)
        assert self.latents is None or len(self.latents) == steps + 1
        assert self.events is None or len(self.events) == steps

    def __len__(self) -> int:
        return len(self.actions_taken)


@dataclass(frozen=True)
class Batch:
    """Block arrays under the led-to convention: block `i` holds the action that
    produced its observation and the reward that arrived with it.

    `scored` marks blocks whose encoder history matches deployment, per block and
    per row. `valid` is false only at a true episode start. `relevant` is the row's
    sampling role, `None` while pretraining. `support` marks auxiliary
    terminal-exposure rows that only the continuation loss may read.
    """

    led_to_action: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    valid: Tensor
    scored: Tensor
    burn_in: int
    relevant: Tensor | None = None
    support: Tensor | None = None
    patches: Tensor | None = None
    latents: Tensor | None = None

    def rows(self, role: str) -> Tensor:
        """Which rows a loss may read. `dynamics` is the uniform half, `policy` the
        relevant half, `reward` everything except support, `continuation` everything
        -- support rows exist to give the continuation head terminals it would
        otherwise almost never see, and must not reach any other objective."""
        count = self.led_to_action.shape[0]
        support = torch.zeros(count, dtype=torch.bool) if self.support is None else self.support
        support = support.to(self.led_to_action.device)
        if role == "continuation":
            return torch.ones(count, dtype=torch.bool, device=support.device)
        if role == "reward":
            return ~support
        if self.relevant is None:
            return ~support
        return (self.relevant if role == "policy" else ~self.relevant) & ~support


def patchify(frames: Tensor, patch: int) -> Tensor:
    """(B, T, H, W, C) uint8 -> (B, T, n_patches, patch_dim) float in [0, 1]."""
    b, t, h, w, c = frames.shape
    grid = h // patch
    tiles = frames.reshape(b, t, grid, patch, grid, patch, c).permute(0, 1, 2, 4, 3, 5, 6)
    return tiles.reshape(b, t, grid * grid, patch * patch * c).float() / 255.0


def unpatchify(patches: Tensor, config: Config) -> Tensor:
    """Inverse of `patchify`, in channels-first for LPIPS."""
    b, t = patches.shape[:2]
    grid, p, c = config.resolution // config.patch, config.patch, config.channels
    tiles = patches.view(b, t, grid, grid, p, p, c).permute(0, 1, 6, 2, 4, 3, 5)
    return tiles.reshape(b, t, c, config.resolution, config.resolution)


def episode_splits(count: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """Whole-episode 80/10/10. Windows never cross episodes, so splitting whole
    episodes is what keeps evaluation frames out of training."""
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    train, dev = int(0.8 * count), int(0.9 * count)
    return order[:train], order[train:dev], order[dev:]


def sample_batch(
    episodes: list[Episode],
    rng: torch.Generator,
    config: Config,
    step: int = 0,
    total: int = 0,
    mixture: bool = False,
) -> Batch:
    """Short and long batches alternate; the last `long_only_fraction` is long-only.
    `mixture` is the §4.1 regime, used by Phases 2 and 3.

    Uniform rows are drawn over eligible *(episode, start)* pairs. Episode-start
    rows form an explicit stratum; terminal supervision uses a separate batch.
    """
    cached = [episode.latents is not None for episode in episodes]
    assert len(set(cached)) == 1, "cache is present on some episodes and missing on others"
    burn_in = 0 if cached[0] else config.burn_in
    finetune = total > 0 and step >= total * (1 - config.long_only_fraction)
    long = finetune or (config.long_batch_every > 0 and (step + 1) % config.long_batch_every == 0)
    length = burn_in + (config.sequence_long if long else config.sequence)

    usable = [e for e in episodes if len(e) + 1 >= length]
    uniform = [e for e in usable if e.uniform_eligible]
    cloneable = [e for e in usable if e.bc_eligible]
    eventful = [e for e in cloneable if e.events is not None and bool(e.events.any())]
    if not uniform:
        raise ValueError(f"no eligible episode reaches the required length {length}")
    if mixture and not cloneable:
        raise ValueError("the relevant half needs BC-eligible episodes")

    wanted = config.batch // 2 if mixture else 0

    def draw(pool):
        """Uniform over eligible (episode, start) pairs, so a short episode's every
        window does not outweigh a long one's (S56)."""
        counts = torch.tensor([len(e) + 1 - length + 1 for e in pool], dtype=torch.float)
        return pool[int(torch.multinomial(counts, 1, generator=rng))]

    def offset_for(episode, at_start: bool) -> int:
        span = len(episode) + 1 - length
        return 0 if at_start else int(torch.randint(span + 1, (1,), generator=rng))

    chosen, offsets, roles = [], [], []
    for row in range(config.batch):
        relevant = row < wanted
        if relevant:
            # Behaviour cloning reads ordinary expert behaviour, with task events
            # oversampled rather than exclusive (S51 revised): navigation, survival
            # and positioning are all worth cloning for one aggregate policy, and
            # event-only windows reached 15.5% of expert transitions.
            centre = eventful and float(torch.rand((), generator=rng)) < config.event_fraction
            episode = draw(eventful) if centre else draw(cloneable)
            offset = (
                _event_start(episode, length, rng)
                if centre
                else offset_for(episode, float(torch.rand((), generator=rng)) < config.episode_start_fraction)
            )
        else:
            episode = draw(uniform)
            offset = offset_for(episode, float(torch.rand((), generator=rng)) < config.episode_start_fraction)
        chosen.append(episode)
        offsets.append(offset)
        roles.append(relevant)

    rows = [_window(e, offset, length, config) for e, offset in zip(chosen, offsets)]
    stack = {field: torch.stack([row[field] for row in rows]) for field in rows[0]}
    return Batch(
        burn_in=burn_in,
        relevant=torch.tensor(roles) if mixture else None,
        **stack,
    )


def sample_terminal_batch(
    episodes: list[Episode], rng: torch.Generator, config: Config, step: int, total: int
) -> Batch:
    """Tail-aligned terminal rows reserved for the continuation objective."""
    cached = [episode.latents is not None for episode in episodes]
    assert len(set(cached)) == 1, "cache is present on some episodes and missing on others"
    burn_in = 0 if cached[0] else config.burn_in
    finetune = total > 0 and step >= total * (1 - config.long_only_fraction)
    long = finetune or (config.long_batch_every > 0 and (step + 1) % config.long_batch_every == 0)
    length = burn_in + (config.sequence_long if long else config.sequence)
    terminal = [
        episode
        for episode in episodes
        if episode.uniform_eligible and len(episode) + 1 >= length and bool(episode.terminated.any())
    ]
    if not terminal:
        raise ValueError(f"no terminal episode reaches the required length {length}")

    chosen = [
        terminal[int(torch.randint(len(terminal), (1,), generator=rng))]
        for _ in range(config.terminal_batch)
    ]
    rows = [_window(episode, _terminal_start(episode, length), length, config) for episode in chosen]
    stack = {field: torch.stack([row[field] for row in rows]) for field in rows[0]}
    return Batch(
        burn_in=burn_in,
        relevant=torch.zeros(config.terminal_batch, dtype=torch.bool),
        support=torch.ones(config.terminal_batch, dtype=torch.bool),
        **stack,
    )


def _event_start(episode: Episode, length: int, rng: torch.Generator) -> int:
    """A window holding the whole event transition. `events[e]` says action `e`
    caused an achievement arriving at observation `e + 1`, so both must be inside:
    a start of `e - length + 1` puts observation `e` last and leaves the arrival,
    and the BC target that depends on it, outside."""
    span = len(episode) + 1 - length
    events = episode.events.nonzero().flatten()
    event = int(events[int(torch.randint(len(events), (1,), generator=rng))])
    low, high = max(0, event - length + 2), min(span, event)
    return low + int(torch.randint(high - low + 1, (1,), generator=rng))


def _terminal_start(episode: Episode, length: int) -> int:
    """The tail window, so the terminal transition is inside it. A terminal is only
    visible when the window is tail-aligned, which at a uniform start happens about
    once in a span."""
    return max(0, len(episode) + 1 - length)


def _window(episode: Episode, start: int, length: int, config: Config) -> dict[str, Tensor]:
    """One window in block coordinates. Block `i` covers episode step `start + i`;
    its incoming transition is `start + i - 1`, which exists unless that is -1.

    A block is `scored` once it holds a full receptive field, and unconditionally
    when the window starts at the episode start, where nothing earlier is missing.
    """
    steps = torch.arange(start, start + length)
    incoming = steps - 1
    valid = incoming >= 0
    source = incoming.clamp(min=0)
    cached = episode.latents is not None
    depth = torch.arange(length) + 1
    scored = (
        torch.ones(length, dtype=torch.bool)
        if cached
        else (depth >= config.receptive_field) | torch.tensor(start == 0)
    )
    frames = (
        episode.latents[steps]
        if cached
        else patchify(episode.observations[steps][None], config.patch)[0]
    )
    return {
        "led_to_action": torch.where(
            valid, episode.actions_taken[source], torch.tensor(config.n_actions)
        ),
        "reward": torch.where(valid, episode.rewards[source], torch.zeros(())),
        "terminated": episode.terminated[source] & valid,
        "truncated": episode.truncated[source] & valid,
        "valid": valid,
        "scored": scored,
        "latents" if cached else "patches": frames,
    }


def save_episodes(path: Path, episodes: list[Episode]) -> None:
    payload = {"format": FORMAT, "episodes": [vars(episode) for episode in episodes]}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.rename(path)


def load_episodes(path: Path, digest: str | None = None) -> list[Episode]:
    payload = torch.load(path, weights_only=False)
    if payload["format"] != FORMAT:
        raise ValueError(f"expected {FORMAT}, found {payload['format']}")
    episodes = [Episode(**fields) for fields in payload["episodes"]]
    cached = [episode for episode in episodes if episode.latents is not None]
    if cached and digest is None:
        raise ValueError("cached latents require the expected C* digest to load")
    if any(episode.latent_digest != digest for episode in cached):
        raise ValueError("cached latents were produced under a different C*")
    return episodes
