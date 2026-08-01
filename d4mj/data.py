from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .config import Config

FORMAT = "d4mj_episodes_v2"


@dataclass(frozen=True)
class Episode:
    """Unshifted storage. `actions_taken[t]` is taken at `observations[t]` and
    causes `rewards[t]`, `terminated[t]`, `truncated[t]`, arriving at
    `observations[t + 1]`. Never reinterpret an index without renaming the array.

    `latents` is the frozen Z* cache written at the Phase-1A boundary; its digest
    covers the whole C* declaration, so a cache from another encoder or window
    cannot be silently reused.

    `events[t]` marks a step accomplishing a task -- per step, not per episode,
    because D4 §4.1 oversamples "relevant sequences that accomplish one of the
    tasks" and a random window from a successful 2500-step episode mostly shows
    walking and inventory management.
    """

    observations: Tensor
    actions_taken: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    latents: Tensor | None = None
    latent_digest: str | None = None
    events: Tensor | None = None

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
    *produced* its observation, and the reward that arrived with it.

    `scored` marks blocks whose encoder history is the history they would have at
    deployment, so the reconstruction loss may use them. It is per block and per
    row, not one leading count (S31 withdrawn): a window starting at the episode
    start is missing nothing, so its earliest blocks are faithful at their own
    shorter history and must be scored. A single integer masked them out of every
    batch, which is why the first `receptive_field - 1` states of an episode --
    exactly the states deployment begins from -- were never supervised.

    `valid` marks blocks whose incoming transition exists -- false only at a true
    episode start. Exactly one of `patches` and `latents` is populated, by phase.

    `relevant` is the **sampling role** of a row, not a property of its data, and
    is `None` during pretraining, where D4 applies no mixture and every row is
    scored by every loss. A uniformly drawn window containing a task event is still
    a uniform row.
    """

    led_to_action: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    valid: Tensor
    scored: Tensor
    burn_in: int
    relevant: Tensor | None = None
    patches: Tensor | None = None
    latents: Tensor | None = None


def patchify(frames: Tensor, patch: int) -> Tensor:
    """(B, T, H, W, C) uint8 -> (B, T, n_patches, patch_dim) float in [0, 1]."""
    b, t, h, w, c = frames.shape
    grid = h // patch
    tiles = frames.reshape(b, t, grid, patch, grid, patch, c).permute(0, 1, 2, 4, 3, 5, 6)
    return tiles.reshape(b, t, grid * grid, patch * patch * c).float() / 255.0


def unpatchify(patches: Tensor, config: Config) -> Tensor:
    """Inverse of `patchify`, for the perceptual term. (B, T, N, D) -> (B, T, C, H, W)
    in channels-first, which is what an LPIPS network expects."""
    b, t = patches.shape[:2]
    grid, p, c = config.resolution // config.patch, config.patch, config.channels
    tiles = patches.view(b, t, grid, grid, p, p, c).permute(0, 1, 6, 2, 4, 3, 5)
    return tiles.reshape(b, t, c, config.resolution, config.resolution)


def episode_splits(count: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """Whole-episode 80/10/10. Windows never cross episodes, so splitting whole
    episodes is what keeps evaluation frames out of training entirely."""
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
    """Short and long batches alternate, and the last `long_only_fraction` of
    training is long-only, as D4's finetune is. Only the long batch exceeds the
    dynamics context, which is what stops the model assuming every context begins
    at an episode start.

    `mixture` selects the regime. Pretraining (Phases 1A and 1B) passes False: D4
    pretrains on the whole corpus and every row is scored by every loss. Agent
    finetuning passes True for the §4.1 mixture -- half the rows drawn uniformly,
    half drawn so the window *contains* a task event.

    Nothing falls back to the other pool. A silent substitution would train dynamics
    on task-accomplishing play, which is the one thing the mixture exists to prevent.

    A quarter of the uniformly drawn rows start at the episode start. Those are the
    only windows whose earliest blocks are scorable at all (see `_window`), and at a
    uniform start they arrive with probability 1/span -- about 0.5% here -- so the
    states deployment actually begins from would be supervised essentially never.
    """
    cached = [episode.latents is not None for episode in episodes]
    assert len(set(cached)) == 1, "cache is present on some episodes and missing on others"
    burn_in = 0 if cached[0] else config.burn_in
    finetune = total > 0 and step >= total * (1 - config.long_only_fraction)
    long = finetune or (config.long_batch_every > 0 and (step + 1) % config.long_batch_every == 0)
    length = burn_in + (config.sequence_long if long else config.sequence)

    usable = [episode for episode in episodes if len(episode) + 1 >= length]
    if not usable:
        raise ValueError(f"no episode reaches the required length {length}")

    eventful = [e for e in usable if e.events is not None and bool(e.events.any())]
    if mixture and not eventful:
        raise ValueError("the relevant half needs episodes carrying task events")

    chosen, starts, roles = [], [], []
    for row in range(config.batch):
        relevant = mixture and row < config.batch // 2
        episode = (eventful if relevant else usable)[
            int(torch.randint(len(eventful if relevant else usable), (1,), generator=rng))
        ]
        span = len(episode) + 1 - length
        if relevant:
            events = episode.events.nonzero().flatten()
            event = int(events[int(torch.randint(len(events), (1,), generator=rng))])
            low, high = max(0, event - length + 1), min(span, event)
            start = low + int(torch.randint(high - low + 1, (1,), generator=rng))
        elif row % 4 == 0:
            start = 0
        else:
            start = int(torch.randint(span + 1, (1,), generator=rng))
        chosen.append(episode)
        starts.append(start)
        roles.append(relevant)

    rows = [_window(episode, start, length, config) for episode, start in zip(chosen, starts)]
    stack = {field: torch.stack([row[field] for row in rows]) for field in rows[0]}
    return Batch(burn_in=burn_in, relevant=torch.tensor(roles) if mixture else None, **stack)


def _window(episode: Episode, start: int, length: int, config: Config) -> dict[str, Tensor]:
    """One window in block coordinates. Block `i` covers episode step `start + i`;
    its incoming transition is `start + i - 1`, which exists unless that is -1.

    Block `i` sees `i + 1` frames of history from this window against the
    `start + i + 1` it has in the episode, so it is faithful once it holds a full
    receptive field, and unconditionally when the window starts at the episode
    start -- there is nothing earlier to be missing.
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
        episode.latents[steps] if cached else patchify(episode.observations[steps][None], config.patch)[0]
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
