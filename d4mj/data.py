from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from .config import Config, canonical_json
from .lewm_config import LeWMConfig

FORMAT = "d4mj_episodes_v3"
SHARD_FORMAT = "d4mj_episode_shard_v1"
STORE_FORMAT = "d4mj_episode_store_v1"


@dataclass(frozen=True)
class Episode:
    """Unshifted storage: `actions_taken[t]` is taken at `observations[t]` and causes
    `rewards[t]`, `terminated[t]`, `truncated[t]`, arriving at `observations[t + 1]`.

    `events[t]` is what happened; `uniform_eligible` and `bc_eligible` are separate
    facts about where the rollout belongs. Degraded exploratory data is
    uniform-eligible and not BC-eligible while keeping its true events, which one
    combined flag could not express.
    """

    observations: Tensor | None
    actions_taken: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    latents: Tensor | None = None
    latent_digest: str | None = None
    events: Tensor | None = None
    uniform_eligible: bool = True
    bc_eligible: bool = True
    epsilon: float | None = None
    split: Literal["train", "dev", "final"] | None = None
    episode_id: str | None = None
    terminal_cause: str | None = None

    def __post_init__(self) -> None:
        steps = len(self.actions_taken)
        assert self.observations is None or len(self.observations) == steps + 1
        assert len(self.rewards) == len(self.terminated) == len(self.truncated) == steps
        assert (self.latents is None) == (self.latent_digest is None)
        assert self.latents is None or len(self.latents) == steps + 1
        assert self.observations is not None or self.latents is not None
        assert self.events is None or len(self.events) == steps
        assert self.split in (None, "train", "dev", "final")

    def __len__(self) -> int:
        return len(self.actions_taken)


class EpisodeCorpus(Sequence[Episode]):
    """An indexed episode collection whose tensor storage may be memory mapped.

    The small Python records stay resident; observations or cached latents remain
    page-backed by their shard files. Sampling pools are indexed once per required
    length instead of rescanning a large corpus at every optimizer step.
    """

    def __init__(self, episodes: Iterable[Episode], *, source: Path | None = None):
        self._episodes = tuple(episodes)
        self.source = source
        self._profiles: dict[tuple[int, bool], dict[str, tuple[int, ...]]] = {}
        self._draw_tables: dict[tuple[int, str], Tensor] = {}

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return EpisodeCorpus(self._episodes[index], source=self.source)
        return self._episodes[index]

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes)

    def subset(self, indices: Iterable[int]) -> "EpisodeCorpus":
        return EpisodeCorpus((self._episodes[index] for index in indices), source=self.source)

    def pools(self, length: int) -> dict[str, tuple[int, ...]]:
        if not self._episodes:
            raise ValueError("episode corpus is empty")
        cached = self._episodes[0].latents is not None
        key = (length, cached)
        if key not in self._profiles:
            if any((episode.latents is not None) != cached for episode in self._episodes):
                raise AssertionError("cache is present on some episodes and missing on others")
            usable = tuple(
                index
                for index, episode in enumerate(self._episodes)
                if len(episode) + 1 >= length
            )
            uniform = tuple(index for index in usable if self._episodes[index].uniform_eligible)
            cloneable = tuple(index for index in usable if self._episodes[index].bc_eligible)
            eventful = tuple(
                index
                for index in cloneable
                if self._episodes[index].events is not None
                and bool(self._episodes[index].events.any())
            )
            terminal = tuple(
                index
                for index in uniform
                if bool(self._episodes[index].terminated.any())
            )
            self._profiles[key] = {
                "usable": usable,
                "uniform": uniform,
                "cloneable": cloneable,
                "eventful": eventful,
                "terminal": terminal,
            }
        return self._profiles[key]

    def window_weights(self, indices, span: int, *, dtype=torch.float) -> Tensor:
        """Number of valid windows per episode; callers retain their RNG draw order.

        `span` is the native extent a window occupies, which equals the frame
        count only for an unstrided, unwidened recipe.
        """
        counts = [len(self._episodes[index]) + 2 - span for index in indices]
        if any(count < 1 for count in counts):
            raise ValueError("window pool contains an episode shorter than the requested length")
        return torch.tensor(counts, dtype=dtype)

    def draw_window_episode(
        self, pool: str, length: int, rng: torch.Generator
    ) -> Episode:
        indices = self.pools(length)[pool]
        if not indices:
            raise ValueError(f"episode pool {pool!r} is empty at length {length}")
        key = (length, pool)
        if key not in self._draw_tables:
            self._draw_tables[key] = self.window_weights(indices, length)
        position = int(torch.multinomial(self._draw_tables[key], 1, generator=rng))
        return self._episodes[indices[position]]


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


@dataclass(frozen=True)
class BridgeBatch:
    """A Phase-2/3 main window plus its separate no-loss recurrent prefix.

    Every prefix ends at the first latent in ``main`` and contains exactly the recorded outgoing
    actions between its latents.  Prefixes remain ragged because padding observations would invent
    history.  ``burn_lengths`` counts completed prefix pairs (zero at a true episode start).
    Absolute episode/frame metadata remains outside recurrent state, as TC-14/TC-16 require.
    """

    main: Batch
    burn_latents: tuple[Tensor, ...]
    burn_actions: tuple[Tensor, ...]
    burn_lengths: Tensor
    episode_ids: tuple[str, ...]
    starts: Tensor

    def to(self, device: str) -> "BridgeBatch":
        moved = {
            name: value.to(device) if isinstance(value, Tensor) else value
            for name, value in vars(self.main).items()
        }
        return replace(
            self,
            main=Batch(**moved),
            burn_latents=tuple(value.to(device) for value in self.burn_latents),
            burn_actions=tuple(value.to(device) for value in self.burn_actions),
            burn_lengths=self.burn_lengths.to(device),
            starts=self.starts.to(device),
        )


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
    episodes: Sequence[Episode],
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
    if not episodes:
        raise ValueError("episode corpus is empty")
    cached = episodes[0].latents is not None
    burn_in = 0 if cached else config.burn_in
    finetune = total > 0 and step >= total * (1 - config.long_only_fraction)
    long = finetune or (config.long_batch_every > 0 and (step + 1) % config.long_batch_every == 0)
    length = burn_in + (config.sequence_long if long else config.sequence)

    corpus = episodes if isinstance(episodes, EpisodeCorpus) else None
    if corpus is not None:
        pools = corpus.pools(length)
        uniform = pools["uniform"]
        cloneable = pools["cloneable"]
        eventful = pools["eventful"]
    else:
        flags = [episode.latents is not None for episode in episodes]
        assert len(set(flags)) == 1, "cache is present on some episodes and missing on others"
        usable = [e for e in episodes if len(e) + 1 >= length]
        uniform = [e for e in usable if e.uniform_eligible]
        cloneable = [e for e in usable if e.bc_eligible]
        eventful = [e for e in cloneable if e.events is not None and bool(e.events.any())]
    if not uniform:
        raise ValueError(f"no eligible episode reaches the required length {length}")
    if mixture and not cloneable:
        raise ValueError("the relevant half needs BC-eligible episodes")

    wanted = config.batch // 2 if mixture else 0

    def draw(pool, name: str):
        """Uniform over eligible (episode, start) pairs, so a short episode's every
        window does not outweigh a long one's (S56)."""
        if corpus is not None:
            return corpus.draw_window_episode(name, length, rng)
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
            episode = draw(eventful, "eventful") if centre else draw(cloneable, "cloneable")
            offset = (
                _event_start(episode, length, rng)
                if centre
                else offset_for(episode, float(torch.rand((), generator=rng)) < config.episode_start_fraction)
            )
        else:
            episode = draw(uniform, "uniform")
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
    episodes: Sequence[Episode], rng: torch.Generator, config: Config, step: int, total: int
) -> Batch:
    """Tail-aligned terminal rows reserved for the continuation objective."""
    if not episodes:
        raise ValueError("episode corpus is empty")
    cached = episodes[0].latents is not None
    burn_in = 0 if cached else config.burn_in
    finetune = total > 0 and step >= total * (1 - config.long_only_fraction)
    long = finetune or (config.long_batch_every > 0 and (step + 1) % config.long_batch_every == 0)
    length = burn_in + (config.sequence_long if long else config.sequence)
    if isinstance(episodes, EpisodeCorpus):
        terminal = episodes.pools(length)["terminal"]
    else:
        flags = [episode.latents is not None for episode in episodes]
        assert len(set(flags)) == 1, "cache is present on some episodes and missing on others"
        terminal = [
            episode
            for episode in episodes
            if episode.uniform_eligible
            and len(episode) + 1 >= length
            and bool(episode.terminated.any())
        ]
    if not terminal:
        raise ValueError(f"no terminal episode reaches the required length {length}")

    chosen = []
    for _ in range(config.terminal_batch):
        selected = int(torch.randint(len(terminal), (1,), generator=rng))
        chosen.append(episodes[terminal[selected]] if isinstance(episodes, EpisodeCorpus) else terminal[selected])
    rows = [_window(episode, _terminal_start(episode, length), length, config) for episode in chosen]
    stack = {field: torch.stack([row[field] for row in rows]) for field in rows[0]}
    return Batch(
        burn_in=burn_in,
        relevant=torch.zeros(config.terminal_batch, dtype=torch.bool),
        support=torch.ones(config.terminal_batch, dtype=torch.bool),
        **stack,
    )


def _bridge_length(config: LeWMConfig, update: int) -> int:
    settings = config.agent
    if settings is None:
        raise ValueError("phase_gate: bridge sampling requires an M4 recipe")
    return settings.sequence_long if (update + 1) % settings.long_every == 0 else settings.sequence


def _weighted_episode(pool: Sequence[Episode], length: int, rng: torch.Generator,
                      *, nonstart: bool = False) -> Episode:
    """Draw uniformly over eligible windows, optionally excluding offset zero."""
    eligible = [episode for episode in pool if len(episode) + 1 >= length + int(nonstart)]
    if not eligible:
        kind = "non-start " if nonstart else ""
        raise ValueError(f"bridge_data: no {kind}episode reaches {length} frames")
    counts = torch.tensor(
        [len(episode) + 2 - length - int(nonstart) for episode in eligible], dtype=torch.float64
    )
    return eligible[int(torch.multinomial(counts, 1, generator=rng))]


def _bridge_event_start(episode: Episode, length: int, rng: torch.Generator) -> int | None:
    """A nonzero start containing an event action and its factual successor."""
    if episode.events is None:
        return None
    span = len(episode) + 1 - length
    candidates = []
    for event in episode.events.nonzero().flatten().tolist():
        low, high = max(1, event - length + 2), min(span, event)
        if low <= high:
            candidates.append((low, high))
    if not candidates:
        return None
    low, high = candidates[int(torch.randint(len(candidates), (), generator=rng))]
    return low + int(torch.randint(high - low + 1, (), generator=rng))


def _bridge_batch(episodes: Sequence[Episode], rng: torch.Generator, config: LeWMConfig,
                  update: int, *, terminal: bool) -> BridgeBatch:
    """Canonical cached-latent sampler for TC-14 through TC-17.

    Main rows are 32 frames, 128 every fourth update.  Main batches are exactly 50/50
    relevant/uniform and exactly 25% true starts within each half; terminal rows are tail aligned.
    Counterfactual forks are not a training source here.
    """
    settings = config.agent
    if settings is None:
        raise ValueError("phase_gate: bridge sampling requires an M4 recipe")
    if not episodes or any(episode.latents is None for episode in episodes):
        raise ValueError("bridge_data: Phase 2/3 requires the frozen post-joint latent cache")
    if any(episode.observations is not None for episode in episodes):
        raise ValueError("bridge_data: cached bridge inputs must not retain a pixel side channel")
    length = _bridge_length(config, update)
    count = settings.terminal_batch if terminal else settings.batch
    usable = [episode for episode in episodes if len(episode) + 1 >= length]
    uniform = [episode for episode in usable if episode.uniform_eligible]
    cloneable = [episode for episode in usable if episode.bc_eligible]
    eventful = [episode for episode in cloneable if _has_nonstart_event(episode, length)]
    terminal_pool = [episode for episode in uniform if bool(episode.terminated.any())]
    if not uniform or (not terminal and not cloneable):
        raise ValueError("bridge_data: required uniform/BC sampling pools are empty")
    if terminal and not terminal_pool:
        raise ValueError("bridge_data: no terminal support episode reaches the scheduled length")

    chosen: list[Episode] = []
    starts: list[int] = []
    roles: list[bool] = []
    if terminal:
        for _ in range(count):
            episode = terminal_pool[int(torch.randint(len(terminal_pool), (), generator=rng))]
            chosen.append(episode)
            starts.append(_terminal_start(episode, length))
            roles.append(False)
    else:
        relevant_count = count // 2
        # The start stratum is exact in each half, rather than merely true in expectation.
        start_per_half = int(round(relevant_count * settings.true_start_fraction))
        if 2 * start_per_half != int(round(count * settings.true_start_fraction)):
            raise ValueError("bridge_data: true-start fraction must divide both mixture halves")
        start_positions = set()
        for first in (0, relevant_count):
            order = torch.randperm(relevant_count, generator=rng)[:start_per_half].tolist()
            start_positions.update(first + item for item in order)
        for row in range(count):
            relevant = row < relevant_count
            pool = cloneable if relevant else uniform
            if row in start_positions:
                episode, start = _weighted_episode(pool, length, rng), 0
            else:
                centre = relevant and eventful and float(torch.rand((), generator=rng)) < settings.event_fraction
                if centre:
                    episode = _weighted_episode(eventful, length, rng, nonstart=True)
                    start = _bridge_event_start(episode, length, rng)
                    if start is None:  # Defensive: ``eventful`` was filtered by this condition.
                        raise AssertionError("bridge event pool contains no non-start event window")
                else:
                    episode = _weighted_episode(pool, length, rng, nonstart=True)
                    span = len(episode) + 1 - length
                    start = 1 + int(torch.randint(span, (), generator=rng))
            chosen.append(episode)
            starts.append(start)
            roles.append(relevant)

    rows = [_window(episode, start, length, config) for episode, start in zip(chosen, starts)]
    stack = {field: torch.stack([row[field] for row in rows]) for field in rows[0]}
    main = Batch(
        burn_in=0,
        relevant=torch.tensor(roles, dtype=torch.bool),
        support=(torch.ones(count, dtype=torch.bool) if terminal else None),
        **stack,
    )
    prefixes, actions, burn_lengths = [], [], []
    for episode, start in zip(chosen, starts):
        first = max(0, start - settings.burn_in)
        # Include z[start] so prefill ends on exactly main[:,0].
        prefixes.append(episode.latents[first:start + 1])
        actions.append(episode.actions_taken[first:start])
        burn_lengths.append(start - first)
    batch = BridgeBatch(
        main=main,
        burn_latents=tuple(prefixes),
        burn_actions=tuple(actions),
        burn_lengths=torch.tensor(burn_lengths, dtype=torch.long),
        episode_ids=tuple(str(episode.episode_id) for episode in chosen),
        starts=torch.tensor(starts, dtype=torch.long),
    )
    validate_bridge_batch(batch, config, terminal=terminal)
    return batch


def _has_nonstart_event(episode: Episode, length: int) -> bool:
    if episode.events is None:
        return False
    span = len(episode) + 1 - length
    return any(max(1, int(event) - length + 2) <= min(span, int(event))
               for event in episode.events.nonzero().flatten())


def sample_bridge_batch(episodes: Sequence[Episode], rng: torch.Generator,
                        config: LeWMConfig, update: int) -> BridgeBatch:
    return _bridge_batch(episodes, rng, config, update, terminal=False)


def sample_bridge_terminals(episodes: Sequence[Episode], rng: torch.Generator,
                            config: LeWMConfig, update: int) -> BridgeBatch:
    return _bridge_batch(episodes, rng, config, update, terminal=True)


def to_head_batch(batch: BridgeBatch) -> Batch:
    """The single adapter from explicit bridge metadata to legacy led-to head targets."""
    return batch.main


def validate_bridge_batch(batch: BridgeBatch, config: LeWMConfig, *, terminal: bool = False) -> None:
    settings = config.agent
    if settings is None:
        raise ValueError("phase_gate: bridge validation requires an M4 recipe")
    main, count = batch.main, settings.terminal_batch if terminal else settings.batch
    if main.latents is None or main.patches is not None or main.latents.shape[0] != count:
        raise ValueError("bridge_data: wrong cached batch geometry or a pixel side channel is present")
    if tuple(main.latents.shape[2:]) != (1, config.encoder.latent_dim):
        raise ValueError("bridge_data: cached latent shape differs from the exported contract")
    if len(batch.burn_latents) != count or len(batch.burn_actions) != count:
        raise ValueError("bridge_data: ragged prefix metadata does not match the batch")
    for row, (latents, actions, length) in enumerate(
        zip(batch.burn_latents, batch.burn_actions, batch.burn_lengths.tolist())
    ):
        if latents.shape[0] != length + 1 or actions.shape != (length,):
            raise ValueError("bridge_data: a burn-in must contain T+1 latents and T actions")
        if length > settings.burn_in or not torch.equal(latents[-1], main.latents[row, 0]):
            raise ValueError("bridge_data: prefix does not end at the main-window boundary")
    if terminal:
        if main.support is None or not bool(main.support.all()):
            raise ValueError("bridge_data: terminal rows must be continuation-only support")
    else:
        if int(main.relevant.sum()) != count // 2:
            raise ValueError("bridge_data: main rows must be exactly 50/50 relevant/uniform")
        expected = int(round(count * settings.true_start_fraction))
        if int((batch.starts == 0).sum()) != expected:
            raise ValueError("bridge_data: main rows do not satisfy the declared true-start stratum")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def save_episode_shard(path: Path, episodes: Sequence[Episode]) -> dict:
    """Write one independently recoverable, mmap-compatible episode shard."""
    if not episodes:
        raise ValueError("cannot write an empty episode shard")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"format": SHARD_FORMAT, "episodes": [vars(episode) for episode in episodes]},
        temporary,
    )
    temporary.replace(path)
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "episodes": len(episodes),
        "transitions": sum(len(episode) for episode in episodes),
        "terminal_episodes": sum(bool(episode.terminated.any()) for episode in episodes),
    }


def load_episode_store(
    path: Path,
    digest: str | None = None,
    *,
    verify: bool = True,
    allow_incomplete: bool = False,
) -> EpisodeCorpus:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != STORE_FORMAT:
        raise ValueError(f"expected {STORE_FORMAT}, found {manifest.get('format')}")
    if not allow_incomplete and not manifest.get("complete", False):
        raise ValueError(f"episode store is incomplete: {path}")
    episodes: list[Episode] = []
    for record in manifest["shards"]:
        shard_path = path / record["file"]
        if verify and _sha256(shard_path) != record["sha256"]:
            raise ValueError(f"episode shard digest mismatch: {shard_path}")
        payload = torch.load(shard_path, weights_only=False, mmap=True)
        if payload.get("format") != SHARD_FORMAT:
            raise ValueError(f"expected {SHARD_FORMAT}, found {payload.get('format')}")
        if len(payload["episodes"]) != record["episodes"]:
            raise ValueError(f"episode shard count mismatch: {shard_path}")
        episodes.extend(Episode(**fields) for fields in payload["episodes"])
    if len(episodes) != manifest["episodes"]:
        raise ValueError("episode-store manifest count does not match its shards")
    cached = [episode for episode in episodes if episode.latents is not None]
    if cached and digest is None:
        raise ValueError("cached latents require the expected C* digest to load")
    if any(episode.latent_digest != digest for episode in cached):
        raise ValueError("cached latents were produced under a different C*")
    return EpisodeCorpus(episodes, source=path)


def load_episodes(
    path: Path,
    digest: str | None = None,
    *,
    verify: bool = True,
    allow_incomplete: bool = False,
) -> EpisodeCorpus:
    if path.is_dir():
        return load_episode_store(
            path, digest, verify=verify, allow_incomplete=allow_incomplete
        )
    payload = torch.load(path, weights_only=False, mmap=True)
    if payload["format"] != FORMAT:
        raise ValueError(f"expected {FORMAT}, found {payload['format']}")
    episodes = [Episode(**fields) for fields in payload["episodes"]]
    cached = [episode for episode in episodes if episode.latents is not None]
    if cached and digest is None:
        raise ValueError("cached latents require the expected C* digest to load")
    if any(episode.latent_digest != digest for episode in cached):
        raise ValueError("cached latents were produced under a different C*")
    return EpisodeCorpus(episodes, source=path)


@dataclass(frozen=True)
class JointBatch:
    frames: Tensor
    actions: Tensor
    episode_ids: tuple[str, ...]
    starts: Tensor

    def to(self, device: str):
        return replace(self, frames=self.frames.to(device), actions=self.actions.to(device))


def validate_episode(episode: Episode, config: LeWMConfig) -> None:
    e = config.encoder
    if not episode.episode_id or episode.split not in ("train", "dev", "final"):
        raise ValueError("dataset_lineage: every episode needs an explicit ID and split")
    if episode.observations is None or episode.latents is not None:
        raise ValueError("joint_data: joint training requires raw observations, not a latent cache")
    if (episode.actions_taken.ndim != 1 or episode.rewards.shape != episode.actions_taken.shape
            or episode.terminated.shape != episode.actions_taken.shape or episode.truncated.shape != episode.actions_taken.shape
            or len(episode.observations) != len(episode.actions_taken)+1):
        raise ValueError("joint_data: require T outgoing actions/outcomes and T+1 observations")
    if episode.observations.dtype != torch.uint8 or episode.observations.shape[1:] != (e.resolution, e.resolution, 3):
        raise ValueError("joint_data: wrong raw pixel geometry/dtype")
    if episode.actions_taken.dtype != torch.long:
        raise ValueError("joint_data: actions must be int64")
    if bool(((episode.actions_taken < 0) | (episode.actions_taken >= config.dynamics.n_actions)).any()):
        raise ValueError("joint_data: invalid outgoing action")
    if episode.terminated.dtype != torch.bool or episode.truncated.dtype != torch.bool:
        raise ValueError("joint_data: terminal and timeout flags must be boolean")
    if bool((episode.terminated[:-1] | episode.truncated[:-1]).any()):
        raise ValueError("joint_data: episode contains a reset boundary before its final successor")
    if not bool(torch.isfinite(episode.rewards).all()):
        raise ValueError("joint_data: nonfinite rewards")


def audit_episodes(episodes, config: LeWMConfig) -> dict:
    ids, rows, split_counts = set(), [], {s: 0 for s in ("train", "dev", "final")}
    for episode in episodes:
        validate_episode(episode, config)
        if episode.episode_id in ids:
            raise ValueError("dataset_split: duplicate episode ID, including across splits")
        ids.add(episode.episode_id)
        split_counts[episode.split] += 1
        rows.append({"id": episode.episode_id, "split": episode.split, "steps": len(episode),
                     "uniform": episode.uniform_eligible, "bc": episode.bc_eligible,
                     "epsilon": episode.epsilon, "terminal_cause": episode.terminal_cause,
                     "terminals": int(episode.terminated.sum()), "timeouts": int(episode.truncated.sum()),
                     "events": None if episode.events is None else int(episode.events.sum())})
    span = (config.joint.frames - 1) * config.joint.stride + 1
    if not any(r["split"] == "train" and r["uniform"] and r["steps"] >= span-1 for r in rows):
        raise ValueError("joint_data: no eligible TRAIN windows")
    return {"episodes": rows, "split_counts": split_counts,
            "split_digest": hashlib.sha256(canonical_json(rows).encode()).hexdigest(),
            "transitions": sum(r["steps"] for r in rows)}


def load_joint_corpus(path: str | Path | Sequence[str | Path], config: LeWMConfig):
    """Validate bytes and full episode metadata. Never invent collector/split lineage.

    Several paths compose one corpus whose sources keep their own splits, eligibility flags and
    provenance. The corpus is immutable by reference rather than by copy: each source is pinned by
    the digest of its own manifest, so a merged corpus is auditable without duplicating tens of
    gigabytes of observations, and a single source keeps its original contract byte for byte.
    """
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    if not paths:
        raise ValueError("joint_data: a corpus needs at least one source")
    parts, sources = [], []
    for entry in paths:
        entry = Path(entry)
        if entry.name == "manifest.json":
            entry = entry.parent
        part = load_episodes(entry, verify=True)
        source_file = entry / "manifest.json" if entry.is_dir() else entry
        if entry.is_dir():
            manifest = json.loads(source_file.read_text())
            provenance = {k: v for k, v in manifest.items() if k != "shards"}
        else:
            sidecar = entry.with_suffix(entry.suffix + ".manifest.json")
            provenance = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if (config.agent is not None and config.runtime.purpose == "research"
                and provenance.get("kind") in {"d4mj_forks_flat_v1", "d4mj_m4_forkpool_v1"}):
            raise ValueError("joint_data: TC-17 keeps counterfactual fork corpora evaluation-only")
        parts.append(part)
        # Unknown upstream training access remains explicit, never relabeled as zero.
        sources.append({"path": str(entry), "sha256": _sha256(source_file), "episodes": len(part),
                        "provenance": provenance,
                        "collector_training_access": provenance.get("collector_training_access", "unknown")})
    # A single source returns the very object `load_episodes` built, so its `source` path and its
    # cached window pools survive. Rebuilding it as a plain list silently disabled
    # `EpisodeCorpus.pools` for every existing caller, which is a regression, not a refactor.
    episodes = parts[0] if len(parts) == 1 else EpisodeCorpus(e for part in parts for e in part)
    audit = audit_episodes(episodes, config)
    contract = {"schema": "d4mj_lewm_dataset_v1", "sha256": sources[0]["sha256"],
                "audit": audit, "provenance": sources[0]["provenance"],
                "collector_training_access": sources[0]["collector_training_access"]}
    if len(sources) > 1:
        # A merged corpus is a different dataset and must not be mistakable for its first source.
        contract["sha256"] = hashlib.sha256(
            canonical_json([s["sha256"] for s in sources]).encode()).hexdigest()
        contract["sources"] = sources
    return episodes, contract


class JointSampler:
    def __init__(self, episodes, config: LeWMConfig, generator: torch.Generator):
        audit_episodes(episodes, config)
        self.config, self.generator = config, generator
        from .lewm_config import window_layout
        self.offsets = window_layout(config.joint)[0]
        span = self.offsets[-1] + 1
        self.episodes = EpisodeCorpus(e for e in episodes if e.split == "train" and e.uniform_eligible
                              and len(e)+1 >= span)
        self.counts = self.episodes.window_weights(range(len(self.episodes)), span, dtype=torch.float64)
        self.draws = 0

    def sample(self) -> JointBatch:
        j = self.config.joint
        selected = torch.multinomial(self.counts, j.batch, replacement=True, generator=self.generator)
        frames, actions, ids, starts = [], [], [], []
        for index in selected.tolist():
            episode = self.episodes[index]
            start = int(torch.randint(int(self.counts[index]), (), generator=self.generator))
            # Encoded frames sit at the layout's native offsets: the prediction
            # frames plus any extra frames a widened centering window needs.
            frames.append(episode.observations[[start + o for o in self.offsets]])
            # Actions belong to the prediction pairs, which span (frames-1)*stride
            # native steps -- not the whole encoded window, which a widened
            # centering set makes longer without adding transitions to predict.
            inner = episode.actions_taken[start:start + (j.frames - 1) * j.stride]
            actions.append(inner if j.stride == 1 else inner.reshape(j.frames - 1, j.stride))
            ids.append(episode.episode_id)
            starts.append(start)
        self.draws += j.batch
        return JointBatch(torch.stack(frames), torch.stack(actions), tuple(ids), torch.tensor(starts))

    def state_dict(self):
        return {"generator": self.generator.get_state(), "draws": self.draws}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"].cpu())
        self.draws = int(state["draws"])


def screen_windows(episodes, config: LeWMConfig, screen, split: str) -> dict:
    """Fixed episode-cluster sample for G1; no FINAL windows or probe-fit leakage."""
    if split not in ("train", "dev"):
        raise ValueError("joint screen never selects FINAL")
    wanted = screen.train_episodes if split == "train" else screen.dev_episodes
    from .lewm_config import window_layout
    length, stride = config.joint.frames, config.joint.stride
    # The screen samples the same window shape training does, so a widened
    # centering window is screened on the frames it actually encodes.
    offsets = window_layout(config.joint)[0]
    span = offsets[-1] + 1
    def order(e):
        return hashlib.sha256(f"{screen.seed}:{e.episode_id}".encode()).digest()
    pool = sorted((e for e in episodes if e.split == split and e.uniform_eligible
                   and len(e)+2-span >= screen.windows_per_episode), key=order)
    if len(pool) < wanted:
        raise ValueError(f"screen_coverage: {split} has {len(pool)} eligible episodes, needs {wanted}")
    frames, actions, labels, valid, ids, starts, clusters = [], [], [], [], [], [], []
    for cluster, episode in enumerate(pool[:wanted]):
        rng = torch.Generator().manual_seed(int.from_bytes(order(episode)[:8], "little") % (2**63-1))
        positions = torch.randperm(len(episode)+2-span, generator=rng)[:screen.windows_per_episode]
        for start in positions.tolist():
            end = start+span-1
            # Each retained transition covers `stride` native steps, so its label
            # aggregates over them: total reward, and any event or termination.
            inner = lambda t: t[start:start+(length-1)*stride].reshape(length-1, stride)
            reward = inner(episode.rewards).sum(-1)
            event = (torch.zeros_like(reward, dtype=torch.bool) if episode.events is None
                     else inner(episode.events).any(-1))
            truth = torch.stack((reward > 0, reward < 0, event, inner(episode.terminated).any(-1)), -1)
            mask = torch.ones_like(truth)
            if episode.events is None:
                mask[:, 2] = False
            frames.append(episode.observations[[start + o for o in offsets]])
            outgoing = episode.actions_taken[start:start + (length - 1) * stride]
            actions.append(outgoing if stride == 1 else outgoing.reshape(length - 1, stride))
            labels.append(truth); valid.append(mask); ids.append(episode.episode_id)
            starts.append(start); clusters.append(cluster)
    return {"frames": torch.stack(frames), "actions": torch.stack(actions),
            "labels": torch.stack(labels), "valid": torch.stack(valid), "episode_ids": tuple(ids),
            "starts": torch.tensor(starts), "clusters": torch.tensor(clusters),
            "label_names": ("positive_reward", "negative_reward", "achievement_event", "termination")}
