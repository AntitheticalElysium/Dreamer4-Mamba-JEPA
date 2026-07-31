from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .config import Config

FORMAT = "d4mj_episodes_v1"


@dataclass(frozen=True)
class Episode:
    """Unshifted storage. `actions_taken[t]` is taken at `observations[t]` and
    causes `rewards[t]`, `terminated[t]`, `truncated[t]`, arriving at
    `observations[t + 1]`. Never reinterpret an index without renaming the array.

    `latents` is the frozen Z* cache written at the Phase-1A boundary; its digest
    covers the whole C* declaration, so a cache from another encoder or window
    cannot be silently reused.
    """

    observations: Tensor
    actions_taken: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    latents: Tensor | None = None
    latent_digest: str | None = None

    def __len__(self) -> int:
        return len(self.actions_taken)


@dataclass(frozen=True)
class Batch:
    """Block arrays under the led-to convention: block `i` holds the action that
    *produced* its observation, and the reward that arrived with it.

    `burn_in` leading blocks update encoder memory and score nothing. `valid` marks
    blocks whose incoming transition exists -- false only at a true episode start.
    Exactly one of `patches` and `latents` is populated, by phase.
    """

    led_to_action: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    valid: Tensor
    burn_in: int
    patches: Tensor | None = None
    latents: Tensor | None = None


def patchify(frames: Tensor, patch: int) -> Tensor:
    """(B, T, H, W, C) uint8 -> (B, T, n_patches, patch_dim) float in [0, 1]."""
    b, t, h, w, c = frames.shape
    grid = h // patch
    tiles = frames.view(b, t, grid, patch, grid, patch, c).permute(0, 1, 2, 4, 3, 5, 6)
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
    episodes: list[Episode], rng: torch.Generator, config: Config, step: int = 0
) -> Batch:
    """Dreamer 4 alternates short and long batches and finetunes on long ones. The
    long batch is the only one that exceeds the dynamics context, which is what
    stops the model assuming every context begins at an episode start.
    """
    cached = episodes[0].latents is not None
    burn_in = 0 if cached else config.burn_in
    long = config.long_batch_every > 0 and (step + 1) % config.long_batch_every == 0
    length = burn_in + (config.sequence_long if long else config.sequence)

    starts, chosen = [], []
    while len(chosen) < config.batch:
        index = int(torch.randint(len(episodes), (1,), generator=rng))
        episode = episodes[index]
        if len(episode) + 1 < length:
            continue
        chosen.append(episode)
        span = len(episode) + 1 - length
        starts.append(int(torch.randint(span + 1, (1,), generator=rng)))

    rows = [_window(episode, start, length, config) for episode, start in zip(chosen, starts)]
    stack = {field: torch.stack([row[field] for row in rows]) for field in rows[0]}
    return Batch(burn_in=burn_in, **stack)


def _window(episode: Episode, start: int, length: int, config: Config) -> dict[str, Tensor]:
    """One window in block coordinates. Block `i` covers episode step `start + i`;
    its incoming transition is `start + i - 1`, which exists unless that is -1."""
    steps = torch.arange(start, start + length)
    incoming = steps - 1
    valid = incoming >= 0
    source = incoming.clamp(min=0)

    cached = episode.latents is not None
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
    stale = [e.latent_digest for e in episodes if e.latents is not None and e.latent_digest != digest]
    if digest is not None and stale:
        raise ValueError("cached latents were produced under a different C*")
    return episodes
