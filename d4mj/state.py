from dataclasses import dataclass

from torch import Tensor

Memory = tuple[tuple[Tensor, Tensor], ...]
"""Per temporal layer: (keys, values) for attention, (conv, ssm) for Mamba. Opaque
outside `time_mixer`, and never mutated in place."""


@dataclass(frozen=True)
class WorldState:
    """The imagined state S_t = (z_t, m_t). `latent` is the accepted clean latent;
    `memory` ingested whatever the committed block held, corrupted for flow, and
    covers the prefix through block t inclusive -- so `latent` is never ingested
    twice."""

    latent: Tensor
    memory: Memory
    step: int
    features: Tensor | None = None


@dataclass(frozen=True)
class RealState:
    """The deployed state, adding the tokenizer's own bounded-window memory, which
    imagination has no use for. A reset clears both."""

    encoder_memory: Memory
    world: WorldState
