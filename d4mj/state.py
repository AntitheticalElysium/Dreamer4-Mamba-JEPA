from dataclasses import dataclass

from torch import Tensor

Memory = tuple[tuple[Tensor, Tensor], ...]
"""Per temporal layer, a pair: (keys, values) for attention, (conv, ssm) for Mamba.

Opaque outside `time_mixer`. Never mutated in place, so a candidate evaluation is
read-only by construction and branching needs no explicit copy.
"""


@dataclass(frozen=True)
class WorldState:
    """The imagined state S_t = (z_t, m_t).

    `latent` is always the accepted clean latent in Z* space. `memory` ingests
    whatever the committed block actually held -- for the flow arm, a corrupted
    copy. Losses, decoding and diagnostics read `latent`; nothing reads the
    corrupted copy back out. `memory` covers the prefix through block t inclusive,
    so `latent` must never be ingested a second time.
    """

    latent: Tensor
    memory: Memory
    step: int


@dataclass(frozen=True)
class RealState:
    """The deployed state S_t^real = (e_t, z_t, m_t).

    `encoder_memory` is the tokenizer's own bounded-window state and is absent from
    imagination, where no new observation is encoded. An actual environment reset
    clears both memories, independently of whether the transition bootstraps.
    """

    encoder_memory: Memory
    world: WorldState
