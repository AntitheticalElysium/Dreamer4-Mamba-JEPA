from __future__ import annotations
from dataclasses import dataclass
import warnings
import torch
from torch import Tensor, nn
from .utils import RMSNorm, reset_where


@dataclass
class TemporalState:
    """Opaque recurrent state.

    `backend_state` is intentionally not exposed to actor/reward heads. A Mamba
    implementation may hold conv/SSM caches; the GRU fallback holds hidden states.
    """
    backend_state: object
    output: Tensor                 # [B,S,D] latest emitted token state


class GRUTemporalBackend(nn.Module):
    """Correctness fallback used by smoke tests and unsupported hardware."""
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.cells = nn.ModuleList([nn.GRUCell(dim, dim) for _ in range(depth)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])
        self.depth = depth
        self.dim = dim

    def init_state(self, batch: int, streams: int, device: torch.device, dtype: torch.dtype) -> TemporalState:
        hs = [torch.zeros(batch * streams, self.dim, device=device, dtype=dtype) for _ in range(self.depth)]
        out = torch.zeros(batch, streams, self.dim, device=device, dtype=dtype)
        return TemporalState(hs, out)

    def step(self, x: Tensor, state: TemporalState, reset: Tensor | None = None) -> tuple[Tensor, TemporalState]:
        # x [B,S,D], flatten streams into independent temporal sequences.
        b, s, d = x.shape
        y = x.reshape(b * s, d)
        hs = list(state.backend_state)
        if reset is not None:
            stream_reset = reset[:, None].expand(b, s).reshape(-1)
            hs = [reset_where(h, stream_reset) for h in hs]
        new_hs = []
        for cell, norm, h in zip(self.cells, self.norms, hs):
            h_new = cell(y, h)
            y = y + norm(h_new)
            new_hs.append(h_new)
        out = y.reshape(b, s, d)
        return out, TemporalState(new_hs, out)

    def forward_sequence(self, x: Tensor, reset: Tensor | None = None) -> tuple[Tensor, TemporalState]:
        # x [B,T,S,D], reset [B,T].
        b, t, s, d = x.shape
        state = self.init_state(b, s, x.device, x.dtype)
        ys = []
        for i in range(t):
            r = None if reset is None else reset[:, i]
            y, state = self.step(x[:, i], state, r)
            ys.append(y)
        return torch.stack(ys, dim=1), state


class OptionalMambaBackend(nn.Module):
    """Adapter for official mamba_ssm when available.

    The exact Mamba-3 constructor/API is intentionally isolated here because its
    kernels are evolving. If import or construction fails, the caller uses GRU.
    """
    def __init__(self, dim: int, depth: int, backend: str, d_state: int, headdim: int):
        super().__init__()
        self.backend_name = backend
        try:
            if backend == "mamba3":
                from mamba_ssm.modules.mamba3 import Mamba3  # type: ignore
                cls = Mamba3
                kwargs = dict(d_model=dim, d_state=d_state, headdim=headdim)
            else:
                from mamba_ssm.modules.mamba2 import Mamba2  # type: ignore
                cls = Mamba2
                kwargs = dict(d_model=dim, d_state=d_state, headdim=headdim, use_mem_eff_path=False)
            self.layers = nn.ModuleList([cls(**kwargs) for _ in range(depth)])
            self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])
        except Exception as exc:
            raise RuntimeError(f"could not construct {backend}: {exc}") from exc

    def forward_sequence(self, x: Tensor, reset: Tensor | None = None):
        # Official fused sequence kernels generally do not accept interspersed reset
        # masks. Split at reset boundaries in production; smoke tests use GRU.
        if reset is not None and bool(reset[:, 1:].any()):
            raise NotImplementedError("Mamba sequence reset boundaries must be segmented before calling")
        b, t, s, d = x.shape
        y = x.permute(0, 2, 1, 3).reshape(b * s, t, d)
        for layer, norm in zip(self.layers, self.norms):
            y = y + layer(norm(y))
        y = y.reshape(b, s, t, d).permute(0, 2, 1, 3)
        return y, TemporalState(None, y[:, -1])

    def init_state(self, batch: int, streams: int, device: torch.device, dtype: torch.dtype):
        # A production implementation should call allocate_inference_cache() for each
        # layer. The exact official Mamba-3 step cache remains kernel/version specific.
        raise NotImplementedError("recurrent Mamba cache adapter must be pinned to the installed mamba_ssm version")

    def step(self, x: Tensor, state: TemporalState, reset: Tensor | None = None):
        raise NotImplementedError("pin and implement official Mamba step() API for deployment hardware")


class TemporalModel(nn.Module):
    """Temporal contract used by the rest of the architecture."""
    def __init__(self, dim: int, depth: int, backend: str, d_state: int, headdim: int):
        super().__init__()
        selected = backend
        if backend == "auto":
            selected = "mamba3"
        if selected in ("mamba3", "mamba2"):
            try:
                self.impl = OptionalMambaBackend(dim, depth, selected, d_state, headdim)
                self.name = selected
            except Exception as exc:
                warnings.warn(f"{exc}; falling back to GRU for correctness", RuntimeWarning)
                self.impl = GRUTemporalBackend(dim, depth)
                self.name = "gru"
        else:
            self.impl = GRUTemporalBackend(dim, depth)
            self.name = "gru"

    def forward_sequence(self, x: Tensor, reset: Tensor | None = None):
        return self.impl.forward_sequence(x, reset)

    def init_state(self, batch: int, streams: int, device: torch.device, dtype: torch.dtype):
        return self.impl.init_state(batch, streams, device, dtype)

    def step(self, x: Tensor, state: TemporalState, reset: Tensor | None = None):
        return self.impl.step(x, state, reset)
