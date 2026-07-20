"""Official Mamba-2 adapter for the upstream temporal-attention interface."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import D4LiteConfig
from .source import load_mmbench2_model, verify_installed_mamba2


@dataclass
class MambaTemporalState:
    """State for one temporal Mamba block over flattened spatial streams."""

    conv: Tensor
    ssm: Tensor

    def clone(self) -> "MambaTemporalState":
        return MambaTemporalState(self.conv.clone(), self.ssm.clone())


class MambaTimeMixer(nn.Module):
    """Drop-in replacement for MMBench2 ``TimeSelfAttention``.

    The surrounding upstream block retains its pre-norm and residual. This
    module therefore returns only the Mamba update, with no additional residual
    connection.

    Cache calls deliberately clone the registered prefix state before invoking
    official ``Mamba2.step``. Shortcut denoising evaluates multiple noisy
    candidates against the same clean prefix, and ``step`` mutates its cache in
    place; mutating the caller's prefix would leak one candidate into the next.
    """

    def __init__(self, cfg: D4LiteConfig):
        super().__init__()
        verify_installed_mamba2()
        from mamba_ssm.modules.mamba2 import Mamba2

        self.d_model = cfg.dynamics_d_model
        self.mamba = Mamba2(
            d_model=cfg.dynamics_d_model,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            headdim=cfg.mamba_headdim,
            ngroups=1,
            use_mem_eff_path=False,
        )

    @staticmethod
    def _flatten(x: Tensor) -> tuple[Tensor, tuple[int, int, int, int]]:
        if x.ndim != 4:
            raise ValueError("temporal input must have shape [B,T,S,D]")
        B, T, S, D = x.shape
        flat = x.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
        return flat, (B, T, S, D)

    @staticmethod
    def _restore(x: Tensor, shape: tuple[int, int, int, int]) -> Tensor:
        B, T, S, D = shape
        return x.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()

    def _empty_state(self, flat_batch: int, dtype: torch.dtype) -> MambaTemporalState:
        conv, ssm = self.mamba.allocate_inference_cache(
            flat_batch, max_seqlen=1, dtype=dtype
        )
        return MambaTemporalState(conv, ssm)

    def _recurrent(
        self, flat: Tensor, state: MambaTemporalState
    ) -> tuple[Tensor, MambaTemporalState]:
        outputs = []
        conv, ssm = state.conv, state.ssm
        for index in range(flat.shape[1]):
            output, conv, ssm = self.mamba.step(
                flat[:, index : index + 1], conv, ssm
            )
            outputs.append(output)
        if not outputs:
            raise ValueError("temporal sequence length must be positive")
        return torch.cat(outputs, dim=1), MambaTemporalState(conv, ssm)

    def forward(
        self,
        x_btSd: Tensor,
        kv_cache: MambaTemporalState | None = None,
        return_kv_cache: bool = False,
    ):
        flat, shape = self._flatten(x_btSd)
        if kv_cache is not None and return_kv_cache:
            raise ValueError("cannot consume and return a prefix cache together")

        if kv_cache is not None:
            if flat.shape[1] != 1:
                raise ValueError("cached Mamba decoding requires one timestep")
            # Official step mutates both cache tensors. Clone to make the prefix
            # reusable by every shortcut-denoising candidate.
            output, _ = self._recurrent(flat, kv_cache.clone())
            return self._restore(output, shape)

        if return_kv_cache:
            state = self._empty_state(flat.shape[0], flat.dtype)
            output, state = self._recurrent(flat, state)
            return self._restore(output, shape), state

        output = self.mamba(flat)
        return self._restore(output, shape)


def replace_dynamics_time_attention(
    dynamics: nn.Module, cfg: D4LiteConfig
) -> int:
    """Replace only registered dynamics temporal-attention modules."""
    upstream = load_mmbench2_model()
    replaced = 0
    for layer in dynamics.transformer.layers:
        if not layer.do_time:
            continue
        if type(layer.time) is not upstream.TimeSelfAttention:
            raise RuntimeError(
                f"unexpected temporal source module {type(layer.time).__name__}"
            )
        if layer.time.latents_only:
            raise RuntimeError("dynamics temporal mixing must include all tokens")
        layer.time = MambaTimeMixer(cfg)
        replaced += 1
    expected = cfg.dynamics_depth // cfg.dynamics_time_every
    if replaced != expected:
        raise RuntimeError(
            f"replaced {replaced} temporal modules; expected {expected}"
        )
    return replaced

