from torch import Tensor, nn

from .config import Config

State = tuple[Tensor, Tensor]


class TimeAttention(nn.Module):
    """Causal attention over one token slot's history, with a persistent KV cache.

    The cache is carried across accepted frames rather than rebuilt per frame: the
    pinned reproduction rebuilds only because it re-corrupts its prefix at read
    time, which we do not.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        from .backbone import Attention

        self.attention = Attention(d_model, n_heads)

    def forward(self, x: Tensor, memory: State | None) -> tuple[Tensor, State]:
        return self.attention(x, causal=memory is None, cache=memory)


class TimeMamba(nn.Module):
    """Mamba-2 over one token slot's history: the whole Mamba blast radius.

    State is the pair Mamba-2 actually keeps -- a convolution window and an SSM
    state -- and both are cloned on entry, because `step` mutates in place and a
    candidate evaluation must not disturb the prefix it was given.
    """

    def __init__(self, config: Config, d_model: int):
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            headdim=config.mamba_headdim,
            layer_idx=0,
        )

    def forward(self, x: Tensor, memory: State | None) -> tuple[Tensor, State]:
        from mamba_ssm.utils.generation import InferenceParams

        params = InferenceParams(max_seqlen=x.shape[1] + 1, max_batch_size=x.shape[0])
        params.seqlen_offset = 0 if memory is None else 1
        params.key_value_memory_dict[0] = (
            self.mamba.allocate_inference_cache(x.shape[0], x.shape[1], dtype=x.dtype)
            if memory is None
            else tuple(tensor.clone() for tensor in memory)
        )
        return self.mamba(x, inference_params=params), params.key_value_memory_dict[0]


def time_mixer(config: Config, d_model: int) -> nn.Module:
    if config.time_mixer == "mamba":
        return TimeMamba(config, d_model)
    return TimeAttention(d_model, config.n_heads)
