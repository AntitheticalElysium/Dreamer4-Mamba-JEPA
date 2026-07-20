from dataclasses import replace
import importlib.util

import pytest
import torch

from d4_mamba_jepa.model import D4LiteWorld
from d4_mamba_jepa.source import load_mmbench2_model, verify_installed_mamba2
from d4_mamba_jepa.temporal import MambaTemporalState, MambaTimeMixer
from d4_mamba_jepa.tests.test_baseline import tiny_config


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
    reason="official mamba_ssm and CUDA are required",
)


def test_installed_mamba_matches_registered_official_source():
    assert (
        verify_installed_mamba2()
        == "605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707"
    )


def test_mamba_replaces_only_dynamics_time_attention():
    cfg = tiny_config(temporal_backend="mamba2")
    world = D4LiteWorld(cfg).cuda().eval()
    upstream = load_mmbench2_model()
    dynamics_time = [
        layer.time for layer in world.dynamics.transformer.layers if layer.do_time
    ]
    tokenizer_time = [
        layer.time for layer in world.encoder.transformer.layers if layer.do_time
    ]
    assert dynamics_time and all(
        isinstance(module, MambaTimeMixer) for module in dynamics_time
    )
    assert tokenizer_time and all(
        type(module) is upstream.TimeSelfAttention for module in tokenizer_time
    )


def test_parameter_matched_default_is_close_to_attention():
    cfg = replace(
        tiny_config(),
        dynamics_d_model=64,
        dynamics_heads=4,
        mamba_headdim=32,
        mamba_d_state=16,
        mamba_expand=1,
    )
    upstream = load_mmbench2_model()
    attention = upstream.TimeSelfAttention(64, 4, 0.0, False, cfg.n_spatial)
    mamba = MambaTimeMixer(cfg)
    attention_count = sum(parameter.numel() for parameter in attention.parameters())
    mamba_count = sum(parameter.numel() for parameter in mamba.parameters())
    assert abs(mamba_count - attention_count) / attention_count < 0.15


def test_sequence_prefill_step_equivalence_and_candidate_isolation():
    torch.manual_seed(19)
    cfg = tiny_config(temporal_backend="mamba2")
    mixer = MambaTimeMixer(cfg).cuda().to(torch.bfloat16).eval()
    x = torch.randn(2, 6, 3, cfg.dynamics_d_model, device="cuda", dtype=torch.bfloat16)
    candidate = torch.randn(
        2, 1, 3, cfg.dynamics_d_model, device="cuda", dtype=torch.bfloat16
    )
    with torch.no_grad():
        sequence = mixer(x)
        recurrent, state = mixer(x, return_kv_cache=True)
        assert isinstance(state, MambaTemporalState)
        before_conv = state.conv.clone()
        before_ssm = state.ssm.clone()
        branch_1 = mixer(candidate, kv_cache=state)
        branch_2 = mixer(candidate, kv_cache=state)
        torch.cuda.synchronize()
    torch.testing.assert_close(
        sequence.float(), recurrent.float(), atol=0.05, rtol=0.05
    )
    torch.testing.assert_close(branch_1.float(), branch_2.float(), atol=0, rtol=0)
    torch.testing.assert_close(state.conv, before_conv, atol=0, rtol=0)
    torch.testing.assert_close(state.ssm, before_ssm, atol=0, rtol=0)


def test_mamba_mixed_precision_gradients_are_finite():
    torch.manual_seed(23)
    cfg = tiny_config(temporal_backend="mamba2")
    mixer = MambaTimeMixer(cfg).cuda().float().train()
    x = torch.randn(
        2,
        6,
        3,
        cfg.dynamics_d_model,
        device="cuda",
        requires_grad=True,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = mixer(x)
        loss = output.float().square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in mixer.parameters()
    )


def test_mamba_world_flow_backward_is_finite():
    from d4_mamba_jepa.objectives import shortcut_flow_loss

    torch.manual_seed(29)
    cfg = tiny_config(temporal_backend="mamba2")
    world = D4LiteWorld(cfg).cuda().float().train()
    clean = torch.randn(
        2,
        cfg.sequence_length,
        cfg.n_spatial,
        cfg.d_spatial,
        device="cuda",
    )
    actions = torch.tensor(
        [[-1, 0, 1, 2], [-1, 3, 4, 5]], device="cuda"
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = shortcut_flow_loss(
            world.dynamics,
            clean=clean,
            led_to_actions=actions,
            k_max=cfg.k_max,
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for module in world.dynamics.transformer.layers
        if module.do_time
        for parameter in module.time.parameters()
    )
