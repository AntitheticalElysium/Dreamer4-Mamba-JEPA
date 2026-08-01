from dataclasses import replace

import torch

from d4mj.backbone import AGENT, Backbone, Layout, space_mask


def build(config, checkpointing: bool):
    layout = Layout.dynamics(config)
    torch.manual_seed(0)
    backbone = Backbone(
        replace(config, gradient_checkpointing=checkpointing),
        layout,
        "dynamics",
        config.d_model,
        config.n_heads,
        config.depth,
        config.dynamics_context,
    )
    return layout, backbone


def test_checkpointing_is_exact(config):
    """Recomputation, not approximation: it must change cost and nothing else."""
    results = []
    for checkpointing in (False, True):
        layout, backbone = build(config, checkpointing)
        backbone.train()
        x = torch.randn(
            2, 6, layout.size, config.d_model, generator=torch.Generator().manual_seed(1)
        )
        out, _ = backbone(x)
        out.square().mean().backward()
        grads = torch.cat(
            [p.grad.flatten() for _, p in sorted(backbone.named_parameters()) if p.grad is not None]
        )
        results.append((out.detach(), grads))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.allclose(results[0][1], results[1][1], atol=1e-6)


def test_checkpointing_is_skipped_on_the_cached_path(config):
    """Re-entering a block would discard the state it returned."""
    layout, backbone = build(config, True)
    backbone.eval()
    x = torch.randn(2, 4, layout.size, config.d_model, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        scanned, scanned_memory = backbone(x)
        stepped, memory = [], None
        for index in range(x.shape[1]):
            out, memory = backbone(x[:, index : index + 1], memory, index)
            stepped.append(out)
    assert torch.allclose(torch.cat(stepped, dim=1), scanned, atol=1e-4)
    assert len(memory) == len(scanned_memory) == config.depth // config.time_every


def test_agent_firewall_is_one_way(config):
    """Agent state reaches the world only through the chosen action."""
    layout = Layout.dynamics(config)
    agent = layout.kinds == AGENT
    mask = space_mask(layout, "dynamics")
    assert not mask[~agent][:, agent].any(), "world reads agent keys"
    assert mask[agent].all(), "agent cannot read the world"
    assert not space_mask(layout, "dynamics", agent_active=False)[:, agent].any()


def test_attention_window_is_bounded(config):
    """The cache bound must hold in the batched path too, or it is not part of Z*."""
    layout, backbone = build(config, False)
    backbone.eval()
    length = config.dynamics_context + 4
    x = torch.randn(1, length, layout.size, config.d_model, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        _, memory = backbone(x)
    assert all(pair[0].shape[2] <= config.dynamics_context for pair in memory)
