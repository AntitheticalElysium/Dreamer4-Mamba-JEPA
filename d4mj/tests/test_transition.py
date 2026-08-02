from dataclasses import replace

import pytest
import torch

from d4mj.transition import World, commit_inputs, flow_conditioning, transition_loss

from .conftest import latent_batch

ARMS = ("flow", "direct")


def world_for(config, transition, seed=0):
    torch.manual_seed(seed)
    return World(replace(config, transition=transition))


def loss_of(world, batch, config, seed=3):
    return transition_loss(world, batch, torch.Generator().manual_seed(seed), config)


@pytest.mark.parametrize("transition", ARMS)
def test_pretraining_scores_every_row(config, transition):
    """`relevant is None` is the pretraining regime: restricting it to uniform rows
    would discard half the corpus D4 pretrains on."""
    config = replace(config, transition=transition)
    world = world_for(config, transition)
    base = latent_batch(config, 4, 6)
    moved = latent_batch(config, 4, 6)
    moved.latents[:2] = torch.randn_like(moved.latents[:2]).tanh()
    assert not torch.equal(loss_of(world, base, config), loss_of(world, moved, config))


@pytest.mark.parametrize("transition", ARMS)
def test_finetuning_dynamics_ignores_the_relevant_half(config, transition):
    """§4.1 applies the continued dynamics loss to uniform rows only, so that
    imagination is not fitted to task-accomplishing play."""
    config = replace(config, transition=transition)
    world = world_for(config, transition)
    half = [True, True, False, False]
    base = latent_batch(config, 4, 6, relevant=half)
    touched_relevant = latent_batch(config, 4, 6, relevant=half)
    touched_relevant.latents[:2] = torch.randn_like(touched_relevant.latents[:2]).tanh()
    touched_uniform = latent_batch(config, 4, 6, relevant=half)
    touched_uniform.latents[2:] = torch.randn_like(touched_uniform.latents[2:]).tanh()

    assert torch.equal(loss_of(world, base, config), loss_of(world, touched_relevant, config))
    assert not torch.equal(loss_of(world, base, config), loss_of(world, touched_uniform, config))


def test_direct_commits_both_generated_states(config):
    """Both generated readouts must reach the heads at their own indices. Predicting
    the second latent without committing it leaves the heads trained on one
    generated state while Phase 3 reads them after every generated state."""
    config = replace(config, transition="direct")
    world = world_for(config, "direct").eval()
    blocks = 6
    batch = latent_batch(config, 2, blocks)
    with torch.no_grad():
        _, readout = transition_loss(
            world, batch, torch.Generator().manual_seed(3), config, return_agent=True
        )
        committed, conditioning = commit_inputs(
            batch.latents, torch.Generator().manual_seed(3), config
        )
        real = world(None, batch.led_to_action, committed, conditioning)[1]
    differs = [not torch.equal(readout[:, i], real[:, i]) for i in range(blocks)]
    assert differs == [i >= blocks - 2 for i in range(blocks)]


def test_commit_prefix_does_not_reweight_the_step_grid(config):
    """A prefix row exists to give its last block a runtime-like history. Scoring
    the prefix too would shift the supervised share away from what S67 sets."""
    config = replace(config, transition="flow")
    finest = total = 0
    for seed in range(64):
        conditioning, scored = flow_conditioning(
            torch.Generator().manual_seed(seed), (8, 32), config, "cpu", bootstrap=True
        )
        keep = scored > 0
        total += int(keep.sum())
        finest += int(((conditioning[..., 1] == config.step_index) & keep).sum())
    assert abs(finest / total - (1 - config.self_fraction)) < 0.05


def test_shortcut_supervises_before_bootstrap_start(config):
    """The pinned source keeps every row supervised at d_min until `bootstrap_start`,
    then splits `self_fraction` of *rows* onto coarser steps. Sampling the step per
    position instead leaves 75% of positions chasing an untrained model."""
    config = replace(config, transition="flow", commit_prefix_fraction=0.0)
    before, _ = flow_conditioning(torch.Generator().manual_seed(0), (8, 16), config, "cpu", False)
    after, _ = flow_conditioning(torch.Generator().manual_seed(0), (8, 16), config, "cpu", True)
    assert (before[..., 1] == config.step_index).all(), "warmup must be fully supervised"

    per_row = [(after[r, :, 1] == config.step_index).float().mean().item() for r in range(8)]
    assert all(x in (0.0, 1.0) for x in per_row), "the partition is by row, not by position"
    assert sum(x == 0.0 for x in per_row) == round(config.self_fraction * 8)


def test_commit_prefix_rows_carry_the_commit_condition(config):
    config = replace(config, transition="flow")
    conditioning, scored = flow_conditioning(
        torch.Generator().manual_seed(0), (8, 32), config, "cpu"
    )
    prefix = (scored == 0).any(dim=1)
    assert prefix.any(), "no row carried the rollout prefix"
    for row in prefix.nonzero().flatten():
        assert (conditioning[row, :-1, 0] == config.tau_ctx_index).all()
        assert (conditioning[row, :-1, 1] == config.step_index).all()
        assert bool(scored[row, -1]), "the prefix row's target block must be scored"


@pytest.mark.parametrize("transition", ARMS)
def test_loss_is_finite(config, transition):
    config = replace(config, transition=transition)
    assert torch.isfinite(loss_of(world_for(config, transition), latent_batch(config, 4, 6), config))
