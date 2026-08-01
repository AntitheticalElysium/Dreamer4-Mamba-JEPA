from dataclasses import replace

import torch

from d4mj.train import _checkpoint, _share_initialisation, _generators, optimizer
from d4mj.transition import World


def test_shared_initialisation_matches_every_common_tensor(config):
    """`manual_seed` alone does not: the mixers consume different numbers of draws
    at construction, so everything built after the first time layer diverged."""
    def build(mixer, share):
        arm = replace(config, time_mixer=mixer)
        torch.manual_seed(arm.seed + 1)
        world = World(arm)
        return (_share_initialisation(world, arm) if share else world).state_dict()

    for share, expect_all in ((False, False), (True, True)):
        attention, mamba = build("attention", share), build("mamba", share)
        common = [k for k in attention if k in mamba and attention[k].shape == mamba[k].shape]
        identical = [k for k in common if torch.equal(attention[k], mamba[k])]
        assert common
        assert (len(identical) == len(common)) is expect_all


def test_resume_restores_optimizer_normalisers_and_streams(config, tmp_path):
    """Restoring weights alone restarts every normaliser and replays every window
    and noise draw from zero, which is not a resume."""
    path = tmp_path / "phase.pt"
    torch.manual_seed(config.seed + 1)
    world = World(config)
    optimiser = optimizer([world], config)
    balance = {"dynamics": 4.0}
    sampler, rng = _generators(config, 1)

    world.readout.weight.data.add_(0.5)
    for group in optimiser.param_groups:
        group["lr"] = 1e-9
    sampler.manual_seed(99)
    saved_sampler = sampler.get_state().clone()
    saved_weight = world.readout.weight.detach().clone()
    _checkpoint(path, config, [world, optimiser], balance, sampler, rng, step=7)

    torch.manual_seed(config.seed + 1)
    other = World(config)
    other_optimiser = optimizer([other], config)
    other_balance: dict[str, float] = {}
    other_sampler, other_rng = _generators(config, 1)
    step = _checkpoint(path, config, [other, other_optimiser], other_balance, other_sampler, other_rng)

    assert step == 7
    assert other_balance == balance
    assert torch.equal(other.readout.weight.detach(), saved_weight)
    assert torch.equal(other_sampler.get_state(), saved_sampler)


def test_resume_from_absent_checkpoint_starts_at_zero(config, tmp_path):
    torch.manual_seed(0)
    world = World(config)
    sampler, rng = _generators(config, 1)
    missing = tmp_path / "absent.pt"
    assert _checkpoint(missing, config, [world], {}, sampler, rng) == 0
    assert _checkpoint(None, config, [world], {}, sampler, rng) == 0
