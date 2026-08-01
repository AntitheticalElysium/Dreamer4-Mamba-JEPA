import pytest
import torch

from d4mj.data import FORMAT, Episode, load_episodes, sample_batch, save_episodes

from .conftest import episode, window_start


def test_pretraining_does_not_stratify(config, episodes):
    """D4 pretrains on the whole corpus; only agent finetuning applies the mixture."""
    assert sample_batch(episodes, torch.Generator().manual_seed(0), config).relevant is None


def test_mixture_is_half_and_holds_the_whole_event_transition(config):
    """`events[e]` says action e caused an achievement arriving at observation
    e + 1, so a relevant window must hold both. Requiring only observation e lets
    the BC target that depends on the arrival fall outside the window."""
    one = [episode(0, config)]
    event = len(one[0]) // 2
    batch = sample_batch(one, torch.Generator().manual_seed(0), config, mixture=True)
    blocks = batch.led_to_action.shape[1]
    assert int(batch.relevant.sum()) == config.batch // 2
    for row in range(config.batch):
        if bool(batch.relevant[row]):
            start = window_start(batch, row)
            assert start <= event, "the outgoing action precedes the window"
            assert event + 1 <= start + blocks - 1, "the achievement arrives after the window"


def test_mixture_without_events_raises(config):
    """A missing pool must fail rather than silently borrow the other one."""
    source = episode(0, config)
    plain = [
        Episode(
            observations=source.observations,
            actions_taken=source.actions_taken,
            rewards=source.rewards,
            terminated=source.terminated,
            truncated=source.truncated,
        )
    ]
    with pytest.raises(ValueError, match="task events"):
        sample_batch(plain, torch.Generator().manual_seed(0), config, mixture=True)


def test_episode_start_windows_score_every_block(config, episodes):
    """The first `receptive_field - 1` states of an episode are what deployment
    begins from. A single burn-in integer masked them out of every batch."""
    batch = sample_batch(episodes, torch.Generator().manual_seed(0), config)
    starts = [window_start(batch, row) for row in range(config.batch)]
    assert 0 in starts, "no window began at an episode start"
    for row, start in enumerate(starts):
        if start == 0:
            assert batch.scored[row].all()
        else:
            assert not batch.scored[row][: config.burn_in].any()
            assert batch.scored[row][config.burn_in :].all()


def test_led_to_convention_holds(config, episodes):
    """Block i carries the action that produced its observation and the reward that
    arrived with it; only a true episode start is invalid."""
    batch = sample_batch(episodes, torch.Generator().manual_seed(1), config)
    for row in range(config.batch):
        steps = torch.arange(window_start(batch, row), window_start(batch, row) + batch.led_to_action.shape[1])
        exists = steps > 0
        source = (steps - 1).clamp(min=0)
        assert torch.equal(batch.valid[row], exists)
        assert torch.equal(batch.reward[row][exists], source[exists].float())
        assert torch.equal(batch.led_to_action[row][exists], source[exists] % config.n_actions)
        assert (batch.led_to_action[row][~exists] == config.n_actions).all()


def test_round_trip_preserves_events(config, tmp_path):
    path = tmp_path / "episodes.pt"
    original = [episode(0, config, length=40)]
    save_episodes(path, original)
    restored = load_episodes(path)
    assert torch.equal(restored[0].events, original[0].events)
    assert torch.equal(restored[0].observations, original[0].observations)


def test_format_version_is_current(config, tmp_path):
    """Each field that changed what an episode *means* -- `events`, then the split
    of eligibility from events -- bumps the format, so an older file cannot load as
    if the new fields had simply defaulted."""
    path = tmp_path / "episodes.pt"
    save_episodes(path, [episode(0, config, length=40)])
    assert torch.load(path, weights_only=False)["format"] == FORMAT == "d4mj_episodes_v3"
