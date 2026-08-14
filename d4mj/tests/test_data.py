from dataclasses import replace

import pytest
import torch

from d4mj.data import (
    FORMAT,
    STORE_FORMAT,
    Episode,
    EpisodeCorpus,
    atomic_manifest,
    load_episodes,
    sample_batch,
    sample_terminal_batch,
    save_episode_shard,
    save_episodes,
)

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


def test_mixture_without_bc_eligible_episodes_raises(config):
    """A missing pool must fail rather than silently borrow the other one."""
    source = episode(0, config)
    support = [
        Episode(
            observations=source.observations,
            actions_taken=source.actions_taken,
            rewards=source.rewards,
            terminated=source.terminated,
            truncated=source.truncated,
            events=source.events,
            bc_eligible=False,
        )
    ]
    with pytest.raises(ValueError, match="BC-eligible"):
        sample_batch(support, torch.Generator().manual_seed(0), config, mixture=True)


def test_behaviour_cloning_does_not_need_events(config):
    """S51 revised: events oversample BC windows, they no longer gate them. An
    expert episode with no achievement at all is still behaviour worth cloning."""
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
    batch = sample_batch(plain, torch.Generator().manual_seed(0), config, mixture=True)
    assert int(batch.relevant.sum()) == config.batch // 2


def test_behaviour_cloning_reaches_ordinary_windows(config, episodes):
    """The defect S65 records: event-only windows left 84.5% of expert behaviour
    unreachable. Every start must now be reachable by some BC row."""
    starts, without_event = set(), 0
    total = 0
    generator = torch.Generator().manual_seed(0)
    for step in range(200):
        batch = sample_batch(episodes, generator, config, step, 0, mixture=True)
        blocks = batch.led_to_action.shape[1]
        for row in range(config.batch):
            if not bool(batch.relevant[row]):
                continue
            start = window_start(batch, row)
            starts.add(start)
            total += 1
            # every probe episode carries its single event at the midpoint
            event = len(episodes[0]) // 2
            if not start <= event <= start + blocks - 1:
                without_event += 1
    assert 0 in starts, "no BC window began at an episode start"
    assert without_event > 0, "every BC window still contained a task event"
    assert without_event / total > 0.1, f"only {without_event / total:.1%} of BC windows are ordinary"


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


def test_terminal_batch_is_tail_aligned_and_continuation_only(config, episodes):
    source = episodes[0]
    ended = source.terminated.clone()
    ended[-1] = True
    terminal = replace(source, terminated=ended)
    batch = sample_terminal_batch(
        [terminal], torch.Generator().manual_seed(0), config, step=0, total=10
    )
    assert batch.terminated[:, -1].all()
    assert batch.support.all() and not batch.relevant.any()
    assert batch.rows("continuation").all()
    assert not batch.rows("policy").any()
    assert not batch.rows("reward").any()
    assert not batch.rows("dynamics").any()


def test_terminal_support_never_replaces_the_main_mixture(config, episodes):
    source = episodes[0]
    ended = source.terminated.clone()
    ended[-1] = True
    batch = sample_batch(
        [replace(source, terminated=ended)],
        torch.Generator().manual_seed(0),
        config,
        mixture=True,
    )
    assert batch.support is None
    assert int(batch.relevant.sum()) == config.batch // 2


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


def test_sharded_store_is_mmap_loaded_and_preserves_episode_metadata(config, tmp_path):
    root = tmp_path / "store"
    source = replace(
        episode(0, config, length=40),
        epsilon=0.25,
        split="dev",
        episode_id="probe:0",
    )
    record = save_episode_shard(root / "shard-000000.pt", [source])
    atomic_manifest(
        root / "manifest.json",
        {
            "format": STORE_FORMAT,
            "kind": "test",
            "complete": True,
            "episodes": 1,
            "shards": [record],
        },
    )
    restored = load_episodes(root)
    assert isinstance(restored, EpisodeCorpus)
    assert restored[0].epsilon == 0.25
    assert restored[0].split == "dev"
    assert restored[0].episode_id == "probe:0"
    assert torch.equal(restored[0].observations, source.observations)


def test_sharded_store_refuses_incomplete_or_modified_data(config, tmp_path):
    root = tmp_path / "store"
    record = save_episode_shard(root / "shard-000000.pt", [episode(0, config, length=40)])
    manifest = {
        "format": STORE_FORMAT,
        "kind": "test",
        "complete": False,
        "episodes": 1,
        "shards": [record],
    }
    atomic_manifest(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="incomplete"):
        load_episodes(root)
    manifest["complete"] = True
    manifest["shards"][0]["sha256"] = "0" * 64
    atomic_manifest(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_episodes(root)


def test_indexed_corpus_sampling_matches_list_sampling(config, episodes):
    first = sample_batch(episodes, torch.Generator().manual_seed(44), config)
    second = sample_batch(EpisodeCorpus(episodes), torch.Generator().manual_seed(44), config)
    for name in ("led_to_action", "reward", "valid", "scored", "patches"):
        assert torch.equal(getattr(first, name), getattr(second, name))
