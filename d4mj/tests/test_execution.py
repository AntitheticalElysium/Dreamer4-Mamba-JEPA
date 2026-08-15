from dataclasses import replace

import torch

from d4mj.execution import Result, evaluate, score


def episode(seed, unlocked, steps=100, reward=0.0, terminated=False):
    flags = tuple(index < unlocked for index in range(22))
    return Result(seed, steps, reward, terminated, not terminated, flags)


def test_score_matches_the_official_definition():
    """Geometric mean of per-achievement rates in percent, exp(mean(log(1+r)))-1."""
    assert abs(score([episode(0, 22)]) - 100.0) < 1e-9
    assert score([episode(0, 0)]) == 0.0
    assert score([]) == 0.0
    half = [episode(i, 11) for i in range(4)]
    rates = torch.tensor([100.0] * 11 + [0.0] * 11, dtype=torch.float64)
    assert abs(score(half) - float(torch.expm1(torch.log1p(rates).mean()))) < 1e-12


def test_score_is_not_the_mean_of_episode_scores():
    """The log makes it nonlinear, so averaging per-episode scores is a different
    statistic -- the reason intervals must be recomputed from raw rows.

    Two episodes with *complementary* halves show it: every achievement is reached
    half the time, so the set scores 50, while each episode alone scores 9.05.
    """
    first = Result(0, 100, 0.0, False, True, tuple(index < 11 for index in range(22)))
    second = Result(1, 100, 0.0, False, True, tuple(index >= 11 for index in range(22)))
    per_episode = (score([first]) + score([second])) / 2
    assert abs(score([first, second]) - 50.0) < 1e-9
    assert abs(score([first, second]) - per_episode) > 40.0


def test_score_rewards_breadth_over_repetition():
    """Two achievements at 50% each beats one at 100% and one never: the geometric
    mean is what makes coverage the objective."""
    broad = [episode(0, 2), episode(1, 2)]
    narrow = [episode(0, 1), episode(1, 1)]
    assert score(broad) > score(narrow)


def test_paired_bootstrap_detects_a_real_gap(config):
    cfg = replace(config, bootstrap=200)
    strong = lambda seed: episode(seed, 12)
    weak = lambda seed: episode(seed, 4)
    report = evaluate({"actor": strong, "bc": weak}, list(range(16)), cfg)
    assert report["actor"]["score"] > report["bc"]["score"]
    assert report["actor"]["versus_bc"]["beats"]
    assert report["actor"]["versus_bc"]["achievements_beats"]
    assert not report["bc"]["versus_actor"]["beats"]


def test_identical_policies_do_not_beat_each_other(config):
    """The decision rule must not fire on noise: two policies with the same
    behaviour have a gap interval straddling zero."""
    cfg = replace(config, bootstrap=200)
    same = lambda seed: episode(seed, 6)
    report = evaluate({"a": same, "b": same}, list(range(16)), cfg)
    assert not report["a"]["versus_b"]["beats"]
    assert not report["a"]["versus_b"]["achievements_beats"]
    low, high = report["a"]["versus_b"]["interval"]
    assert low <= 0.0 <= high
    low, high = report["a"]["versus_b"]["achievements_interval"]
    assert low <= 0.0 <= high


def test_achievement_gate_is_independent_of_geometric_breadth(config):
    """Count and geometric-score gates remain independent."""
    cfg = replace(config, bootstrap=200)

    def narrow(seed):
        return Result(seed, 100, 8.0, True, False, tuple(index < 8 for index in range(22)))

    def broad(seed):
        flags = tuple((index + 6 * seed) % 22 < 6 for index in range(22))
        return Result(seed, 100, 6.0, True, False, flags)

    report = evaluate({"actor": narrow, "bc": broad}, list(range(22)), cfg)
    comparison = report["actor"]["versus_bc"]
    assert comparison["achievements_beats"]
    assert comparison["achievements_gap"] == 2.0
    assert comparison["gap"] < 0.0 and not comparison["beats"]


def test_report_carries_secondary_metrics_and_raw_rows(config):
    cfg = replace(config, bootstrap=50)
    report = evaluate({"actor": lambda s: episode(s, 3, steps=77, reward=2.0, terminated=True)},
                      list(range(8)), cfg)
    entry = report["actor"]
    assert entry["length"] == 77 and entry["reward"] == 2.0 and entry["terminated"] == 1.0
    assert entry["achievements"] == 3
    assert entry["reward_interval"] == (2.0, 2.0)
    assert entry["achievements_interval"] == (3.0, 3.0)
    assert len(entry["rates"]) == 22 and entry["rates"][0] == 1.0 and entry["rates"][21] == 0.0
    assert len(entry["episodes"]) == 8


def test_evaluation_horizon_is_the_native_one(config):
    """The 2500 cap is the collector's, not Craftax's; scoring at a quarter of the
    horizon would measure the cap."""
    assert config.horizon_eval == 10000
