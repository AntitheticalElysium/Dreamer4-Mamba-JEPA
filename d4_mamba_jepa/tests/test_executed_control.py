import math

from d4_mamba_jepa.executed_control import _crafter_score


def test_crafter_score_matches_official_geometric_mean_formula():
    episodes = [
        {"achievements": {"a": 1, "b": 0}},
        {"achievements": {"a": 1, "b": 1}},
    ]
    score, rates = _crafter_score(episodes)
    assert rates == {"a": 100.0, "b": 50.0}
    expected = math.exp((math.log1p(100.0) + math.log1p(50.0)) / 2) - 1
    assert abs(score - expected) < 1e-12
