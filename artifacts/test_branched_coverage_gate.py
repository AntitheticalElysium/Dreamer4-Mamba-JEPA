import torch

from artifacts.probe_branched_policy_states import (
    cluster_means,
    flatten,
    paired_interval,
    shuffled_actions,
    state_auc_rows,
)


def test_flatten_keeps_action_target_alignment():
    features = {
        "state": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "target": torch.tensor([[False, True, False], [True, False, True]]),
    }
    rows = flatten(features)
    assert rows["action"].tolist() == [0, 1, 2, 0, 1, 2]
    assert rows["label"].tolist() == [False, True, False, True, False, True]
    assert rows["group"].tolist() == [0, 0, 0, 1, 1, 1]
    assert torch.equal(rows["state"][:3], features["state"][0].expand(3, -1))


def test_outcome_class_shuffle_preserves_action_label_marginals():
    action = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    label = torch.tensor([False, False, True, True, False, True, False, True, True])
    shuffled = shuffled_actions(action, label, 7)
    for outcome in (False, True):
        rows = label == outcome
        assert torch.equal(action[rows].sort().values, shuffled[rows].sort().values)


def test_state_auc_and_cluster_bootstrap_use_whole_pairs():
    logits = torch.tensor([0.0, 1.0, 0.8, 0.2])
    labels = torch.tensor([False, True, True, False])
    groups = torch.tensor([0, 0, 1, 1])
    per_state = state_auc_rows(logits, labels, groups)
    assert per_state.tolist() == [1.0, 1.0]

    values = torch.tensor([0.2, 0.4, 0.8, 1.0])
    pair = torch.tensor([0, 0, 1, 1])
    assert torch.allclose(cluster_means(values, pair), torch.tensor([0.3, 0.9]))
    comparison = paired_interval(values, torch.zeros_like(values), pair, 11, draws=500)
    assert abs(comparison["difference"] - 0.6) < 1e-6

