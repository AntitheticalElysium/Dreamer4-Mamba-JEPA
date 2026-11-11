"""Frozen, episode-held-out binary readout of the next awake decrement."""
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .native import RUNS, dev
from saccade.eval.gates import average_precision
from saccade.train import file_sha256

T = importlib.import_module("artifacts.experiments.20261111_belief_trajectory.trajectory")
HERE = Path(__file__).resolve().parent
NAMES = ("hunger", "thirst", "fatigue")


def outcomes(rows, phase):
    from craftax.craftax_classic.constants import Action

    with np.load(T.S.LABELS / "s0_dev_labels.npz") as labels:
        offset = np.r_[0, np.cumsum(labels["ep_T"] + 1)]
        body = labels["s_body"][:, 1:4].astype(int)
        sleep = labels["s_sleep"]
        loc = {(s, t): i for i, (s, t) in enumerate(zip(labels["ep_shard"], labels["ep_slot"]))}
    states = np.concatenate([offset[loc[e.shard, e.slot]] + t for e, t, *_ in rows])
    action = np.concatenate([a[t] for _, t, a, *_ in rows])
    legal = ~np.isin(action, [Action.DO.value, Action.SLEEP.value])
    valid = np.isfinite(phase) & legal[:, None]
    truth = (body[states + 1] - body[states]) == -1
    due = phase + 1 > np.array([25, 20, 30])
    assert not sleep[states + 1][valid.any(1)].any()
    assert np.array_equal(due[valid], truth[valid])
    return np.where(valid, due, np.nan).astype("float32")


def predict(features, targets, fold, nonlinear):
    x = torch.as_tensor(features, device=dev)
    y = torch.as_tensor(targets, device=dev)
    prediction = np.zeros_like(targets)
    for k in range(4):
        test = fold == k
        val = fold == (k + 1) % 4
        train = ~(test | val) if nonlinear else ~test
        mean = x[train].mean(0)
        std = x[train].std(0).clamp_min(1e-5)
        z = ((x - mean) / std).clamp(-20, 20)
        torch.manual_seed(6820 + k)
        width = z.shape[1]
        if nonlinear:
            net = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 3)).to(dev)
        else:
            net = nn.Linear(width, 3).to(dev)

        def score(ids):
            target = y[ids]
            mask = target.isfinite()
            losses = F.binary_cross_entropy_with_logits(net(z[ids]), target.nan_to_num(), reduction="none")
            return ((losses * mask).sum(0) / mask.sum(0).clamp_min(1)).mean()

        if nonlinear:
            opt = torch.optim.AdamW(net.parameters(), lr=.001)
            best = float("inf")
            ids = torch.as_tensor(np.flatnonzero(train), device=dev)
            for _ in range(24):
                for batch in ids[torch.randperm(len(ids), device=dev)].split(2048):
                    opt.zero_grad(set_to_none=True)
                    score(batch).backward()
                    opt.step()
                with torch.no_grad():
                    val_loss = float(score(val))
                    if val_loss < best:
                        best = val_loss
                        prediction[test] = net(z[test]).sigmoid().cpu().numpy()
        else:
            opt = torch.optim.LBFGS(net.parameters(), max_iter=120, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = score(train) + .001 * net.weight.square().sum()
                loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                prediction[test] = net(z[test]).sigmoid().cpu().numpy()
    return prediction


def compare(y, pred, ref, ep):
    rng = np.random.default_rng(50912)
    result = {}
    for j, name in enumerate(NAMES):
        valid = np.isfinite(y[:, j])
        truth, p, q = y[valid, j], pred[valid, j], ref[valid, j]
        labels, ix = np.unique(ep[valid], return_inverse=True)
        error = np.square(p - truth)
        delta = error - np.square(q - truth)
        n = np.bincount(ix)
        d = np.bincount(ix, weights=delta)
        draw = rng.integers(len(labels), size=(5000, len(labels)))
        result[name] = {
            "n": len(truth),
            "positives": int(truth.sum()),
            "episodes": len(labels),
            "ap": average_precision(truth, p),
            "brier": float(error.mean()),
            "brier_minus_reference": float(delta.mean()),
            "paired_episode_bootstrap_95": np.quantile(d[draw].sum(1) / n[draw].sum(1), [.025, .975]).tolist(),
        }
    return result


def main():
    rows, x, phase, ep, oracle, _, fold = T.study_data()
    assert np.allclose(phase[:, :3], oracle[:, :3], equal_nan=True)
    y = outcomes(rows, phase[:, :3])
    reference = {name: predict(x, y, fold, nonlinear) for name, nonlinear in (("linear", False), ("mlp", True))}
    parent = {}
    report = {"recent": {k: compare(y, p, p, ep) for k, p in reference.items()}, "checkpoints": {}}
    for name, path in RUNS.items():
        _, bp = T.encode(path, rows)
        inputs = np.concatenate([x, bp], 1)
        result = {"sha256": file_sha256(path)}
        for reader, nonlinear in (("linear", False), ("mlp", True)):
            p = predict(inputs, y, fold, nonlinear)
            if name == "parent":
                parent[reader] = p
            result[reader] = {"against_recent": compare(y, p, reference[reader], ep),
                              "against_parent": compare(y, p, parent[reader], ep)}
        report["checkpoints"][name] = result
        (HERE / "DUE.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, "complete", flush=True)


if __name__ == "__main__":
    main()
