"""Separating check for conditional C2ST: preserve marginals, break only their dependence."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from saccade.eval import gates as G, s1
from saccade import model as M

OUT = Path(__file__).resolve().parent


def neural_c2st(context, real, generated, ep, rng, corruption=None):
    blocks = [context.shape[1] - 322, 256, 17, 49] if context.shape[1] > 322 else [context.shape[1]]
    return s1.c2st(context, real, generated, ep, rng, corruption, blocks)


def tree_c2st(context, real, generated, ep, rng):
    """Historical raw-coordinate baseline retained for the balanced toy control only."""
    import lightgbm as lgb
    X = np.concatenate([np.concatenate([context, real], 1), np.concatenate([context, generated], 1)]).astype(np.float32)
    y = np.repeat([0, 1], len(ep))
    test = G.episode_split(ep, ep, rng)
    model = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 12, "seed": 0, "num_leaves": 31},
                      lgb.Dataset(X[~test], y[~test]), 200)
    return {"accuracy": float(((model.predict(X[test]) > .5) == y[test]).mean()), "n_test": int(test.sum())}


if __name__ == "__main__":
    if "--real-power" in sys.argv:
        ck = torch.load(OUT / "smoke/latest.pt", weights_only=False, map_location="cpu")
        model = M.M0(ck["cfg"]).cuda().eval()
        model.load_state_dict(ck["model"])
        labels = s1.first_episodes(dict(np.load(s1.LABELS / "s0_dev_labels.npz")), 15)
        enc = s1.encode(model, labels, ck["cfg"])
        lens = np.array([len(r["p"]) for r in enc])
        rng = np.random.default_rng(0)
        n = min(30000, int((lens - 1).sum()))
        ep = rng.choice(len(enc), n, p=(lens - 1) / (lens - 1).sum())
        ts = np.array([rng.integers(1, lens[e]) for e in ep])
        context = s1.conditioning_features(enc, ep, ts, 8)
        real = {k: torch.stack([enc[e][kk][t].float() for e, t in zip(ep, ts)])
                for k, kk in (("p", "p"), ("g", "g"), ("v", "v"), ("r", "reward"), ("term", "terminated"))}
        outcome = s1.outcome_features(real)
        result = {kind: neural_c2st(context, outcome, outcome, ep, np.random.default_rng(1), kind)
                  for kind in ("shuffled_history", "mismatched_actions")}
        (OUT / "conditional_classifier_real.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        sys.exit(0)
    rng = np.random.default_rng(0)
    # EXACT matching empirical marginals: class 0 = (x,x), class 1 = (x,-x), x in {-1,+1} balanced per episode.
    x = np.tile(np.array([-1., 1.]), 500)[:, None].astype(np.float16)
    ep = np.repeat(np.arange(100), 10)
    result = {"tree": tree_c2st(x, x, -x, ep, np.random.default_rng(1)),
              "neural": neural_c2st(x, x, -x, ep, np.random.default_rng(1))}
    (OUT / "conditional_classifier.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
