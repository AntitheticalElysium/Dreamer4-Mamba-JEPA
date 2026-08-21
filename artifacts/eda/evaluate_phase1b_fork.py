"""Does damage emerge in the *predicted* successor?

The world model was never told which action damages. So the test is: fit a damage
readout on TRUE successor latents from fit roots, freeze it, and apply it to the
world's PREDICTED successors on held-out S82 seeds. If the representation and
dynamics carry consequence, a probe trained only on real latents should read it off
generated ones.

Reported per arm: within-state AUC from predicted successors (primary), the same
probe on true successors (ceiling), and latent MSE of the prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluate_damage_classifier import auc, interval
from train_phase1b_fork import load_forkset, seed_split

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World, commit_inputs

DEVICE = "cuda"


class Readout(nn.Module):
    """The repo's small probe shape, one logit per successor latent."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)[:, 0]


def fit_readout(x, y, seed, epochs=60):
    torch.manual_seed(seed)
    model = Readout(x.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    weight = torch.tensor(float((y <= 0).sum() / y.sum().clamp(min=1)), device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    index = np.arange(len(x))
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(index)
        for lo in range(0, len(order), 512):
            batch = torch.from_numpy(order[lo : lo + 512]).to(DEVICE)
            opt.zero_grad()
            criterion(model(x[batch]), y[batch]).backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def predict_branches(world, config, history, batch=8):
    out = []
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    actions = torch.arange(17, device=DEVICE)
    spatial, d = config.n_spatial, config.d_spatial
    for lo in range(0, len(history), batch):
        z = history[lo : lo + batch].to(DEVICE)
        n, steps = z.shape[0], z.shape[1]
        led = torch.full((n, steps), config.n_actions, dtype=torch.long, device=DEVICE)
        committed, conditioning = commit_inputs(z.view(n, steps, spatial, d), rng, config)
        features, _, _ = world(None, led, committed, conditioning)
        last = features[:, -1:]
        out.append(world.predict(last.expand(n, 17, *last.shape[2:]),
                                 actions[None].expand(n, -1)).flatten(2).cpu())
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", type=int, default=20000)
    parser.add_argument("--suffix", type=str, default="s1")
    args = parser.parse_args()

    print(f"\nPhase-1B translation, {args.suffix} @ {args.milestone} steps")
    print(f"{'arm':<8}{'pred AUC':>11}{'95% CI':>22}{'true AUC':>11}"
          f"{'latent MSE':>13}{'roots':>8}")
    results = {}
    for n_latents in (32, 64):
        folder = HERE / f"forkset_{args.suffix}_n{n_latents}"
        rows = load_forkset(folder)
        splits = np.array([seed_split(r["seed"]) for r in rows])
        history = torch.stack([r["z_history"] for r in rows])
        branch = torch.stack([r["z_branch"] for r in rows])
        labels = torch.stack([r["label"] for r in rows])

        report = json.loads((HERE / "capacity6k" /
                             f"n{n_latents}d16_{args.suffix}" /
                             "training_report.json").read_text())
        config = replace(Config(transition="direct", time_mixer="attention"),
                         n_latents=n_latents, d_bottleneck=16, seed=Config().seed)
        world = World(config).to(DEVICE)
        load(HERE / f"phase1b_{args.suffix}_n{n_latents}" /
             f"world_{args.milestone:06d}.pt", config, part0=world)
        world.eval()

        fit, test = splits == "fit", splits == "test"
        # readout fitted on TRUE successors from fit roots only
        x_fit = branch[fit].reshape(-1, branch.shape[2]).to(DEVICE)
        y_fit = labels[fit].reshape(-1).to(DEVICE)
        readout = fit_readout(x_fit, y_fit, seed=11)

        predicted = predict_branches(world, config, history[test])
        truth = labels[test].numpy()
        mse = float((predicted - branch[test]).pow(2).mean())
        with torch.no_grad():
            scored_pred = readout(predicted.reshape(-1, predicted.shape[2]).to(DEVICE))
            scored_true = readout(branch[test].reshape(-1, branch.shape[2]).to(DEVICE))
        scored_pred = scored_pred.cpu().numpy().reshape(-1, 17)
        scored_true = scored_true.cpu().numpy().reshape(-1, 17)

        def within(scores):
            values = [auc(scores[i], truth[i]) for i in range(len(truth))]
            values = np.array(values)
            return values[~np.isnan(values)]

        v_pred, v_true = within(scored_pred), within(scored_true)
        a, (lo, hi) = interval(v_pred, 17)
        b, _ = interval(v_true, 17)
        print(f"{f'{n_latents}x16':<8}{a:>11.4f}{f'[{lo:.4f}, {hi:.4f}]':>22}"
              f"{b:>11.4f}{mse:>13.5f}{len(v_pred):>8}")
        results[f"{n_latents}x16"] = {"predicted_auc": a, "ci": [lo, hi],
                                      "true_auc": b, "latent_mse": mse,
                                      "roots": int(len(v_pred)),
                                      "values": v_pred.tolist()}
        del world
        torch.cuda.empty_cache()

    if len(results) == 2:
        a = np.array(results["64x16"]["values"])
        b = np.array(results["32x16"]["values"])
        if len(a) == len(b):
            d, (lo, hi) = interval(a - b, 23)
            print(f"\npaired 64x16 minus 32x16, predicted-successor AUC: "
                  f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]   <-- PRIMARY")
            results["paired"] = {"delta": d, "ci": [lo, hi]}
    (HERE / f"phase1b_report_{args.suffix}_{args.milestone:06d}.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
