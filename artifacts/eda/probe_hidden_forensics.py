"""Is the hidden -> pre_tanh drop real, and if so what does hidden actually carry?

Part 1 exists because the original path probe's "matched capacity" claim is false in
parameter terms. Both depths use Linear(width, 64), but hidden is 32x256 = 8192 wide
and pre_tanh is 32x32 = 1024, so the hidden probe carries 524,288 first-layer weights
against 65,536 -- eight times more. Part of the reported drop may be the probe getting
weaker rather than the signal.

Two corrections to a first pass of this gate, which tested a different object than the
localization it was checking:

  centring       the localization table is the root-centred one (hidden_centred 0.8438
                 against pre_tanh_centred 0.7401). Within-state AUC removes
                 root-constant scores, but a nonlinear probe can still use uncentred
                 root context to modulate action features, so both depths are centred
                 over the 17 candidates before projection.
  conditioning   a random *square* Gaussian preserves rank but worsens conditioning, and
                 handicaps pre_tanh for nothing. pre_tanh now keeps its exact identity
                 representation and only hidden is projected down, through an
                 orthonormal (QR) map. That favours pre_tanh, so a surviving gap is
                 conservative.

Controls:

  action_only      one-hot action alone, so we can see how much of any score is the
                   global action prior rather than state-conditional structure
  action_shuffled  hidden vectors permuted across the 17 candidates within each root,
                   breaking the action-to-feature correspondence while leaving all
                   root-level content intact

Part 2 splits each hidden token by the SVD of the terminal projection into the 32
directions it can see and the 224 it cannot, and asks where the extractable signal
lives. Part 3 asks whether that signal can actually correct the successor.

Read-only throughout: existing checkpoints, existing caches, existing root split.
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
from evaluate_damage_classifier import interval
from evaluate_phase1b_fork import Readout
from probe_hidden_ceiling import cache_hidden
from reevaluate_phase1b_delta import within_state
from train_phase1b_fork import MixerWorld, fork_actions, load_forkset, seed_split

from d4mj.checkpoint import load
from d4mj.config import Config

DEVICE = "cuda"


class LinearProbe(nn.Module):
    """Parameter-matched second reading: no hidden layer at all."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Linear(width, 1)

    def forward(self, x):
        return self.net(x)[:, 0]


def standardise(x, fit_mask):
    flat = x[fit_mask].reshape(-1, x.shape[-1]).float()
    mean, std = flat.mean(0), flat.std(0).clamp(min=1e-6)
    return ((x.float() - mean) / std).half()


def fit_and_score(x, labels, masks, seed, linear=False, epochs=40):
    """Fit on fit roots, select the epoch on tune, read once on test."""
    fit_m, tune_m, test_m = masks
    width = x.shape[-1]
    torch.manual_seed(seed)
    model = (LinearProbe if linear else Readout)(width).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    y = torch.from_numpy(labels).float()
    positives = float(y[fit_m].sum())
    weight = torch.tensor(max(y[fit_m].numel() - positives, 1.0) / max(positives, 1.0),
                          device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    def run(mask):
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(x[mask][lo : lo + 256].to(DEVICE).float()
                                   .reshape(-1, width)).cpu()
                             for lo in range(0, int(mask.sum()), 256)])
        return out.numpy().reshape(-1, 17)

    index = np.arange(int(fit_m.sum()))
    xf, yf = x[fit_m], y[fit_m]
    best = {"auc": -1.0, "state": None}
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(index)
        for lo in range(0, len(order), 256):
            pick = torch.from_numpy(order[lo : lo + 256])
            opt.zero_grad()
            criterion(model(xf[pick].to(DEVICE).float().reshape(-1, width)),
                      yf[pick].to(DEVICE).reshape(-1)).backward()
            opt.step()
        value = float(np.mean(within_state(run(tune_m), labels[tune_m.numpy()])))
        if value > best["auc"]:
            best = {"auc": value, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    values = within_state(run(test_m), labels[test_m.numpy()])
    mean, (lo, hi) = interval(values, 17)
    return {"test_auc": mean, "ci": [lo, hi], "tune_auc": best["auc"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="abm0")
    parser.add_argument("--milestone", type=int, default=20000)
    parser.add_argument("--widths", type=int, nargs="+", default=(256, 512, 1024))
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260824, 7, 101))
    args = parser.parse_args()

    rows = load_forkset(HERE / f"forkset_{args.suffix}_n64")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    history = torch.stack([r["z_history"] for r in rows])
    labels = torch.stack([r["label"] for r in rows]).numpy()
    masks = tuple(torch.from_numpy(splits == s) for s in ("fit", "tune", "test"))

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=64, d_bottleneck=16, seed=Config().seed)
    world = MixerWorld(config).to(DEVICE)
    load(HERE / f"phase1b_{args.suffix}_n64" / f"world_{args.milestone:06d}.pt",
         config, part0=world)
    world.eval()
    for p in world.parameters():
        p.requires_grad_(False)

    raw = cache_hidden(world, config, history, fork_actions(rows))
    with torch.no_grad():
        hidden = torch.cat([world.mix_norm(raw[lo : lo + 64].to(DEVICE).float()).half().cpu()
                            for lo in range(0, len(raw), 64)])
        pre = torch.cat([world.readout[2](hidden[lo : lo + 64].to(DEVICE).float()).half().cpu()
                         for lo in range(0, len(hidden), 64)])
    depths = {"hidden": hidden.flatten(2), "pre_tanh": pre.flatten(2)}
    print(f"{args.suffix} @ {args.milestone}: " +
          "  ".join(f"{k} {tuple(v.shape)}" for k, v in depths.items()), flush=True)

    fit_m = masks[0]
    centred = {name: (x.float() - x.float().mean(1, keepdim=True)).half()
               for name, x in depths.items()}

    generator = torch.Generator().manual_seed(args.seeds[0])
    gauss = torch.randn(centred["hidden"].shape[-1], 1024, generator=generator)
    ortho = torch.linalg.qr(gauss)[0].half()          # orthonormal columns, 8192 -> 1024
    inputs = {
        "hidden_centred": standardise(centred["hidden"] @ ortho, fit_m),
        "pre_tanh_centred": standardise(centred["pre_tanh"], fit_m),   # identity, exact
    }

    result = {"arm": args.suffix, "milestone": args.milestone,
              "projection": "orthonormal 8192->1024 for hidden, identity for pre_tanh",
              "centred": True, "gate": {}, "controls": {}}
    for name, x in inputs.items():
        for linear in (False, True):
            key = f"{name}|{'linear' if linear else 'mlp'}"
            result["gate"][key] = fit_and_score(x, labels, masks, seed=11, linear=linear)
            r = result["gate"][key]
            print(f"  {key:<30} AUC {r['test_auc']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]",
                  flush=True)

    onehot = torch.eye(17).expand(len(labels), 17, 17).clone().half()
    for linear in (False, True):
        key = f"action_only|{'linear' if linear else 'mlp'}"
        result["controls"][key] = fit_and_score(onehot, labels, masks, seed=11, linear=linear)
    order = torch.stack([torch.randperm(17, generator=generator) for _ in range(len(labels))])
    shuffled = torch.gather(centred["hidden"], 1,
                            order[..., None].expand_as(centred["hidden"]))
    result["controls"]["action_shuffled|mlp"] = fit_and_score(
        standardise(shuffled @ ortho, fit_m), labels, masks, seed=11)
    for k, v in result["controls"].items():
        print(f"  control {k:<22} AUC {v['test_auc']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]",
              flush=True)

    (HERE / f"hidden_forensics_{args.suffix}_{args.milestone:06d}.json").write_text(
        json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------
# Part 2/3: where the action-specific structure lives, and whether it can correct the
# successor. Split every centred hidden token by the SVD of the trained terminal
# projection W (32 x 256): 32 directions W can see, 224 it discards.
# ---------------------------------------------------------------------------------


def row_null_bases(weight):
    """Orthonormal bases for W's row space and its null space, from the SVD."""
    _, _, vh = torch.linalg.svd(weight.float(), full_matrices=True)   # vh is 256 x 256
    rank = weight.shape[0]
    return vh[:rank].T.contiguous(), vh[rank:].T.contiguous()         # 256x32, 256x224


class Residual(nn.Module):
    """Diagnostic extractor of the action-conditioned successor error, per token."""

    def __init__(self, width: int, out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def fit_residual(x, target, masks, seed, epochs=30, batch=64):
    """Predict e from a hidden view; return the held-out prediction."""
    fit_m, tune_m, test_m = masks
    torch.manual_seed(seed)
    model = Residual(x.shape[-1], target.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    xf, tf = x[fit_m], target[fit_m]
    best = {"loss": float("inf"), "state": None}

    def run(mask):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(x[mask][lo:lo+batch].to(DEVICE).float()).cpu()
                              for lo in range(0, int(mask.sum()), batch)])

    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(int(fit_m.sum()))
        for lo in range(0, len(order), batch):
            pick = torch.from_numpy(order[lo:lo+batch])
            opt.zero_grad()
            (model(xf[pick].to(DEVICE).float()) - tf[pick].to(DEVICE).float()).pow(2).mean().backward()
            opt.step()
        value = float((run(tune_m) - target[tune_m].float()).pow(2).mean())
        if value < best["loss"]:
            best = {"loss": value, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return run(test_m), best["loss"]
