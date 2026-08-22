"""Where inside Direct does the action-conditioned signal disappear?

Six depths along the trained Direct head, every one scored by the same statistic
used everywhere else -- within-state AUC over the 17 simulator-executed successors
at a root, on the held-out split.

    P0  z_t                      the root latent            shared, 17-way probe
    P1  f_t (spatial+register)   world backbone output      shared, 17-way probe
    P2  pooled                   after self.pool            shared, 17-way probe
    P3  hidden                   after readout[0:2], +a_t   per-action, binary probe
    P4  pre-tanh                 after readout[2]           per-action, binary probe
    P5  predicted dz             after tanh, minus z_t      per-action, binary probe
    --  true dz                  simulator truth            per-action, binary probe

P0-P2 are computed before the candidate action exists: Direct builds f_t once and
broadcasts it across all 17 candidates, so those tensors are identical for every
action and a per-action probe there would return chance by construction. The
question they answer is whether the *state's* action-conditional damage map is
present at that depth, which is why they emit all 17 logits from one shared input.
P3-P5 are per-action and take one vector per candidate, and are read twice: raw, and
root-centred (x minus its mean over the 17 candidates). The centred reading is the
primary one -- it is the action effect itself, and it is the only form comparable
across depths, since the raw tensors differ in how much between-root variance they
carry. Every depth is z-scored on fit-split statistics for the same reason.

Both probe families are the repo's existing shapes at matched capacity --
Linear(w, 64), GELU, Linear(64, k) with k = 17 shared and k = 1 per-action -- fitted
on fit roots, selected on tune by the reported metric, then read once on test.

Everything after P0 is a deterministic function of it, so a decline measures
probe-extractable structure, not information.

Also reported without any probe:

  * variance decomposition of the successor targets into between-root and
    within-root (action) parts -- tests whether absolute-successor MSE simply
    swamps the action-conditioned differences;
  * centred action-effect magnitude at P4, P5 and truth -- tests whether the tanh
    compresses the effect;
  * tanh saturation of the pre-activation.
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
from evaluate_phase1b_fork import Readout
from probe_observability import Probe
from reevaluate_phase1b_delta import within_state
from train_phase1b_fork import load_forkset, seed_split

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World, commit_inputs

DEVICE = "cuda"
SHARED = ("z_t", "f_t", "pooled")


@torch.no_grad()
def collect(world, config, history, batch=8):
    """Every depth of predict(), recomputed from the module's own submodules.

    The final tensor is checked against world.predict itself, so the reconstruction
    is verified rather than assumed.
    """
    out = {name: [] for name in ("z_t", "f_t", "pooled", "hidden", "pre_tanh", "post_tanh")}
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
        wide = last.expand(n, 17, *last.shape[2:])
        act = actions[None].expand(n, -1)

        block = torch.cat([last[:, :, world.spatial], last[:, :, world.register]], dim=2)
        pooled = world.pool(block.transpose(2, 3)).transpose(2, 3)          # (n, 1, spatial, d_model)
        wide_pool = pooled.expand(n, 17, *pooled.shape[2:])
        context = world.action_embed(act)[:, :, None].expand_as(wide_pool)
        joined = torch.cat([wide_pool, context], dim=-1)
        hidden = world.readout[1](world.readout[0](joined))
        pre = world.readout[2](hidden)
        post = torch.tanh(pre)
        assert torch.allclose(post, world.predict(wide, act), atol=1e-5), "predict mismatch"

        out["z_t"].append(z[:, -1].cpu())
        out["f_t"].append(block[:, 0].flatten(1).half().cpu())
        out["pooled"].append(pooled[:, 0].flatten(1).half().cpu())
        out["hidden"].append(hidden.flatten(2).half().cpu())
        out["pre_tanh"].append(pre.flatten(2).half().cpu())
        out["post_tanh"].append(post.flatten(2).cpu())
    return {name: torch.cat(chunks) for name, chunks in out.items()}


def standardise(x, fit_mask):
    """Per-dimension z-score from fit-split statistics.

    Depths differ by orders of magnitude in scale and in how much of their variance
    is between-root rather than action-conditioned. Without this a fixed-capacity
    probe measures input conditioning as much as signal -- which is what collapsed
    the `hidden` probe to a constant and let `pred_dz`, an elementwise-monotone
    function of `pre_tanh`, appear to carry more than it does.
    """
    flat = x[fit_mask].reshape(-1, x.shape[-1]).float()
    mean, std = flat.mean(0), flat.std(0).clamp(min=1e-6)
    return ((x.float() - mean) / std).half()


def fit(x_fit, y_fit, x_tune, y_tune, shared: bool, seed: int, epochs=60):
    """Probe fitted on fit roots, epoch chosen on tune by within-state AUC."""
    torch.manual_seed(seed)
    width = x_fit.shape[-1]
    model = (Probe(width) if shared else Readout(width)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    positives = float(y_fit.sum())
    weight = torch.tensor(max(y_fit.numel() - positives, 1.0) / max(positives, 1.0), device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    def score(x, y):
        model.eval()
        with torch.no_grad():
            logits = torch.cat([
                _forward(model, x[lo : lo + 256], shared) for lo in range(0, len(x), 256)])
        return float(np.mean(within_state(logits.cpu().numpy(), y.numpy())))

    best = {"auc": -1.0, "state": None}
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(x_fit))
        for lo in range(0, len(order), 256):
            index = torch.from_numpy(order[lo : lo + 256])
            opt.zero_grad()
            criterion(_forward(model, x_fit[index], shared),
                      y_fit[index].to(DEVICE)).backward()
            opt.step()
        value = score(x_tune, y_tune)
        if value > best["auc"]:
            best = {"auc": value, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model.eval(), best["auc"]


def _forward(model, x, shared: bool):
    """Shared probes emit 17 logits from one vector; per-action probes one each."""
    x = x.to(DEVICE).float()
    if shared:
        return model(x)
    return model(x.reshape(-1, x.shape[-1])).view(x.shape[0], 17)


def report(name, model, x, y, shared, seed):
    logits = torch.cat([_forward(model, x[lo : lo + 256], shared)
                        for lo in range(0, len(x), 256)]).detach().cpu().numpy()
    values = within_state(logits, y.numpy())
    mean, (lo, hi) = interval(values, seed)
    print(f"  {name:<12}{'shared' if shared else 'per-action':<12}"
          f"within AUC {mean:.4f} [{lo:.4f}, {hi:.4f}]  roots {len(values)}", flush=True)
    return {"within_auc": mean, "within_auc_ci": [lo, hi], "roots": int(len(values)),
            "input": "shared" if shared else "per-action", "tune_auc": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-latents", type=int, default=64)
    parser.add_argument("--suffix", type=str, default="s1")
    parser.add_argument("--milestone", type=int, default=20000)
    args = parser.parse_args()

    folder = HERE / f"forkset_{args.suffix}_n{args.n_latents}"
    rows = load_forkset(folder)
    splits = np.array([seed_split(r["seed"]) for r in rows])
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows])
    labels = torch.stack([r["label"] for r in rows])
    root = history[:, -1]

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=args.n_latents, d_bottleneck=16, seed=Config().seed)
    world = World(config).to(DEVICE)
    load(HERE / f"phase1b_{args.suffix}_n{args.n_latents}" /
         f"world_{args.milestone:06d}.pt", config, part0=world)
    world.eval()
    print(f"arm {args.n_latents}x16 {args.suffix} @ {args.milestone}: {len(rows)} roots", flush=True)

    depths = collect(world, config, history)
    depths["pred_dz"] = depths.pop("post_tanh") - root[:, None]
    depths["true_dz"] = branch - root[:, None]

    fit_m, tune_m, test_m = (torch.from_numpy(splits == s) for s in ("fit", "tune", "test"))
    result = {"arm": f"{args.n_latents}x16", "suffix": args.suffix, "milestone": args.milestone,
              "depths": {}}

    order = ["z_t", "f_t", "pooled", "hidden", "pre_tanh", "pred_dz", "true_dz"]
    for seed, name in enumerate(order):
        shared = name in SHARED
        variants = [("", False)] if shared else [("", False), ("_centred", True)]
        for suffix, centre in variants:
            x = depths[name]
            if centre:
                x = (x.float() - x.float().mean(1, keepdim=True))
            x = standardise(x, fit_m)
            probe, tune_auc = fit(x[fit_m], labels[fit_m], x[tune_m], labels[tune_m],
                                  shared, seed=100 + seed)
            entry = report(name + suffix, probe, x[test_m], labels[test_m], shared, 17)
            entry["tune_auc"] = tune_auc
            result["depths"][name + suffix] = entry
            del x

    # ---------------------------------------------------------------- no probe involved
    target = branch[test_m].float()
    predicted = (depths["pred_dz"][test_m] + root[test_m][:, None]).float()

    def split_variance(x):
        """Between-root against within-root (action) variance of the successor."""
        per_root = x.mean(1, keepdim=True)
        within = (x - per_root).pow(2).mean().item()
        between = (per_root - x.mean()).pow(2).mean().item()
        return {"between_root": between, "within_root_action": within,
                "action_fraction": within / (within + between)}

    centred = {"true": target - target.mean(1, keepdim=True),
               "pred": predicted - predicted.mean(1, keepdim=True)}
    pre = depths["pre_tanh"][test_m].float()
    result["variance"] = {"true": split_variance(target), "pred": split_variance(predicted)}
    result["magnitude"] = {
        "true_effect": float(centred["true"].pow(2).sum(-1).sqrt().mean()),
        "pred_effect": float(centred["pred"].pow(2).sum(-1).sqrt().mean()),
        "pre_tanh_effect": float((pre - pre.mean(1, keepdim=True)).pow(2).sum(-1).sqrt().mean()),
    }
    result["saturation"] = {
        "mean_abs_pre_tanh": float(pre.abs().mean()),
        "fraction_above_1": float((pre.abs() > 1.0).float().mean()),
        "fraction_above_2": float((pre.abs() > 2.0).float().mean()),
    }
    v = result["variance"]["true"]
    print(f"\n  successor variance: between-root {v['between_root']:.5f}, "
          f"within-root(action) {v['within_root_action']:.5f} -> action is "
          f"{v['action_fraction']:.2%} of target variance")
    m, s = result["magnitude"], result["saturation"]
    print(f"  centred effect norm: true {m['true_effect']:.4f}  pred {m['pred_effect']:.4f}  "
          f"pre-tanh {m['pre_tanh_effect']:.4f}")
    print(f"  pre-tanh |x|: mean {s['mean_abs_pre_tanh']:.4f}, "
          f">1 {s['fraction_above_1']:.2%}, >2 {s['fraction_above_2']:.2%}")

    (HERE / f"direct_path_{args.suffix}_n{args.n_latents}_{args.milestone:06d}.json").write_text(
        json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
