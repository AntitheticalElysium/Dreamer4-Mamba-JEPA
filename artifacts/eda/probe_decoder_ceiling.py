"""Is the counterfactual detail present in the frozen backbone, or already gone?

The residual-restoration curve showed that recovering fidelity recovers the mechanic
steeply -- half the residual buys 90% of R_delta -- so reconstruction quality is the
repair target. This asks whether that fidelity is reachable at all from what the
backbone hands the head, by training a deliberately oversized action-conditioned
decoder on cached features and nothing else.

Two taps, because the production path pools *before* the candidate action exists
(`transition.py`, `pooled = self.pool(...)` then `context = action_embed(action)`):

  pooled    the 32 tokens after the learned 36 -> 32 token-axis pool, i.e. exactly
            what the production head sees
  prepool   the 36 spatial and register tokens straight off the backbone

A pooled failure alone cannot distinguish "the backbone never had it" from "the pool
discarded it", which is why both are run.

The decoder is given every advantage the production head lacks -- an action token that
interacts with all spatial tokens through full self-attention, four blocks of it, ~20x
the parameters -- so a failure here is evidence about the features, not the head. It is
trained on successor reconstruction only; damage labels never enter training and are
used solely to score the frozen probe afterwards.

`--memorise` overfits a handful of roots as an optimisation control: if the decoder
cannot drive training error to near zero there, a held-out failure says nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from probe_action_matching import TAU, classes_of
from reevaluate_phase1b_delta import fit_probe, within_state
from train_phase1b_fork import NoTanhWorld, fork_actions, load_forkset, seed_split

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World, commit_inputs

DEVICE = "cuda"


class Decoder(nn.Module):
    """Oversized action-conditioned decoder over cached tokens."""

    def __init__(self, config, tokens: int, depth: int = 4, heads: int = 8):
        super().__init__()
        d = config.d_model
        self.n_spatial = config.n_spatial
        self.action = nn.Embedding(17, d)
        self.position = nn.Parameter(torch.randn(tokens + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, config.d_spatial)

    def forward(self, features, action):
        x = torch.cat([self.action(action)[:, None], features], dim=1) + self.position
        x = self.norm(self.blocks(x))[:, 1 : 1 + self.n_spatial]
        return torch.tanh(self.out(x)).flatten(1)


@torch.no_grad()
def cache(world, config, history, led_history, tap: str, batch=8):
    """Backbone features at the root block, one forward pass over the corpus."""
    out = []
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    spatial, d = config.n_spatial, config.d_spatial
    for lo in range(0, len(history), batch):
        z = history[lo : lo + batch].to(DEVICE)
        n, steps = z.shape[0], z.shape[1]
        led = led_history[lo : lo + batch].to(DEVICE)
        committed, conditioning = commit_inputs(z.view(n, steps, spatial, d), rng, config)
        features, _, _ = world(None, led, committed, conditioning)
        last = features[:, -1:]
        block = torch.cat([last[:, :, world.spatial], last[:, :, world.register]], dim=2)
        if tap == "pooled":
            block = world.pool(block.transpose(2, 3)).transpose(2, 3)
        out.append(block[:, 0].cpu())
    return torch.cat(out)


def decompose(pred, true, root):
    pb, tb = pred.mean(1, keepdim=True), true.mean(1, keepdim=True)
    common = (pb - tb).pow(2).mean().item()
    action = ((pred - pb) - (true - tb)).pow(2).mean().item()
    effect = (true - tb).pow(2)
    return {"total": (pred - true).pow(2).mean().item(), "common": common, "action": action,
            "action_r2": 1 - action * true[0].numel() * len(true) / effect.sum().item(),
            "trivial_root": (root[:, None] - true).pow(2).mean().item()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="abt0")
    parser.add_argument("--tap", choices=("pooled", "prepool"), default="pooled")
    parser.add_argument("--milestone", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--no-tanh", action="store_true")
    parser.add_argument("--memorise", type=int, default=0,
                        help="optimisation control: fit this many roots and report train error")
    args = parser.parse_args()

    rows = load_forkset(HERE / f"forkset_{args.suffix}_n64")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows]).float()
    labels = torch.stack([r["label"] for r in rows]).numpy()
    root = history[:, -1].float()

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=64, d_bottleneck=16, seed=Config().seed)
    world = (NoTanhWorld if args.no_tanh else World)(config).to(DEVICE)
    load(HERE / f"phase1b_{args.suffix}_n64" / f"world_{args.milestone:06d}.pt",
         config, part0=world)
    world.eval()
    features = cache(world, config, history, fork_actions(rows), args.tap)
    print(f"{args.tap} features {tuple(features.shape)} from {args.suffix} @ {args.milestone}",
          flush=True)

    fit = np.where(splits == "fit")[0]
    if args.memorise:
        fit = fit[: args.memorise]
    test = torch.from_numpy(splits == "test")
    torch.manual_seed(config.seed + 7)
    model = Decoder(config, features.shape[1], depth=args.depth).to(DEVICE)
    print(f"decoder {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters, "
          f"{len(fit)} fit roots", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    draw = np.random.default_rng(config.seed + 91)
    actions = torch.arange(17, device=DEVICE)

    started, curve = time.time(), []
    for step in range(args.steps):
        pick = draw.choice(fit, min(args.batch, len(fit)), replace=len(fit) < args.batch)
        f = features[pick].to(DEVICE)
        n = len(pick)
        wide = f[:, None].expand(n, 17, *f.shape[1:]).reshape(n * 17, *f.shape[1:])
        act = actions[None].expand(n, -1).reshape(-1)
        target = branch[pick].to(DEVICE).reshape(n * 17, -1)
        loss = (model(wide, act) - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()
        curve.append(float(loss.detach()))
        if (step + 1) % 500 == 0 or step + 1 == args.steps:
            print(f"  {step+1}/{args.steps} loss={sum(curve[-500:])/len(curve[-500:]):.5f} "
                  f"[{time.time()-started:.0f}s]", flush=True)

    model.eval()
    scored = fit if args.memorise else np.where(test.numpy())[0]
    with torch.no_grad():
        pred = []
        for lo in range(0, len(scored), 16):
            pick = scored[lo : lo + 16]
            f = features[pick].to(DEVICE)
            n = len(pick)
            wide = f[:, None].expand(n, 17, *f.shape[1:]).reshape(n * 17, *f.shape[1:])
            pred.append(model(wide, actions[None].expand(n, -1).reshape(-1))
                        .view(n, 17, -1).cpu())
    pred = torch.cat(pred)
    truth, r0 = branch[scored], root[scored]
    result = {"arm": args.suffix, "tap": args.tap, "steps": args.steps,
              "memorise": args.memorise, "roots": len(scored),
              "error": decompose(pred, truth, r0), "train_tail": sum(curve[-200:]) / 200}
    e = result["error"]
    label = "TRAIN (memorisation control)" if args.memorise else "held-out test"
    print(f"\n  {label}: total {e['total']:.5f}  common {e['common']:.5f}  "
          f"action {e['action']:.5f}  action R2 {e['action_r2']:.4f}  "
          f"trivial {e['trivial_root']:.5f}")

    if not args.memorise:
        hat = pred - pred.mean(1, keepdim=True)
        eff = truth - truth.mean(1, keepdim=True)
        cross = torch.cdist(hat, eff).pow(2).numpy()
        gram = torch.cdist(eff, eff).pow(2).numpy()
        pgram = torch.cdist(hat, hat).pow(2).numpy()
        hits, geo = [], []
        for i in range(len(cross)):
            lab = classes_of(gram[i])
            if lab.max() < 1:
                continue
            best = cross[i].argmin(1)
            hits.append(np.mean([lab[best[a]] == lab[a] for a in range(17)]))
            u = np.triu_indices(17, 1)
            if gram[i][u].std() > 0 and pgram[i][u].std() > 0:
                geo.append(np.corrcoef(gram[i][u], pgram[i][u])[0, 1])
        width = branch.shape[-1]
        delta_true = branch - root[:, None]
        ytr = labels
        fitm, tunem = torch.from_numpy(splits == "fit"), torch.from_numpy(splits == "tune")
        probe, _ = fit_probe(delta_true[fitm].reshape(-1, width).to(DEVICE),
                             torch.from_numpy(ytr[splits == "fit"].reshape(-1)).float().to(DEVICE),
                             delta_true[tunem].reshape(-1, width).to(DEVICE),
                             torch.from_numpy(ytr[splits == "tune"].reshape(-1)).float().to(DEVICE),
                             seed=11)

        def read(mix):
            with torch.no_grad():
                s = torch.cat([probe(mix[lo : lo + 128].reshape(-1, width).to(DEVICE)).cpu()
                               for lo in range(0, len(mix), 128)]).numpy().reshape(-1, 17)
            return float(np.mean(within_state(s, labels[splits == "test"])))

        auc_true = read(delta_true[test])
        auc_pred = read(pred - r0[:, None])
        result.update({"retrieval": float(np.mean(hits)),
                       "geometry_correlation": float(np.mean(geo)),
                       "auc_true": auc_true, "auc_pred": auc_pred,
                       "R_delta": (auc_pred - 0.5) / (auc_true - 0.5)})
        print(f"  retrieval {result['retrieval']:.4f}   geometry {result['geometry_correlation']:.4f}"
              f"   R_delta {result['R_delta']:.3f} (pred dz AUC {auc_pred:.4f})")

    tag = f"mem{args.memorise}" if args.memorise else "heldout"
    (HERE / f"decoder_ceiling_{args.suffix}_{args.tap}_{tag}.json").write_text(
        json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
