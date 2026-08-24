"""Is the hidden -> output drop a repairable terminal-decoder loss, or something else?

The mixer arms show a large hidden -> pre_tanh compression (-0.134 and -0.118) while
producing much better successors than the controls. That combination already proves
probe-drop size is not itself a measure of reconstruction failure -- the path probe
measures probe-extractable structure, not information. And pre_tanh's centred AUC
(0.740, 0.739) sits essentially at the true-delta ceiling (0.7405), so the hidden state
is *more* damage-decodable than the target successor requires. Some of what disappears
may be task-relevant detail the deterministic latent target does not contain.

So this asks the causal question directly, on the existing 20k checkpoints, by training
successor-only readouts over cached post-mixer hidden tokens.

    H_raw    the mixer's spatial tokens before mix_norm
    H_norm   mix_norm(H_raw), which is what the path probe calls `hidden`

  existing      the trained Linear(256, 32) + tanh                  reference only
  fresh         same shape, refit from scratch on frozen H_norm     optimisation/co-adaptation?
  d4            H_raw -> zero-init Linear(256, 32), unsquashed      source-shaped endpoint
  pointwise     H_norm -> 256 -> 256 -> 32 + tanh, no mixing        nonlinearly embedded per token?
  interaction   H_raw -> one pre-norm block -> Linear + tanh        needs more cross-token refinement?

The two oracles are diagnostics, not proposed components: an invented MLP is a fine
information test and a poor thing to ship without source justification.

Trained against the true successor only. Damage labels are evaluation-only.

This is not answered by `probe_decoder_ceiling.py`, which cached pooled and pre-pool
*backbone* features -- upstream of the mixer hidden state this file is about.
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
from probe_action_matching import classes_of
from probe_decoder_ceiling import decompose
from reevaluate_phase1b_delta import fit_probe, within_state
from train_phase1b_fork import MixerWorld, fork_actions, load_forkset, seed_split

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import commit_inputs

DEVICE = "cuda"


def build(kind: str, config, world):
    """Each arm consumes H_raw; those wanting H_norm apply the frozen mix_norm first."""
    d, out = config.d_model, config.d_spatial
    if kind == "fresh":
        return nn.Sequential(world.mix_norm, nn.Linear(d, out), nn.Tanh())
    if kind == "d4":
        layer = nn.Linear(d, out)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return nn.Sequential(layer)                      # unsquashed, per the source endpoint
    if kind == "pointwise":
        return nn.Sequential(world.mix_norm, nn.Linear(d, d), nn.GELU(),
                             nn.Linear(d, out), nn.Tanh())
    if kind == "interaction":
        return Interaction(config)
    raise ValueError(kind)


class Interaction(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.block = nn.TransformerEncoderLayer(d, config.n_heads, 4 * d, dropout=0.0,
                                                 batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, config.d_spatial)

    def forward(self, x):
        b, s, d = x.shape
        return torch.tanh(self.out(self.norm(self.block(x))))


@torch.no_grad()
def cache_hidden(world, config, history, led_history, batch=4):
    """H_raw for all 17 candidate actions at every root, before mix_norm."""
    out = []
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    spatial, d = config.n_spatial, config.d_spatial
    actions = torch.arange(17, device=DEVICE)
    for lo in range(0, len(history), batch):
        z = history[lo : lo + batch].to(DEVICE)
        n, steps = z.shape[0], z.shape[1]
        led = led_history[lo : lo + batch].to(DEVICE)
        committed, conditioning = commit_inputs(z.view(n, steps, spatial, d), rng, config)
        features, _, _ = world(None, led, committed, conditioning)
        last = features[:, -1:]
        wide = last.expand(n, 17, *last.shape[2:])
        act = actions[None].expand(n, -1)
        block = torch.cat([wide[:, :, world.spatial], wide[:, :, world.register]], dim=2)
        pooled = world.pool(block.transpose(2, 3)).transpose(2, 3)
        b, t, s, dm = pooled.shape
        token = torch.cat([world.action_embed(act)[:, :, None], pooled], dim=2)
        token = world.mixer(token.reshape(b * t, s + 1, dm)).view(b, t, s + 1, dm)
        out.append(token[:, :, 1:].half().cpu())
    return torch.cat(out)


def score(pred, truth, root, labels_test, splits, branch, rows, tag):
    result = {"arm": tag, "error": decompose(pred, truth, root)}
    hat, eff = pred - pred.mean(1, keepdim=True), truth - truth.mean(1, keepdim=True)
    nse = ((hat - eff).pow(2).sum(-1) / eff.pow(2).sum(-1).clamp(min=1e-12)).mean(1)
    cos = torch.nn.functional.cosine_similarity(hat.reshape(-1, hat.shape[-1]),
                                                eff.reshape(-1, eff.shape[-1]), dim=1)
    cross = torch.cdist(hat, eff).pow(2).numpy()
    gram, pgram = torch.cdist(eff, eff).pow(2).numpy(), torch.cdist(hat, hat).pow(2).numpy()
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
    result.update({"nse": float(nse.mean()), "cosine": float(cos.mean()),
                   "retrieval": float(np.mean(hits)),
                   "geometry": float(np.mean(geo)),
                   "outside_unit": float((pred.abs() > 1.0).float().mean())})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="abm0")
    parser.add_argument("--milestone", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    rows = load_forkset(HERE / f"forkset_{args.suffix}_n64")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows]).float()
    labels = torch.stack([r["label"] for r in rows]).numpy()
    root = history[:, -1].float()

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=64, d_bottleneck=16, seed=Config().seed)
    world = MixerWorld(config).to(DEVICE)
    load(HERE / f"phase1b_{args.suffix}_n64" / f"world_{args.milestone:06d}.pt",
         config, part0=world)
    world.eval()
    for p in world.parameters():
        p.requires_grad_(False)

    hidden = cache_hidden(world, config, history, fork_actions(rows))
    print(f"H_raw {tuple(hidden.shape)} cached from {args.suffix} @ {args.milestone}", flush=True)

    fit = np.where(splits == "fit")[0]
    test = np.where(splits == "test")[0]
    truth, r0 = branch[test], root[test]
    labels_test = labels[test]
    results = {}

    # reference: the trained head itself
    with torch.no_grad():
        existing = torch.cat([
            torch.tanh(world.readout(world.mix_norm(hidden[test][lo:lo+16].to(DEVICE).float())))
            .flatten(2).cpu() for lo in range(0, len(test), 16)])
    predictions = {"existing": existing}
    results["existing"] = score(existing, truth, r0, labels_test, splits, branch, rows, "existing")

    for kind in ("fresh", "d4", "pointwise", "interaction"):
        torch.manual_seed(config.seed + 11)
        model = build(kind, config, world).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        draw = np.random.default_rng(config.seed + 91)
        started, curve = time.time(), []
        for step in range(args.steps):
            pick = draw.choice(fit, args.batch, replace=False)
            x = hidden[pick].to(DEVICE).float().reshape(-1, config.n_spatial, config.d_model)
            target = branch[pick].to(DEVICE).reshape(-1, config.n_spatial * config.d_spatial)
            loss = (model(x).flatten(1) - target).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
            curve.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            pred = torch.cat([
                model(hidden[test][lo:lo+16].to(DEVICE).float()
                      .reshape(-1, config.n_spatial, config.d_model))
                .flatten(1).view(-1, 17, config.n_spatial * config.d_spatial).cpu()
                for lo in range(0, len(test), 16)])
        predictions[kind] = pred
        results[kind] = score(pred, truth, r0, labels_test, splits, branch, rows, kind)
        results[kind]["train_tail"] = sum(curve[-200:]) / 200
        print(f"  {kind} trained in {time.time()-started:.0f}s", flush=True)

    # frozen damage probe, fitted on true dz, applied to every arm
    width = branch.shape[-1]
    delta_true = branch - root[:, None]
    fm, tm = torch.from_numpy(splits == "fit"), torch.from_numpy(splits == "tune")
    probe, _ = fit_probe(delta_true[fm].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(labels[splits == "fit"].reshape(-1)).float().to(DEVICE),
                         delta_true[tm].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(labels[splits == "tune"].reshape(-1)).float().to(DEVICE),
                         seed=11)

    def read(mix):
        with torch.no_grad():
            s = torch.cat([probe(mix[lo:lo+128].reshape(-1, width).to(DEVICE)).cpu()
                           for lo in range(0, len(mix), 128)]).numpy().reshape(-1, 17)
        return float(np.mean(within_state(s, labels_test)))

    auc_true = read(delta_true[torch.from_numpy(splits == "test")])
    for kind, pred in predictions.items():
        auc = read(pred - r0[:, None])
        results[kind]["auc_pred"] = auc
        results[kind]["R_delta"] = (auc - 0.5) / (auc_true - 0.5)
    results["auc_true"] = auc_true

    print(f"\narm {args.suffix} @ {args.milestone}  (true dz AUC {auc_true:.4f})")
    print(f"  {'arm':<13}{'total':>9}{'action':>9}{'NSE':>8}{'cosine':>8}"
          f"{'retr':>8}{'geom':>8}{'R_delta':>9}{'>|1|':>8}")
    for kind in ("existing", "fresh", "d4", "pointwise", "interaction"):
        r = results[kind]
        print(f"  {kind:<13}{r['error']['total']:9.5f}{r['error']['action']:9.5f}"
              f"{r['nse']:8.4f}{r['cosine']:8.4f}{r['retrieval']:8.4f}{r['geometry']:8.4f}"
              f"{r['R_delta']:9.3f}{r['outside_unit']:8.2%}")
    (HERE / f"hidden_ceiling_{args.suffix}_{args.milestone:06d}.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
