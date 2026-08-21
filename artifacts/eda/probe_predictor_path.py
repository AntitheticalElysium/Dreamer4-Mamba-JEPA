"""Where along the Direct path does the actionable signal die?

Four points on the production path, probed with one topology and one split, plus
two controls that change only how the candidate action enters:

  A  z_t                      the committed root latent
  B  z_{t-7..t}               the same, with history
  C  f_t   (pre-pool)         backbone spatial+register tokens
  D  pool(f_t)                after the action-independent 20->16 token mix
  E  pool(f_t), production conditioning   broadcast embedding + shared token-wise MLP
  F  f_t + action token, one cross-token attention layer   action before mixing
  G  the production readout itself, frozen

A-D condition on the action with a per-action output head, so they hold conditioning
fixed while the feature point varies. D vs E then isolates the conditioning mechanism
at a fixed feature point, and F is the early-action positive control.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluate_damage_classifier import auc, interval
from probe_observability import Probe, seed_split
from train_damage_classifier import DamageHead, predict_logits

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World, commit_inputs

DEVICE = "cuda"
HISTORY = 8
config = Config(transition="direct", time_mixer="attention")


@torch.no_grad()
def extract():
    world, head = World(config).to(DEVICE), DamageHead(config).to(DEVICE)
    load(HERE / "damage_classifier/model_020000.pt", config, part0=world, part1=head)
    world.eval(); head.eval()
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    rows = []
    started = time.time()
    paths = sorted((HERE / "branched_damage").glob("seed-*.pt"))
    for n, path in enumerate(paths):
        payload = torch.load(path, weights_only=False)
        seed = int(payload["seed"])
        latents, led = payload["latents"], payload["led_to_action"]
        for row in payload["rows"]:
            label = ((row["health"].numpy() <= -1) | row["dead"].numpy())
            if not (label.any() and not label.all()):
                continue
            t = int(row["step"])
            start = max(0, t - config.sequence_long + 1)
            z = latents[start : t + 1][None].to(DEVICE)
            a = led[start : t + 1][None].to(DEVICE)
            committed, conditioning = commit_inputs(z, rng, config)
            features, _, _ = world(None, a, committed, conditioning)
            last = features[:, -1:]
            tokens = torch.cat([last[:, :, world.spatial], last[:, :, world.register]], dim=2)
            pooled = world.pool(tokens.transpose(2, 3)).transpose(2, 3)
            production = predict_logits(
                world, head, last.expand(1, 17, *last.shape[2:]),
                torch.arange(17, device=DEVICE)[None])
            depth = min(HISTORY, latents.shape[0] - start if t + 1 - start > 0 else 1)
            window = latents[t + 1 - depth : t + 1].reshape(depth, -1)
            if depth < HISTORY:
                window = torch.cat([window[:1].expand(HISTORY - depth, -1), window])
            rows.append({
                "seed": seed, "step": t, "label": torch.from_numpy(label.astype(np.float32)),
                "z": latents[t].reshape(-1).clone(),
                "history": window.clone(),
                "tokens": tokens[0, 0].cpu().clone(),
                "pooled": pooled[0, 0].cpu().clone(),
                "production": production[0].cpu().clone(),
            })
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(paths)} seeds, {len(rows)} roots "
                  f"[{time.time()-started:.0f}s]", flush=True)
    torch.save(rows, HERE / "state_features/predictor_path.pt")
    return rows


class ProductionStyle(nn.Module):
    """DamageHead on frozen pooled features: broadcast action embedding, shared MLP."""

    def __init__(self, config: Config):
        super().__init__()
        self.embed = nn.Embedding(config.n_actions + 1, config.d_model)
        self.head = DamageHead(config)

    def forward(self, pooled, action):
        context = self.embed(action)[:, None].expand_as(pooled)
        return self.head(pooled[:, None], context[:, None])[:, 0]


class EarlyAction(nn.Module):
    """The action as a token, mixed with the world tokens by one attention layer."""

    def __init__(self, config: Config):
        super().__init__()
        self.embed = nn.Embedding(config.n_actions + 1, config.d_model)
        self.attention = nn.TransformerEncoderLayer(
            config.d_model, nhead=4, dim_feedforward=config.d_model * 2,
            batch_first=True, dropout=0.0)
        self.out = nn.Linear(config.d_model, 1)

    def forward(self, tokens, action):
        seq = torch.cat([tokens, self.embed(action)[:, None]], dim=1)
        return self.out(self.attention(seq)[:, -1])[:, 0]


def train_pairwise(model, inputs, action_of, labels, splits, seed, epochs=60):
    """For heads that take (features, action) rather than emitting 17 logits."""
    fit, tune, test = (splits == s for s in ("fit", "tune", "test"))
    y = labels.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    weight = torch.tensor(float((y[fit] <= 0).sum() / y[fit].sum().clamp(min=1)), device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    index = np.where(fit)[0]
    best = {"auc": -1.0, "state": None}
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(index)
        for lo in range(0, len(order), 64):
            batch = torch.from_numpy(order[lo : lo + 64])
            x = inputs[batch].to(DEVICE)
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = torch.stack([model(x, action_of[a].expand(len(batch)))
                                  for a in range(17)], dim=1)
            criterion(logits, y[batch]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            idx = torch.from_numpy(np.where(tune)[0])
            x = inputs[idx].to(DEVICE)
            scores = torch.stack([model(x, action_of[a].expand(len(idx)))
                                  for a in range(17)], dim=1).cpu().numpy()
        truth = labels[tune].numpy()
        values = np.array([auc(scores[i], truth[i]) for i in range(len(scores))])
        values = values[~np.isnan(values)]
        if values.mean() > best["auc"]:
            best = {"auc": float(values.mean()),
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"]); model.eval()
    with torch.no_grad():
        idx = torch.from_numpy(np.where(test)[0])
        x = inputs[idx].to(DEVICE)
        scores = torch.stack([model(x, action_of[a].expand(len(idx)))
                              for a in range(17)], dim=1).cpu().numpy()
    truth = labels[test].numpy()
    values = np.array([auc(scores[i], truth[i]) for i in range(len(scores))])
    return values[~np.isnan(values)], best["auc"]


def main() -> None:
    cache = HERE / "state_features/predictor_path.pt"
    rows = torch.load(cache, weights_only=False) if cache.exists() else extract()
    splits = np.array([seed_split(r["seed"]) for r in rows])
    labels = torch.stack([r["label"] for r in rows])
    print(f"{len(rows)} hazard roots  fit {int((splits=='fit').sum())} "
          f"tune {int((splits=='tune').sum())} test {int((splits=='test').sum())}")
    from probe_observability import run

    print(f"\n{'point':<44}{'within AUC':>11}  {'95% CI':<20}")
    results = {}
    results["H_action_only"] = run("H  action only (control)", None, labels, splits, 1)
    results["A_z_root"] = run("A  z_t (root latent)",
                              torch.stack([r["z"] for r in rows]), labels, splits, 2)
    results["B_z_history"] = run(f"B  z_t-{HISTORY-1}..t (history)",
                                 torch.stack([r["history"].reshape(-1) for r in rows]),
                                 labels, splits, 3)
    results["C_tokens"] = run("C  f_t backbone tokens, pre-pool",
                              torch.stack([r["tokens"].reshape(-1) for r in rows]),
                              labels, splits, 4)
    results["D_pooled"] = run("D  pool(f_t), after the 20->16 mix",
                              torch.stack([r["pooled"].reshape(-1) for r in rows]),
                              labels, splits, 5)

    action_of = torch.arange(17, device=DEVICE)
    torch.manual_seed(11)
    values, tune = train_pairwise(ProductionStyle(config).to(DEVICE),
                                  torch.stack([r["pooled"] for r in rows]),
                                  action_of, labels, splits, 11)
    a, (lo, hi) = interval(values, 17)
    print(f"  {'E  pool(f_t), production conditioning':<44}{a:>11.4f}  [{lo:.4f}, {hi:.4f}]"
          f"   tune {tune:.4f}")
    results["E_production_conditioning"] = {"within_auc": a, "ci": [lo, hi]}

    torch.manual_seed(12)
    values, tune = train_pairwise(EarlyAction(config).to(DEVICE),
                                  torch.stack([r["tokens"] for r in rows]),
                                  action_of, labels, splits, 12)
    a, (lo, hi) = interval(values, 17)
    print(f"  {'F  f_t + action token, one attention layer':<44}{a:>11.4f}  "
          f"[{lo:.4f}, {hi:.4f}]   tune {tune:.4f}")
    results["F_early_action"] = {"within_auc": a, "ci": [lo, hi]}

    scores = torch.stack([r["production"] for r in rows]).numpy()
    test = splits == "test"
    truth = labels[test].numpy()
    sub = scores[test]
    values = np.array([auc(sub[i], truth[i]) for i in range(len(sub))])
    values = values[~np.isnan(values)]
    a, (lo, hi) = interval(values, 17)
    print(f"  {'G  the production readout, frozen':<44}{a:>11.4f}  [{lo:.4f}, {hi:.4f}]")
    results["G_production_readout"] = {"within_auc": a, "ci": [lo, hi]}
    (HERE / "state_features/predictor_path_report.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
