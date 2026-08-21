"""Step 2 evaluation: score all 17 actions at fixed fork roots against simulator truth.

Three held-out sets, each scored the same way: the classifier consumes the root's
committed latent history under the led-to convention, and every candidate action is
read off the same final block's features -- so nothing but the action changes
between the 17 scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import hashlib

import corpus
from train_damage_classifier import DamageHead, predict_logits


def split_of(record) -> str:
    key = f"consequence-probe:{record['shard']}:{record['slot']}:{record['t']}"
    draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % 10
    return "fit" if draw < 6 else ("tune" if draw < 8 else "test")

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World, commit_inputs

BOOT = 2000


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels > 0, labels <= 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique, inverse = np.unique(scores, return_inverse=True)
    for value in range(len(unique)):
        tie = inverse == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def interval(values: np.ndarray, seed: int):
    if not len(values):
        return float("nan"), (float("nan"), float("nan"))
    generator = np.random.default_rng(seed)
    draws = np.array([values[generator.integers(0, len(values), len(values))].mean()
                      for _ in range(BOOT)])
    return float(values.mean()), (float(np.quantile(draws, 0.025)),
                                  float(np.quantile(draws, 0.975)))


@torch.no_grad()
def score_roots(world, head, config, histories, actions_led):
    """Logits for all 17 actions at the final block of each root's history."""
    out = []
    rng = torch.Generator(device=config.device).manual_seed(config.seed + 4242)
    for history, led in zip(histories, actions_led):
        z = history[None].to(config.device)
        led = led[None].to(config.device)
        committed, conditioning = commit_inputs(z, rng, config)
        features, _, _ = world(None, led, committed, conditioning)
        last = features[:, -1:]
        candidates = torch.arange(17, device=config.device)[None, :]
        logits = predict_logits(world, head, last.expand(1, 17, *last.shape[2:]), candidates)
        out.append(logits[0].cpu().numpy())
    return np.stack(out)


def report(name, scores, labels, seed):
    """Within-state AUC and contrast over roots that have both classes."""
    aucs, contrasts = [], []
    for row in range(len(scores)):
        y = labels[row].astype(float)
        s = scores[row]
        if not (y > 0).any() or not (y <= 0).any():
            continue
        aucs.append(auc(s, y))
        scale = s.std() if s.std() > 0 else 1.0
        contrasts.append((s[y > 0].mean() - s[y <= 0].mean()) / scale)
    aucs, contrasts = np.array(aucs), np.array(contrasts)
    a, (alo, ahi) = interval(aucs, seed)
    c, (clo, chi) = interval(contrasts, seed + 1)
    pooled = auc(scores.reshape(-1), labels.reshape(-1).astype(float))
    print(f"  {name:<28}roots {len(aucs):>5}  within AUC {a:.4f} [{alo:.4f}, {ahi:.4f}]"
          f"  contrast {c:+.4f} [{clo:+.4f}, {chi:+.4f}]  pooled {pooled:.4f}")
    return {"roots": len(aucs), "within_auc": a, "within_auc_ci": [alo, ahi],
            "within_contrast": c, "within_contrast_ci": [clo, chi], "pooled_auc": pooled}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "damage_classifier")
    parser.add_argument("--test-roots-only", action="store_true",
                        help="restrict hazard forks to the whole-root test split, so an "
                             "arm trained on fit roots is scored only where it is honest")
    args = parser.parse_args()

    config = Config(transition="direct", time_mixer="attention")
    world, head = World(config).to(config.device), DamageHead(config).to(config.device)
    load(args.model, config, part0=world, part1=head)
    world.eval()
    head.eval()
    results = {}

    # ---------------------------------------------------- hazard forks (damage truth)
    data = HERE / "latent_forks"
    records = []
    for path in sorted(data.glob("shard-*.pt")):
        records += torch.load(path, weights_only=False)
    rows = corpus.train_rows()
    lookup = {(r["shard"], r["slot"]): i for i, r in enumerate(rows) if r["source"] == "support"}
    cached = {}
    for index, value in corpus.iter_cached_latents():
        cached[index] = value

    histories, leds, labels, kinds = [], [], [], []
    import json as _json

    manifest = _json.loads((corpus.SUPPORT / "manifest.json").read_text())
    action_cache = {}
    for record in records:
        health = record["health"].numpy()
        dead = record["dead"].numpy()
        positives = (health <= -1) | dead
        negatives = (health >= 0) & ~dead
        if not positives.any() or not negatives.any():
            continue
        if args.test_roots_only and split_of(record) != "test":
            continue
        episode = lookup[(record["shard"], record["slot"])]
        if record["shard"] not in action_cache:
            payload = torch.load(corpus.SUPPORT / manifest["shards"][record["shard"]]["file"],
                                 weights_only=False, mmap=True)
            action_cache[record["shard"]] = {
                slot: fields["actions_taken"].numpy()
                for slot, fields in enumerate(payload["episodes"])
            }
            del payload
        acts = action_cache[record["shard"]][record["slot"]]
        t = record["t"]
        start = max(0, t - config.sequence_long + 1)
        z = cached[episode][start : t + 1]
        led = np.concatenate([[config.n_actions] if start == 0 else [acts[start - 1]],
                              acts[start : t]]).astype(np.int64)
        histories.append(z.clone())
        leds.append(torch.from_numpy(led))
        kinds.append(positives.astype(float))
    labels = np.stack(kinds)
    scores = score_roots(world, head, config, histories, leds)
    print(f"\nheld-out evaluation, model {args.model.name}")
    results["hazard_forks_damage"] = report("hazard forks (damage)", scores, labels, 11)

    # ----------------------------------------------- branched and policy forks (death)
    for name, seed in (("branched_965", 21), ("policy_fork_104", 31)):
        path = HERE / "fork_histories" / f"{name}.pt"
        if not path.exists():
            print(f"  {name}: histories missing, skipped")
            continue
        rows_f = torch.load(path, weights_only=False)
        histories = [r["history"] for r in rows_f]
        leds = [r["led_to_action"] for r in rows_f]
        truth = np.stack([r["true_death"].numpy().astype(float) for r in rows_f])
        scores = score_roots(world, head, config, histories, leds)
        results[name] = report(f"{name} (death)", scores, truth, seed)

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "_test_roots" if args.test_roots_only else ""
    (args.out / f"evaluation_{args.model.stem}{suffix}.json").write_text(
        json.dumps(results, indent=2))
    print(f"wrote evaluation_{args.model.stem}{suffix}.json")


if __name__ == "__main__":
    main()
