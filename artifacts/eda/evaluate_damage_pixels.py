"""Step 3 evaluation: the pixel-input classifier on the same three fork sets."""

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
import corpus
from evaluate_damage_classifier import report
from train_damage_classifier import DamageHead, predict_logits

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.representation import Encoder, pack
from d4mj.transition import World, commit_inputs


@torch.no_grad()
def score_roots(encoder, world, head, config, frames_list, leds):
    out = []
    rng = torch.Generator(device=config.device).manual_seed(config.seed + 4242)
    for frames, led in zip(frames_list, leds):
        patches = patchify(frames[None], config.patch).to(config.device)
        led = led[None].to(config.device)
        z, _, _ = encoder(patches)
        committed, conditioning = commit_inputs(pack(z, config), rng, config)
        features, _, _ = world(None, led, committed, conditioning)
        last = features[:, -1:]
        candidates = torch.arange(17, device=config.device)[None, :]
        logits = predict_logits(world, head, last.expand(1, 17, *last.shape[2:]), candidates)
        out.append(logits[0].cpu().numpy())
    return np.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "damage_pixels")
    args = parser.parse_args()

    config = Config(transition="direct", time_mixer="attention")
    encoder = Encoder(config).to(config.device)
    world = World(config).to(config.device)
    head = DamageHead(config).to(config.device)
    load(args.model, config, part0=encoder, part1=world, part2=head)
    for module in (encoder, world, head):
        module.eval()
    results = {}

    # -------------------------------------------------- hazard forks (damage truth)
    records = []
    for path in sorted((HERE / "latent_forks").glob("shard-*.pt")):
        records += torch.load(path, weights_only=False)
    manifest = json.loads((corpus.SUPPORT / "manifest.json").read_text())
    cache: dict[int, dict] = {}
    frames_list, leds, labels = [], [], []
    for record in records:
        health = record["health"].numpy()
        dead = record["dead"].numpy()
        positives = (health <= -1) | dead
        if not positives.any() or not ((health >= 0) & ~dead).any():
            continue
        shard = record["shard"]
        if shard not in cache:
            payload = torch.load(corpus.SUPPORT / manifest["shards"][shard]["file"],
                                 weights_only=False, mmap=True)
            cache[shard] = {slot: (fields["observations"], fields["actions_taken"].numpy())
                            for slot, fields in enumerate(payload["episodes"])}
            del payload
        obs, acts = cache[shard][record["slot"]]
        t = record["t"]
        start = max(0, t - config.sequence_long + 1)
        frames_list.append(obs[start : t + 1].clone())
        led = np.concatenate([[config.n_actions] if start == 0 else [acts[start - 1]],
                              acts[start : t]]).astype(np.int64)
        leds.append(torch.from_numpy(led))
        labels.append(positives.astype(float))
    scores = score_roots(encoder, world, head, config, frames_list, leds)
    print(f"\nheld-out evaluation, pixel model {args.model.name}")
    results["hazard_forks_damage"] = report("hazard forks (damage)", scores,
                                            np.stack(labels), 11)

    # ------------------------------------------ branched and policy forks (death truth)
    for name, seed in (("branched_965", 21), ("policy_fork_104", 31)):
        path = HERE / "fork_histories" / f"{name}.pt"
        if not path.exists():
            print(f"  {name}: histories missing, skipped")
            continue
        rows = torch.load(path, weights_only=False)
        if "frames" not in rows[0]:
            print(f"  {name}: histories carry no frames, skipped")
            continue
        scores = score_roots(encoder, world, head, config,
                             [r["frames"] for r in rows], [r["led_to_action"] for r in rows])
        truth = np.stack([r["true_death"].numpy().astype(float) for r in rows])
        results[name] = report(f"{name} (death)", scores, truth, seed)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"evaluation_{args.model.stem}.json").write_text(json.dumps(results, indent=2))
    print(f"wrote evaluation_{args.model.stem}.json")


if __name__ == "__main__":
    main()
