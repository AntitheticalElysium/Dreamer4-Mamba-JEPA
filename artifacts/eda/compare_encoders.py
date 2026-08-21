"""Mini-H2 against the from-scratch pixel encoder, paired over identical roots.

Equal final loss is not evidence that Phase-1A initialization bought nothing; only a
paired comparison on the same roots can support or refuse that.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import corpus
from evaluate_damage_classifier import auc, interval
from evaluate_damage_pixels import score_roots as score_pixel
from train_damage_classifier import DamageHead
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.representation import Encoder
from d4mj.transition import World

config = Config(transition="direct", time_mixer="attention")
records = []
for path in sorted((HERE / "latent_forks").glob("shard-*.pt")):
    records += torch.load(path, weights_only=False)
manifest = json.loads((corpus.SUPPORT / "manifest.json").read_text())
store, frames_list, leds, labels = {}, [], [], []
for record in records:
    health, dead = record["health"].numpy(), record["dead"].numpy()
    positives = (health <= -1) | dead
    if not positives.any() or not ((health >= 0) & ~dead).any():
        continue
    shard = record["shard"]
    if shard not in store:
        payload = torch.load(corpus.SUPPORT / manifest["shards"][shard]["file"],
                             weights_only=False, mmap=True)
        store[shard] = {s: (f["observations"], f["actions_taken"].numpy())
                        for s, f in enumerate(payload["episodes"])}
        del payload
    obs, acts = store[shard][record["slot"]]
    t = record["t"]; start = max(0, t - config.sequence_long + 1)
    frames_list.append(obs[start : t + 1].clone())
    led = np.concatenate([[config.n_actions] if start == 0 else [acts[start - 1]],
                          acts[start : t]]).astype(np.int64)
    leds.append(torch.from_numpy(led)); labels.append(positives.astype(float))
labels = np.stack(labels)
print(f"{len(labels)} hazard roots, identical for both pixel arms")

per = {}
for name, path in (("miniH2 (Phase-1A init)", "damage_miniH2/model_020000.pt"),
                   ("from scratch", "damage_pixels/model_020000.pt")):
    enc = Encoder(config).to(config.device)
    w, h = World(config).to(config.device), DamageHead(config).to(config.device)
    load(HERE / path, config, part0=enc, part1=w, part2=h)
    for m in (enc, w, h):
        m.eval()
    s = score_pixel(enc, w, h, config, frames_list, leds)
    per[name] = np.array([auc(s[i], labels[i]) for i in range(len(labels))])
    a, (lo, hi) = interval(per[name], 7)
    print(f"  {name:<24} within AUC {a:.4f} [{lo:.4f}, {hi:.4f}]")
d, (lo, hi) = interval(per["miniH2 (Phase-1A init)"] - per["from scratch"], 9)
print(f"\npaired Phase-1A init minus from scratch: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
