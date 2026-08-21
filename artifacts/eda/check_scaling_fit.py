"""Do the paired rungs memorise their training roots?

Within-state AUC on the rung's own training roots, on other fit roots it never drew,
and on held-out seeds. A high fit number beside chance elsewhere is memorisation; low
everywhere means the objective learned nothing at all.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from evaluate_damage_classifier import auc, interval, score_roots
from train_damage_classifier import DamageHead
from train_paired_scaling import load_pool
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World

config = Config(transition="direct", time_mixer="attention")
roots, trajectories = load_pool()
fit = [r for r in roots if r["split"] == "fit" and not r["reserved"]]
hazard = [r for r in fit if r["hazard"]]
order = np.random.default_rng(4242).permutation(len(hazard))
held = [r for r in roots if r["split"] == "test" and r["hazard"]]

def measure(world, head, subset, seed):
    subset = subset[:250]
    histories, leds, labels = [], [], []
    for record in subset:
        latents, led = trajectories[record["seed"]]
        t = record["step"]; start = max(0, t - config.sequence_long + 1)
        histories.append(latents[start : t + 1]); leds.append(led[start : t + 1])
        labels.append(record["label"])
    scores = score_roots(world, head, config, histories, leds)
    labels = np.stack(labels)
    values = np.array([auc(scores[i], labels[i]) for i in range(len(subset))])
    values = values[~np.isnan(values)]
    a, (lo, hi) = interval(values, seed)
    return f"{a:.4f} [{lo:.4f}, {hi:.4f}]  n={len(values)}"

for rung in (394, 3152):
    world, head = World(config).to(config.device), DamageHead(config).to(config.device)
    load(HERE / f"scaling/k{rung}/model.pt", config, part0=world, part1=head)
    world.eval(); head.eval()
    trained = [hazard[i] for i in order[:rung]]
    untrained = [hazard[i] for i in order[rung:]]
    print(f"\nrung {rung}")
    print(f"  its own {rung} anchored roots   {measure(world, head, trained, 1)}")
    if untrained:
        print(f"  fit hazard roots not in rung  {measure(world, head, untrained, 2)}")
    print(f"  held-out seeds                {measure(world, head, held, 3)}")
