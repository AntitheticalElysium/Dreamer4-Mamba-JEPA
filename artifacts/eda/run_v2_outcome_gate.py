"""The real post-Phase-2 gate (GATES.md): the model's own heads on live simulator forks.

Every death number so far came from a frozen probe fitted on true successors and applied
to predicted ones. This runs the trained reward and continuation heads on generated
successors, against the state-blind action marginals, with the observed-successor versions
as localization controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.counterfactual import collect_outcome_forks, outcome_metrics
from d4mj.representation import Decoder, Encoder
from d4mj.transition import World

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("attention", "mamba"))
    args = parser.parse_args()
    folder = HERE / f"v2_phase2_{args.arm}"

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    saved = replace(base, transition="direct", time_mixer=args.arm)
    world, heads = World(saved).to(DEVICE), Heads(saved).to(DEVICE)
    load(folder / "phase2_final.pt", saved, part0=world, part1=heads)
    # S68: Direct may not imagine past the rollout its loss trains, which is two
    config = replace(saved, horizon=saved.direct_rollout)

    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))

    prior = heads.eval()
    forks = collect_outcome_forks(world.eval(), encoder.eval(), prior, config)
    torch.save(vars(forks), folder / "outcome_forks.pt")
    gate = outcome_metrics(forks, prior, config)
    (folder / "outcome_gate.json").write_text(json.dumps(gate, indent=2))

    print(f"\n{args.arm}: outcome gate pass={gate['passed']}")
    print(f"  generated  reward regret {gate['reward_choice_regret']:.4f} "
          f"vs marginal {gate['reward_marginal_regret']:.4f}")
    print(f"             terminal BCE  {gate['terminal_bce']:.4f} "
          f"vs marginal {gate['terminal_marginal_bce']:.4f}   AUC {gate['terminal_auc']:.3f}")
    print(f"  observed   reward regret {gate['observed_reward_choice_regret']:.4f}")
    print(f"             terminal BCE  {gate['observed_terminal_bce']:.4f}   "
          f"AUC {gate['observed_terminal_auc']:.3f}")


if __name__ == "__main__":
    main()
