"""Is the gate's terminal-BCE failure calibration or dynamics?

The continuation head ranks death on generated successors almost perfectly (AUC 0.965 and
0.996) while its BCE loses to the action marginal. That is the signature of a wrong
operating point, not a wrong ordering. This fits a scalar temperature and bias on
generated successors from seeds disjoint from the gate's, applies it frozen to the gate
forks, and re-reads the BCE.

Passing here does NOT pass the gate -- the gate reads the raw head. It says where the
defect is: if a two-parameter monotone rescale fixes BCE, the generated latents carry the
death signal and only the scale is wrong.
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
EPS = 1e-6


def logit(p):
    return torch.log(p.clamp(EPS, 1 - EPS) / (1 - p).clamp(EPS, 1 - EPS))


def bce(p, y):
    p = p.clamp(EPS, 1 - EPS)
    return float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("attention", "mamba"))
    parser.add_argument("--fit-seeds", type=int, default=16)
    args = parser.parse_args()
    folder = HERE / f"v2_phase2_{args.arm}"

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    saved = replace(base, transition="direct", time_mixer=args.arm)
    world, heads = World(saved).to(DEVICE), Heads(saved).to(DEVICE)
    load(folder / "phase2_final.pt", saved, part0=world, part1=heads)
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))
    world, heads, encoder = world.eval(), heads.eval(), encoder.eval()

    gate_config = replace(saved, horizon=saved.direct_rollout)
    # seeds the gate never touches
    fit_seeds = tuple(range(12_100, 12_100 + args.fit_seeds))
    assert not set(fit_seeds) & set(gate_config.outcome_gate_seeds), "fit seeds overlap the gate"
    fit_config = replace(gate_config, outcome_gate_seeds=fit_seeds)

    fit = collect_outcome_forks(world, encoder, heads, fit_config)
    varies = fit.true_death.float().amax(1) > fit.true_death.float().amin(1)
    y = fit.true_death.float()[varies].flatten()
    x = logit(fit.model_death[varies].flatten())
    print(f"{args.arm}: fitting on {int(varies.sum())} opportunity states from "
          f"{len(fit_seeds)} disjoint seeds, {len(y)} branches, {y.mean():.3f} fatal")

    a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            a.exp() * x + b, y)
        loss.backward()
        return loss

    opt.step(closure)
    scale, shift = float(a.exp()), float(b)
    print(f"  fitted temperature {1/scale:.3f} (scale {scale:.3f}), bias {shift:+.3f}")

    # re-read through the gate's own function rather than a hand-rolled marginal, so the
    # numbers are the ones the gate would print
    from d4mj.counterfactual import OutcomeForks

    stored_forks = torch.load(folder / "outcome_forks.pt", weights_only=False)
    raw_metrics = outcome_metrics(OutcomeForks(**stored_forks), heads, gate_config)
    recal = torch.sigmoid(scale * logit(stored_forks["model_death"]) + shift)
    recal_metrics = outcome_metrics(
        OutcomeForks(**{**stored_forks, "model_death": recal}), heads, gate_config)

    print(f"  gate forks: {raw_metrics['terminal_opportunity_states']} opportunity states")
    print(f"    raw          BCE {raw_metrics['terminal_bce']:.4f}  "
          f"AUC {raw_metrics['terminal_auc']:.3f}")
    print(f"    recalibrated BCE {recal_metrics['terminal_bce']:.4f}  "
          f"AUC {recal_metrics['terminal_auc']:.3f}")
    print(f"    marginal     BCE {raw_metrics['terminal_marginal_bce']:.4f}")
    verdict = recal_metrics["terminal_bce"] < recal_metrics["terminal_marginal_bce"]
    print(f"  gate would pass after a frozen rescale: {recal_metrics['passed']}")
    print(f"  -> {'calibration' if verdict else 'not calibration alone'}")
    (folder / "calibration_check.json").write_text(json.dumps(
        {"arm": args.arm, "scale": scale, "shift": shift,
         "raw": raw_metrics, "recalibrated": recal_metrics,
         "beats_marginal": verdict}, indent=2))


if __name__ == "__main__":
    main()
