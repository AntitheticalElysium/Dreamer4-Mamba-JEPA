"""Did Phase 3 make the policy less safe than the BC it started from?

Real forks, real successors: every action executed in the simulator from BC-visited
DEV states. The actor's own frozen BC is the control, so this asks whether actor
updates moved probability mass toward actions that truly kill.
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
from d4mj.counterfactual import OutcomeForks, collect_outcome_forks, outcome_metrics
from d4mj.representation import Decoder, Encoder
from d4mj.transition import World

DEVICE = "cuda"
DRAWS = 2000
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
ENCODER_REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"


@torch.no_grad()
def _probabilities(heads: Heads, agent: torch.Tensor) -> torch.Tensor:
    return heads.eval()(agent.to(DEVICE)[:, None])["policy"][:, 0, 0].softmax(-1).cpu()


def _wider_forks(path: Path, seeds: int, world: World, prior: Heads, config: Config) -> dict:
    """The gate's own forks span 8 rollout seeds, which is too few clusters to call an
    increase in death. Same collector, same protocol, more seeds -- collected once and
    cached, so both arms and both budgets are scored on identical states."""
    if path.exists():
        return torch.load(path, weights_only=False)
    stored = json.loads(ENCODER_REPORT.read_text())
    base = replace(Config(), n_latents=64, d_bottleneck=16)
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))
    wider = replace(config, outcome_gate_seeds=tuple(range(12_100, 12_100 + seeds)))
    forks = vars(collect_outcome_forks(world, encoder.eval(), prior, wider))
    torch.save(forks, path)
    return forks


def _clustered(values: torch.Tensor, seed: torch.Tensor) -> tuple[float, float, float]:
    """Bootstrap over whole rollout seeds -- fork states within a seed are one trajectory."""
    keys = seed.unique()
    per_seed = torch.stack([values[seed == key].mean() for key in keys])
    generator = torch.Generator().manual_seed(2**22)
    draws = torch.randint(len(keys), (DRAWS, len(keys)), generator=generator)
    samples = per_seed[draws].mean(1)
    return float(per_seed.mean()), float(samples.quantile(0.025)), float(samples.quantile(0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("attention", "mamba"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--fork-seeds", type=int, default=64)
    args = parser.parse_args()
    source = HERE / f"v2_phase2_{args.arm}"
    out = HERE / f"v2_phase3_{args.arm}{args.tag}"

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    saved = replace(base, transition="direct", time_mixer=args.arm)
    config = replace(saved, horizon=saved.direct_rollout)

    world, prior, actor = World(saved).to(DEVICE), Heads(saved).to(DEVICE), Heads(saved).to(DEVICE)
    load(source / "phase2_final.pt", saved, part0=world, part1=prior)
    load(out / "phase3_final.pt", config, part0=World(saved).to(DEVICE), part1=actor)

    forks = OutcomeForks(**_wider_forks(source / f"outcome_forks_s{args.fork_seeds}.pt",
                                        args.fork_seeds, world.eval(), prior.eval(), config))
    death, reward, seed = forks.true_death.float(), forks.true_reward, forks.seed
    guess_death, guess_reward = forks.model_death, forks.model_reward
    bc, pi = _probabilities(prior, forks.agent), _probabilities(actor, forks.agent)

    lethal = death.sum(1)
    strata = {
        "all": torch.ones(len(death), dtype=torch.bool),
        "all-safe": lethal == 0,
        "escape-rich": (lethal >= 1) & (lethal <= 2),
        "trap-heavy": lethal >= 14,
    }
    report: dict[str, object] = {
        "arm": args.arm, "tag": args.tag, "states": len(death),
        "bc_gate": outcome_metrics(forks, prior, config),
        "actor_gate": outcome_metrics(forks, actor, config),
    }
    rows = {}
    for name, mask in strata.items():
        if not mask.any():
            continue
        pairs = {
            "death_bc": (bc * death).sum(1)[mask],
            "death_actor": (pi * death).sum(1)[mask],
            "reward_bc": (bc * reward).sum(1)[mask],
            "reward_actor": (pi * reward).sum(1)[mask],
            # What the world model told the actor about the actions it actually picks.
            # Optimism is true minus predicted, so a negative death optimism means the
            # model believes the chosen actions are safer than they are.
            "predicted_death_bc": (bc * guess_death).sum(1)[mask],
            "predicted_death_actor": (pi * guess_death).sum(1)[mask],
            "predicted_reward_bc": (bc * guess_reward).sum(1)[mask],
            "predicted_reward_actor": (pi * guess_reward).sum(1)[mask],
        }
        entry = {key: _clustered(value, seed[mask]) for key, value in pairs.items()}
        entry["death_change"] = _clustered(pairs["death_actor"] - pairs["death_bc"], seed[mask])
        entry["reward_change"] = _clustered(pairs["reward_actor"] - pairs["reward_bc"], seed[mask])
        for who in ("bc", "actor"):
            entry[f"death_optimism_{who}"] = _clustered(
                pairs[f"death_{who}"] - pairs[f"predicted_death_{who}"], seed[mask])
            entry[f"reward_optimism_{who}"] = _clustered(
                pairs[f"reward_{who}"] - pairs[f"predicted_reward_{who}"], seed[mask])
        entry["death_optimism_change"] = _clustered(
            (pairs["death_actor"] - pairs["predicted_death_actor"])
            - (pairs["death_bc"] - pairs["predicted_death_bc"]), seed[mask])
        entry["kl_actor_bc"] = _clustered(
            (pi[mask] * (pi[mask].clamp_min(1e-12).log() - bc[mask].clamp_min(1e-12).log())).sum(1),
            seed[mask])
        entry["argmax_agreement"] = _clustered(
            (pi[mask].argmax(1) == bc[mask].argmax(1)).float(), seed[mask])
        entry["states"] = int(mask.sum())
        rows[name] = entry
    report["strata"] = rows

    # The declared safety condition: the actor may not raise true death where it acts.
    change = rows["all"]["death_change"]
    report["unsafe"] = bool(change[1] > 0.0)
    report["verdict"] = ("actor raises true death"
                         if report["unsafe"] else "no significant increase in true death")
    (out / "actor_safety.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"\n{args.arm}{args.tag}: {report['verdict']}  ({len(death)} fork states)")
    print(f"{'stratum':<13}{'n':>5}{'death BC':>10}{'death actor':>13}"
          f"{'change (95%)':>26}{'reward chg':>12}{'KL':>8}{'agree':>8}")
    for name, entry in rows.items():
        c, r = entry["death_change"], entry["reward_change"]
        print(f"{name:<13}{entry['states']:>5}{entry['death_bc'][0]:>10.4f}"
              f"{entry['death_actor'][0]:>13.4f}"
              f"{c[0]:>+11.4f} [{c[1]:+.4f},{c[2]:+.4f}]"
              f"{r[0]:>+12.4f}{entry['kl_actor_bc'][0]:>8.3f}"
              f"{entry['argmax_agreement'][0]:>8.3f}")

    # Exploitation: is the actor moving toward actions the world model gets wrong?
    print(f"\n{'stratum':<13}{'who':<7}{'death true':>11}{'death pred':>11}"
          f"{'optimism (95%)':>27}{'rew true':>10}{'rew pred':>10}{'optimism':>10}")
    for name, entry in rows.items():
        for who in ("bc", "actor"):
            o = entry[f"death_optimism_{who}"]
            print(f"{name:<13}{who:<7}{entry[f'death_{who}'][0]:>11.4f}"
                  f"{entry[f'predicted_death_{who}'][0]:>11.4f}"
                  f"{o[0]:>+12.4f} [{o[1]:+.4f},{o[2]:+.4f}]"
                  f"{entry[f'reward_{who}'][0]:>10.4f}"
                  f"{entry[f'predicted_reward_{who}'][0]:>10.4f}"
                  f"{entry[f'reward_optimism_{who}'][0]:>+10.4f}")


if __name__ == "__main__":
    main()
