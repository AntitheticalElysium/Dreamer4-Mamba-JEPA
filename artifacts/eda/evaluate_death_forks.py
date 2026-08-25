"""Does the repaired stack transfer to the death mechanic, on forks it never saw?

Scope, stated plainly because it is not what was asked for. The intended next step was a
recursive 17-action policy-fork evaluation, and that cannot be run as such:

  * the existing recursive evaluator is built on the retired fixed-`d` fatality axis
    (`artifacts/evaluate_recursive_generated_latent_outcome.py`), so it is not the
    corrected methodology;
  * `true_death` is `env_step(...)[3]` -- the terminated flag of the *immediate*
    successor (`collect_branched_policy_states.py`). There is no multi-step outcome
    truth in these sets, so a recursive claim has nothing to score against.

Genuine recursive truth needs new rollouts: execute the forked action, then continue
under a policy for K steps, recording death. That is a collection job, not an
evaluation.

What this does instead is the corrected one-step methodology on the *death* family --
complementing the damage family everything else has used -- over 1,069 forks where
actions genuinely decide the outcome and which the production world never trained on:

  policy_fork_104   104 roots on the policy trajectory, all 17 actions executed
  branched_965      965 branched roots, likewise

Both are stored at the old 32-slot geometry, so their frames are re-encoded with the
repaired 64x16 tokenizer, and histories are ragged so each root is encoded on its own.

The two sets are pooled into one 1,069-root split, because 104 roots alone leaves 21
held-out and cannot resolve anything. The probe is fitted on PREDICTED successors and
read on held-out roots: true successor latents are not recoverable without re-running
the original policy rollout's RNG, so this is the own-probe reading -- does the
predicted successor carry the death signal at all -- and a weaker claim than the
transfer test R_delta performs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from evaluate_damage_classifier import interval
from reevaluate_phase1b_delta import fit_probe, within_state

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.representation import Encoder, pack
from d4mj.transition import World, commit_inputs

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"


@torch.no_grad()
def encode(encoder, frames, config):
    """Re-encode one root's stored history at the repaired geometry, as
    `encode_fork_dataset` does: one causal pass over the whole window, then pack."""
    z, _, _ = encoder(patchify(frames[None], config.patch).to(DEVICE))
    return pack(z, config)[0].cpu()


@torch.no_grad()
def branches(world, config, history, led):
    """All 17 successors from each root's final block."""
    out = []
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    actions = torch.arange(17, device=DEVICE)
    for lo in range(0, len(history), 8):
        z = history[lo : lo + 8].to(DEVICE)
        n, steps = z.shape[0], z.shape[1]
        committed, conditioning = commit_inputs(z, rng, config)
        features, _, _ = world(None, led[lo : lo + 8].to(DEVICE), committed, conditioning)
        last = features[:, -1:]
        out.append(world.predict(last.expand(n, 17, *last.shape[2:]),
                                 actions[None].expand(n, -1)).flatten(2).cpu())
    return torch.cat(out)


def main() -> None:
    base = replace(Config(), n_latents=64, d_bottleneck=16)
    config = replace(base, transition="direct", time_mixer="attention")

    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]), part0=encoder)
    encoder.eval()

    world = World(config).to(DEVICE)
    world.load_state_dict(torch.load(HERE / "production_1b" / "world.pt",
                                     weights_only=False)["world"])
    world.eval()

    frames, led, death, source = [], [], [], []
    for name in ("policy_fork_104", "branched_965"):
        rows = torch.load(HERE / "fork_histories" / f"{name}.pt", weights_only=False)
        frames += [r["frames"] for r in rows]
        led += [r["led_to_action"].long() for r in rows]
        death += [r["true_death"].float() for r in rows]
        source += [name] * len(rows)
    death = torch.stack(death).numpy()
    lengths = sorted({len(f) for f in frames})
    print(f"{len(frames)} roots pooled, history lengths {lengths}", flush=True)

    predicted = []
    for f, a in zip(frames, led):
        history = encode(encoder, f, base)[None]
        predicted.append(branches(world, config, history, a[None])[0])
    predicted = torch.stack(predicted).float()
    print(f"predicted successors {tuple(predicted.shape)}", flush=True)

    rng = np.random.default_rng(11)
    order = rng.permutation(len(frames))
    fit = torch.zeros(len(frames), dtype=torch.bool); fit[order[: int(0.6 * len(frames))]] = True
    tune = torch.zeros(len(frames), dtype=torch.bool)
    tune[order[int(0.6 * len(frames)) : int(0.8 * len(frames))]] = True
    test = ~(fit | tune)

    width = predicted.shape[-1]
    probe, _ = fit_probe(predicted[fit].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(death[fit.numpy()].reshape(-1)).float().to(DEVICE),
                         predicted[tune].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(death[tune.numpy()].reshape(-1)).float().to(DEVICE),
                         seed=11)
    with torch.no_grad():
        scores = torch.cat([probe(predicted[test][lo:lo+64].reshape(-1, width).to(DEVICE)).cpu()
                            for lo in range(0, int(test.sum()), 64)]).numpy().reshape(-1, 17)
    values = within_state(scores, death[test.numpy()])
    mean, (lo, hi) = interval(values, 17)
    result = {"roots": len(frames), "test_roots": int(test.sum()),
              "within_auc_predicted": mean, "ci": [lo, hi],
              "death_rate": float(death.mean()),
              "reading": "own-probe on predicted successors, not a transfer test"}
    print(f"\npredicted successors, within-state death AUC {mean:.4f} [{lo:.4f}, {hi:.4f}] "
          f"over {int(test.sum())} held-out roots", flush=True)
    print(f"  reference: a supervised probe reading these states directly scored "
          f"0.611-0.660; frozen-latent forward arms scored 0.49-0.58", flush=True)

    (HERE / "death_forks_production.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
