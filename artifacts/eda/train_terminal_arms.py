"""Does the model need same-state death/survival contrasts, or just better tails?

Three arms, identical apart from what the terminal supervision contains:

  control         ordinary production training only -- already trained as
                  `production_1b/world.pt` with the same encoder, corpus, budget,
                  initialisation and streams
  factual         plus a terminal term over the 3,198 actionable roots, always
                  supervising the single logged fatal transition
  counterfactual  plus a terminal term over the same roots on the same root schedule,
                  supervising one of the 17 successors per presentation

Five things this gets right that a first draft did not, each of which would have made the
comparison mean something other than it claims:

  action history   the terminal pass feeds the real incoming actions, not BOS at every
                   block. Feeding BOS is the exact defect that made the original fork
                   trainer's ordinary term an unconditional next-latent loss.
  separate RNG     the terminal pass draws from its own generator, so `transition_loss`
                   consumes the model stream exactly as the control does and the ordinary
                   batches stay aligned step for step.
  production path  the terminal term is blended into `dynamics` before `_balance`, which
                   is how `train_agent` mixes `terminal_dynamics_mass`. Hand-rolling a
                   weighted sum bypasses the running-RMS normalisation and leaves the
                   production checkpoint an unmatched control.
  schedule         a deterministic shuffled pass over all 3,198 x 17 = 54,366
                   (root, action) pairs. Cycling `(step + i) % 17` over randomly drawn
                   roots cycles globally, so no particular root is guaranteed to see all
                   17 actions. One full pass is 13,592 steps at batch 4; 20,000 gives
                   every pair once and about half a second pass.
  matched roots    the factual arm walks the identical root order and substitutes that
                   root's logged fatal action, so the arms differ only in which successor
                   is supervised -- never in how many.

Do not read a counterfactual null before step 13,592, which is the first complete pass.

The outcome is oracle-selected: all-17 replay is what identified which tails were
actionable, so the factual arm tests whether informative tails suffice, and is not a
sampling rule anyone could deploy.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.checkpoint import save
from d4mj.config import Config
from d4mj.data import load_episodes, sample_batch
from d4mj.train import (_balance, _checkpoint, _generators, _share_initialisation, _to,
                        _update, optimizer)
from d4mj.transition import World, commit_inputs, transition_loss

DEVICE = "cuda"
CACHE = HERE / "latent_cache_64"
N_ACTIONS = 17


EXPECTED_ROOTS = 3198


def load_roots():
    """Every completeness check that a partial cache would otherwise slip past.

    A first version loaded whatever shards happened to exist, so an interrupted encode
    would have silently run the whole experiment on a fraction of the roots.
    """
    manifest_path = HERE / "actionable_latents" / "manifest.json"
    assert manifest_path.exists(), "actionable_latents/manifest.json missing: encode first"
    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("z_history_source"), (
        "z_history has not been aligned to the production cache; run "
        "align_actionable_latents.py")

    shards = sorted(glob.glob(str(HERE / "actionable_latents" / "shard-*.pt")))
    assert len(shards) == manifest["shards"], (
        f"{len(shards)} shards on disk, manifest declares {manifest['shards']}")
    rows = []
    for path in shards:
        rows += torch.load(path, weights_only=False)
    assert len(rows) == EXPECTED_ROOTS == manifest["roots"], (
        f"{len(rows)} roots loaded, expected {EXPECTED_ROOTS}, "
        f"manifest declares {manifest['roots']}")

    table_path = HERE / "actionable_actions.pt"
    assert table_path.exists(), "actionable_actions.pt missing: run build_actionable_actions.py"
    table = torch.load(table_path, weights_only=False)
    keys = [(int(r["shard"]), int(r["slot"])) for r in rows]
    assert len(set(keys)) == len(keys), "duplicate roots in the latent shards"
    missing = [k for k in keys if k not in table]
    assert not missing, f"{len(missing)} roots without an action history"

    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows])
    assert torch.isfinite(history).all() and torch.isfinite(branch).all(), "non-finite latents"
    return (history, branch, torch.stack([table[k] for k in keys]),
            torch.tensor([r["lethal_action"] for r in rows]))


def schedule(n_roots: int, seed: int):
    """A deterministic shuffled pass over every (root, action) pair."""
    pairs = np.stack(np.meshgrid(np.arange(n_roots), np.arange(N_ACTIONS),
                                 indexing="ij"), axis=-1).reshape(-1, 2)
    pairs = pairs[np.random.default_rng(seed).permutation(len(pairs))]
    assert len(pairs) == n_roots * N_ACTIONS
    assert len({tuple(x) for x in pairs}) == len(pairs), "schedule repeats a pair"
    counts = np.bincount(pairs[:, 0], minlength=n_roots)
    assert (counts == N_ACTIONS).all(), "some root does not receive all 17 actions"
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("factual", "counterfactual"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--terminal-batch", type=int, default=4)
    parser.add_argument("--terminal-mass", type=float, default=0.2)
    parser.add_argument("--milestones", type=int, nargs="+",
                        default=(5000, 10000, 13592, 20000))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume-every", type=int, default=500)
    args = parser.parse_args()
    out = args.out or HERE / f"terminal_{args.arm}"
    out.mkdir(parents=True, exist_ok=True)

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    config = replace(base, transition="direct", time_mixer="attention")
    digest = json.loads((CACHE / "manifest.json").read_text())["cache_digest"]
    episodes = load_episodes(CACHE, digest, verify=False)
    history, branch, led_history, lethal = load_roots()
    pairs = schedule(len(history), config.seed + 7)
    per_pass = -(-len(pairs) // args.terminal_batch)   # ceil: 54,366/4 is 13,592
    print(f"{args.arm}: {len(episodes)} cached episodes, {len(history)} roots, "
          f"{len(pairs):,} (root, action) pairs, one pass = {per_pass:,} steps", flush=True)

    # identical to train_dynamics: same seed, same construction, same streams
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(DEVICE)
    opt = optimizer([world], config)
    balance: dict[str, float] = {}
    sampler, rng = _generators(config, 1)
    terminal_rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    spatial, d = config.n_spatial, config.d_spatial

    # A resume needs optimizer moments, the running-RMS balance and every stream, not
    # weights. The contract carries the arm and the declared length: `sample_batch`
    # schedules short and long batches as a function of TOTAL steps, so a 40k run has a
    # different curriculum from a 20k one at every step. Resuming across a changed
    # length would silently splice two schedules, and is refused.
    resume_path = args.out / "resume.pt"
    streams = {"sampler": sampler, "model": rng, "terminal": terminal_rng}
    contract = f"terminal:{args.arm}:{args.steps}"
    begin = _checkpoint(resume_path if resume_path.exists() else None, config,
                        [world, opt], balance, streams, contract=contract)
    if begin:
        print(f"resumed from step {begin}", flush=True)

    started, curve = time.time(), []
    for step in range(begin, args.steps):
        batch = _to(sample_batch(episodes, sampler, config, step, args.steps), DEVICE)
        dynamics = transition_loss(world, batch, rng, config, step=step)

        window = pairs[(step * args.terminal_batch + np.arange(args.terminal_batch))
                       % len(pairs)]
        roots = torch.from_numpy(window[:, 0])
        action = (lethal[roots] if args.arm == "factual"
                  else torch.from_numpy(window[:, 1]))
        z = history[roots].to(DEVICE)
        n, t = z.shape[0], z.shape[1]
        committed, conditioning = commit_inputs(z.view(n, t, spatial, d),
                                                terminal_rng, config)
        features, _, _ = world(None, led_history[roots].to(DEVICE), committed, conditioning)
        target = branch[roots, action].to(DEVICE)
        terminal = (world.predict(features[:, -1:], action[:, None].to(DEVICE))
                    .flatten(2)[:, 0] - target).pow(2).mean()

        blended = (1.0 - args.terminal_mass) * dynamics + args.terminal_mass * terminal
        _update(opt, _balance({"dynamics": blended}, balance, config), [world], config, step)
        curve.append({"step": step + 1, "dynamics": float(dynamics.detach()),
                      "terminal": float(terminal.detach())})
        if (step + 1) % 500 == 0 or step + 1 == args.steps:
            recent = curve[-500:]
            print(f"  {step+1}/{args.steps} "
                  f"dyn={sum(c['dynamics'] for c in recent)/len(recent):.5f} "
                  f"term={sum(c['terminal'] for c in recent)/len(recent):.5f} "
                  f"[{time.time()-started:.0f}s]", flush=True)
        if (step + 1) in tuple(args.milestones):
            save(out / f"world_{step+1:06d}.pt", config, part0=world)
        if (step + 1) % args.resume_every == 0 or step + 1 == args.steps:
            _checkpoint(resume_path, config, [world, opt], balance, streams,
                        step + 1, contract)

    torch.save({"world": world.state_dict()}, out / "world.pt")
    (out / "training_report.json").write_text(json.dumps(
        {"arm": args.arm, "steps": args.steps, "roots": len(history),
         "pairs": len(pairs), "steps_per_pass": per_pass,
         "terminal_batch": args.terminal_batch, "terminal_mass": args.terminal_mass,
         "seconds": time.time() - started, "curve_tail": curve[-500:]}, indent=2))
    print(f"done in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
