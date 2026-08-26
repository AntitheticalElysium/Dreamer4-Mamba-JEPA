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


Exposure, and why one arm is slower than the other
--------------------------------------------------

The paired analysis found the factual arm gaining across its first pass and stopping,
while the counterfactual arm was flat to 13,592 and then gained on its partial second
pass. The exposure arithmetic is the obvious candidate. At four targets per step,
20,000 steps is 80,000 presentations. The counterfactual arm spreads those over 54,366
distinct (root, action) pairs -- 1.47 sightings each -- while the factual arm spends all
80,000 on 3,198 distinct transitions, about 25 sightings each. The counterfactual arm's
first 13,592 steps see every target exactly once.

`--terminal-actions 17` tests that directly. One selected root supplies seventeen
action-conditioned targets from a single encoder forward, since the history and the
backbone pass are what cost anything and `predict` is a pool, a one-block mixer and a
readout. At one root per step this is *cheaper* than the original four-root arm and
covers a full pass over all 54,366 pairs every 3,198 steps -- 6.25 passes inside the
same 20,000-step budget, at unchanged terminal loss mass.

This is an additional intervention, not a replication. Changing the sampler and the
seed at once would answer neither question, so the matrix is: the original sampler at a
new seed to ask whether the trajectory reproduces, and the accelerated sampler beside it
at that same seed to ask whether repetition is the remedy.


Extendable, not merely resumable
--------------------------------

`sample_batch` reads the short/long curriculum off the *declared total*, so passing the
requested stopping point made a 40,000-step run a different experiment from step zero
rather than a continuation of a 20,000-step one. `--horizon` fixes the curriculum to its
own 20,000-step schedule regardless of where the run stops: an extension past the
horizon simply stays in the terminal long-only regime, which is where the schedule had
already put it. The stopping point is therefore not part of the resume contract, and
27,184 steps -- exactly two complete passes at four targets per step -- is a meaningful
first extension in a way that a round 40,000 is not.

The seed-0 arms already on disk cannot be extended by any of this: they were trained
before the resume path existed and hold weights only, with no optimizer moments, no
running-RMS balance and no generator states. Extension starts with the new-seed matrix.

`--seed` moves initialisation, the sampler, model noise, the terminal stream and the
root schedule together. It deliberately does not touch the corpus split, which is baked
into the latent cache this reads: the 197 evaluation roots were chosen under the default
split, and reshuffling it would move them into training and invalidate every paired
comparison against the existing arms.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
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


def schedule(n_roots: int, seed: int, actions: int):
    """A deterministic shuffled pass over every (root, action) pair.

    Returns a root order and, beside it, the actions that root carries at each
    presentation. `actions == 1` is the original per-pair schedule: 54,366 rows, one
    action apiece. `actions == 17` is the exposure-efficient form: 3,198 rows, each a
    root carrying all seventeen at once, so one encoder forward supplies seventeen
    targets. Widths between the two are refused -- 17 is prime, so every other chunk
    size leaves a ragged final group, which either repeats pairs inside a pass or makes
    the batch shape depend on the step.

    Either way the invariant that mattered still holds and is still asserted: one pass
    gives every root all seventeen of its actions exactly once.
    """
    generator = np.random.default_rng(seed)
    if actions == 1:
        pairs = np.stack(np.meshgrid(np.arange(n_roots), np.arange(N_ACTIONS),
                                     indexing="ij"), axis=-1).reshape(-1, 2)
        pairs = pairs[generator.permutation(len(pairs))]
        assert len(pairs) == n_roots * N_ACTIONS
        assert len({tuple(x) for x in pairs}) == len(pairs), "schedule repeats a pair"
        order, choice = pairs[:, 0], pairs[:, 1:2]
    else:
        assert actions == N_ACTIONS, "1 or 17 actions per root only; 17 is prime"
        order = generator.permutation(n_roots)
        choice = np.tile(np.arange(N_ACTIONS), (n_roots, 1))
    counts = np.bincount(np.repeat(order, choice.shape[1]), minlength=n_roots)
    assert (counts == N_ACTIONS).all(), "some root does not receive all 17 actions"
    return order, choice


def data_identity() -> str:
    """What the arms are trained on, folded into the resume contract. `Config` cannot
    see it: the roots, their aligned histories and their action tables all live on
    disk, and a resume across a re-encoded cache would splice two datasets."""
    digest = hashlib.sha256()
    for path in (HERE / "actionable_latents" / "manifest.json",
                 CACHE / "manifest.json", HERE / "actionable_actions.pt"):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("factual", "counterfactual"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=20000,
                        help="the curriculum's own length, fixed independently of where "
                             "the run stops, so an extension continues one schedule "
                             "instead of splicing two")
    parser.add_argument("--seed", type=int, default=None,
                        help="initialisation, sampler, model noise, terminal stream and "
                             "root schedule; never the corpus split, which the cache fixes")
    parser.add_argument("--terminal-roots", type=int, default=4,
                        help="roots per step, one encoder forward each")
    parser.add_argument("--terminal-actions", type=int, default=1, choices=(1, N_ACTIONS),
                        help="action-conditioned targets read off each root's features")
    parser.add_argument("--terminal-mass", type=float, default=0.2)
    parser.add_argument("--milestones", type=int, nargs="+",
                        default=(5000, 10000, 13592, 20000))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true",
                        help="marks the contract, so a smoke checkpoint can never be "
                             "mistaken for a real run's resume point")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.out = args.out or HERE / f"terminal_{args.arm}"
    out.mkdir(parents=True, exist_ok=True)

    # the factual arm supervises one target -- its logged fatal transition -- so
    # seventeen copies of it would be the same gradient with a longer runtime
    assert args.arm == "counterfactual" or args.terminal_actions == 1, (
        "the factual arm has one successor per root; --terminal-actions 17 would "
        "average seventeen copies of it")

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    config = replace(base, transition="direct", time_mixer="attention")
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    digest = json.loads((CACHE / "manifest.json").read_text())["cache_digest"]
    episodes = load_episodes(CACHE, digest, verify=False)
    history, branch, led_history, lethal = load_roots()
    order, choice = schedule(len(history), config.seed + 7, args.terminal_actions)
    targets = args.terminal_roots * args.terminal_actions
    per_pass = -(-len(order) // args.terminal_roots)
    print(f"{args.arm}: {len(episodes)} cached episodes, {len(history)} roots, seed "
          f"{config.seed}, {targets} targets/step over {len(order):,} presentations, "
          f"one pass = {per_pass:,} steps ({args.steps / per_pass:.2f} passes in "
          f"{args.steps:,})", flush=True)

    # identical to train_dynamics: same seed, same construction, same streams
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(DEVICE)
    opt = optimizer([world], config)
    balance: dict[str, float] = {}
    sampler, rng = _generators(config, 1)
    terminal_rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    spatial, d = config.n_spatial, config.d_spatial

    # A resume needs optimizer moments, the running-RMS balance and every stream, not
    # weights. The contract carries everything `Config` cannot see and that changing
    # would make the continuation a different experiment -- the sampler, the loss mass,
    # the data on disk and the curriculum's own horizon. It deliberately excludes the
    # stopping point: that is exactly what an extension is allowed to change.
    resume_path = out / "resume.pt"
    streams = {"sampler": sampler, "model": rng, "terminal": terminal_rng}
    contract = ":".join(["terminal", args.arm, f"seed{config.seed}",
                         f"roots{args.terminal_roots}", f"actions{args.terminal_actions}",
                         f"mass{args.terminal_mass:g}", f"horizon{args.horizon}",
                         data_identity(), "smoke" if args.smoke else "run"])
    begin = _checkpoint(resume_path if resume_path.exists() else None, config,
                        [world, opt], balance, streams, contract=contract)
    if begin:
        print(f"resumed from step {begin} of {args.steps}", flush=True)
    elif (out / "world.pt").exists() and not args.overwrite:
        raise SystemExit(f"{out} already holds a finished run and there is no resume "
                         f"point to continue from; pass --overwrite to discard it")
    if begin >= args.steps:
        raise SystemExit(f"the checkpoint is already at step {begin}")

    started, curve = time.time(), []
    for step in range(begin, args.steps):
        batch = _to(sample_batch(episodes, sampler, config, step, args.horizon), DEVICE)
        dynamics = transition_loss(world, batch, rng, config, step=step)

        window = (step * args.terminal_roots
                  + np.arange(args.terminal_roots)) % len(order)
        roots = torch.from_numpy(order[window])
        action = (lethal[roots][:, None] if args.arm == "factual"
                  else torch.from_numpy(choice[window]))
        z = history[roots].to(DEVICE)
        n, t = z.shape[0], z.shape[1]
        committed, conditioning = commit_inputs(z.view(n, t, spatial, d),
                                                terminal_rng, config)
        features, _, _ = world(None, led_history[roots].to(DEVICE), committed, conditioning)

        # the history and the backbone pass are what cost anything, so every action a
        # root carries reads off the same features; `predict` is a pool, a one-block
        # mixer and a readout, and is the only part that has to run per action
        a = action.shape[1]
        flat = action.reshape(-1, 1)
        target = branch[roots.repeat_interleave(a), flat[:, 0]].to(DEVICE)
        terminal = (world.predict(features[:, -1:].repeat_interleave(a, dim=0),
                                  flat.to(DEVICE)).flatten(2)[:, 0] - target).pow(2).mean()

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
        {"arm": args.arm, "steps": args.steps, "horizon": args.horizon,
         "seed": config.seed, "roots": len(history), "pairs": len(history) * N_ACTIONS,
         "presentations": len(order), "steps_per_pass": per_pass,
         "passes": args.steps / per_pass, "terminal_roots": args.terminal_roots,
         "terminal_actions": args.terminal_actions, "targets_per_step": targets,
         "terminal_mass": args.terminal_mass, "contract": contract,
         "resumed_from": begin, "seconds": time.time() - started,
         "curve_tail": curve[-500:]}, indent=2))
    print(f"done in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
