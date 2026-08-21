"""Phase-1B translation diagnostic: does the better Z* make consequences learnable?

Frozen encoder, world backbone, Direct transition predictor. Nothing else -- no BC,
reward, continuation, agent adaptation, value, actor or imagination.

The objective is the ordinary Direct latent-dynamics loss. The model is never told
which action damages; it is told the current latent state, the candidate action, and
the true next latent. Damage has to emerge in the predicted successor.

    L = L_ordinary + lambda * L_all17

`L_ordinary` is teacher-forced next-latent prediction across each root's real causal
history; `L_all17` is the same prediction at the root for all 17 simulator-executed
successors. Both are means over target scalars, so neither depends on latent width or
successor count, and one fixed lambda applies to both geometries.

Resumable: optimizer, RMS state, both RNG streams and the step counter are written
atomically every `--resume-every` steps.
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from resume import load_state, save_state

from d4mj.checkpoint import save
from d4mj.config import Config
from d4mj.train import _share_initialisation, _update, optimizer
from d4mj.transition import World, commit_inputs

DEVICE = "cuda"


def seed_split(seed: int) -> str:
    draw = int.from_bytes(hashlib.sha256(f"paired-seed:{seed}".encode()).digest()[:8],
                          "little") % 10
    return "fit" if draw < 7 else ("tune" if draw < 8 else "test")


FULL_HISTORY = 32


def load_forkset(folder: Path, full_only: bool = True):
    """Roots with the complete causal history.

    Roots within 31 steps of an episode start carry a shorter history. Padding them
    would alter their causal context and grouping by length would change batch
    composition between arms, so the 4.7% affected are dropped -- identically in the
    preflight, both training arms and the evaluator.
    """
    rows = []
    for path in sorted(folder.glob("shard-*.pt")):
        rows += torch.load(path, weights_only=False)
    if full_only:
        rows = [r for r in rows if r["z_history"].shape[0] == FULL_HISTORY]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-latents", type=int, required=True)
    parser.add_argument("--suffix", type=str, default="s1")
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--milestones", type=int, nargs="+", default=(5_000, 10_000, 20_000))
    parser.add_argument("--resume-every", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    folder = HERE / f"forkset_{args.suffix}_n{args.n_latents}"
    manifest = json.loads((folder / "manifest.json").read_text())
    report = json.loads((HERE / "capacity6k" /
                         f"n{args.n_latents}d16_{args.suffix}" / "training_report.json").read_text())
    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=args.n_latents, d_bottleneck=16, batch=args.batch,
                     seed=Config().seed)
    rows = load_forkset(folder)
    splits = np.array([seed_split(r["seed"]) for r in rows])
    fit = np.where(splits == "fit")[0]
    print(f"arm {args.n_latents}x16 {args.suffix}: z dim {manifest['z_dim']}, "
          f"{len(rows)} roots, fit {len(fit)}, lambda {args.lam}", flush=True)

    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows])
    steps_per_root = history.shape[1]
    spatial, d = config.n_spatial, config.d_spatial

    # world initialization is seeded identically for both geometries
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(DEVICE)
    opt = optimizer([world], config)
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 1001)
    draw = np.random.default_rng(config.seed + 91)
    actions = torch.arange(17, device=DEVICE)

    args.out.mkdir(parents=True, exist_ok=True)
    resume_path = args.out / "resume.pt"
    begin, extra = load_state(resume_path, {"world": world}, opt, rng, draw)
    curve = list(extra.get("curve", []))
    if begin:
        print(f"resumed from step {begin}", flush=True)

    started = time.time()
    for step in range(begin, args.steps):
        chosen = draw.choice(fit, args.batch, replace=False)
        z = history[chosen].to(DEVICE)
        target = branch[chosen].to(DEVICE)
        led = torch.full((args.batch, steps_per_root), config.n_actions,
                         dtype=torch.long, device=DEVICE)
        committed, conditioning = commit_inputs(
            z.view(args.batch, steps_per_root, spatial, d), rng, config)
        features, _, _ = world(None, led, committed, conditioning)

        ordinary = (world.predict(features[:, :-1], led[:, 1:]).flatten(2)
                    - z[:, 1:]).pow(2).mean()
        last = features[:, -1:]
        fork = (world.predict(last.expand(args.batch, 17, *last.shape[2:]),
                              actions[None].expand(args.batch, -1)).flatten(2)
                - target).pow(2).mean()
        loss = ordinary + args.lam * fork
        _update(opt, loss, [world], config, step)
        curve.append({"step": step + 1, "ordinary": float(ordinary.detach()),
                      "fork": float(fork.detach())})
        if (step + 1) % 500 == 0 or step + 1 == args.steps:
            window = curve[-500:]
            rate = (step + 1) / (time.time() - started + 1e-9)
            print(f"  {step+1}/{args.steps} "
                  f"ord={sum(c['ordinary'] for c in window)/len(window):.5f} "
                  f"fork={sum(c['fork'] for c in window)/len(window):.5f} "
                  f"[{time.time()-started:.0f}s]", flush=True)
        if (step + 1) in tuple(args.milestones):
            save(args.out / f"world_{step+1:06d}.pt", config, part0=world)
        if (step + 1) % args.resume_every == 0 or step + 1 == args.steps:
            save_state(resume_path, step + 1, {"world": world}, opt, rng, draw,
                       extra={"curve": curve[-2000:]})

    (args.out / "training_report.json").write_text(json.dumps({
        "n_latents": args.n_latents, "suffix": args.suffix, "lambda": args.lam,
        "steps": args.steps, "roots": len(rows), "fit_roots": int(len(fit)),
        "z_dim": manifest["z_dim"], "seconds": time.time() - started,
        "curve_tail": curve[-500:]}, indent=2))
    print(f"done in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
