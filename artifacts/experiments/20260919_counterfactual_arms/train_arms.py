"""Stage 2: three arms continuing from Raw-10k, differing in ONE thing.

The question is whether LeWM's world fails the gate because it was never trained to distinguish
different actions from the SAME state. Direct was; LeWM was not. But Direct also trains on a
corpus LeWM has never seen, so "new data" and "sibling contrast" are entangled and a two-arm
test cannot separate them.

  A   control            factual windows only -- a faithful continuation of Raw-10k
  A'  data control       factual windows + the fork roots' FACTUAL transition only, one logged
                         action per root, drawn from 17x as many INDEPENDENT roots as B uses
  B   treatment          factual windows + all 17 SAME-ROOT successors from the fork roots

A' and B receive the same number of extra gradient targets per update. They differ only in how
those targets are distributed: A' spreads them over many independent roots, B concentrates them
as siblings of one root. That is the contrast that tests sibling structure rather than volume.

Everything else is pinned identical: the same frozen encoder and pre-encoded latents, the same
Raw-10k world initialization, optimizer, learning rate, schedule, update count, RNG streams,
factual batch order, target space, MSE, and context. SIGReg is absent from all three because the
encoder is frozen, so it has no gradient to the world -- dropping it changes nothing and keeps
the arms identical.

  loss = factual_mse                          (A)
  loss = 0.8 * factual_mse + 0.2 * extra_mse  (A', B)

with `extra` the fork-factual term for A' and the 17-way sibling term for B, both averaged per
root so a root cannot dominate by having many branches.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from d4mj.config import load_recipe
from d4mj.data import _sha256
from d4mj.m03.gate import load_m03_bundle

RAW = ROOT / "artifacts/lewm_gates_20260906/paired/raw/joint/step-010000.pt"
DATASET = ROOT / "artifacts/craftax_support_v2/manifest.json"
RECIPE = ROOT / "d4mj/recipes/lewm_mamba_raw.json"
ARMS = ("A_control", "Ap_data", "B_sibling")


def factual_loss(world, z, actions):
    """The recipe's own objective, on pre-encoded latents: teacher-forced next-latent MSE."""
    pairs = z.unsqueeze(2)                       # [N, T, 1, D]
    predicted = world.teacher(pairs, actions).predicted
    return (predicted.float() - pairs[:, 1:].float()).square().mean()


def branch_loss(world, bundle, z_history, past, targets, actions):
    """Predict `targets` from the SAME prefilled root state, one entry per action in `actions`.

    Averaged over the actions of a root and then over roots, so a root with many branches does
    not outweigh one with few -- the same per-root normalization the ranking objective uses.
    """
    state = bundle.prefill(z_history.unsqueeze(2), past)
    count = actions.shape[1]
    branches = bundle.repeat_state(state, count)
    flat = actions.reshape(-1, 1)
    predicted, _ = bundle.advance(branches, flat)
    got = predicted.latent[:, 0, 0].reshape(len(z_history), count, -1)
    return (got.float() - targets.float()).square().mean(-1).mean(1).mean()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=32, help="factual windows per update")
    parser.add_argument("--fork-roots", type=int, default=4, help="B's roots per update")
    parser.add_argument("--mass", type=float, default=0.2)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    args = parser.parse_args(argv)
    out = args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)

    config = load_recipe(RECIPE)
    bundle, _, _ = load_m03_bundle(RAW, device=args.device, dataset_sha256=_sha256(DATASET))
    bundle.encoder.eval()
    for p in bundle.encoder.parameters():
        p.requires_grad_(False)
    world = bundle.world
    world.train()
    world.requires_grad_(True)
    # The predictor projector's BatchNorm running statistics are FROZEN for every arm.
    #
    # Two reasons, and both matter. `advance` refuses to stream with BN in training mode
    # (`predictor_normalization`) because a branch fan would push 17 correlated samples per root
    # into the running statistics. And if those statistics updated from branch batches in A'/B
    # but only from factual batches in A, the arms would differ in a SECOND way and the contrast
    # would be confounded. Pinning them at the Raw-10k values makes them identical across arms
    # by construction. The affine weights still train.
    frozen_bn = []
    for module in world.predictor_projector.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
            frozen_bn.append(type(module).__name__)

    factual = torch.load(args.cache / "factual.pt", map_location="cpu", weights_only=False)
    fork = torch.load(args.cache / "fork.pt", map_location="cpu", weights_only=False)
    # A' must see the SAME number of extra targets as B, spread over 17x more independent roots.
    extra_roots = args.fork_roots * 17 if args.arm == "Ap_data" else args.fork_roots

    j = config.joint
    optimizer = torch.optim.AdamW(world.parameters(), lr=j.learning_rate, betas=j.betas,
                                  eps=j.optimizer_eps, weight_decay=j.weight_decay)
    generator = torch.Generator().manual_seed(config.seed + 7717)
    device = args.device
    n_factual, n_fork = len(factual["z"]), len(fork["z_branch"])
    curve, started = [], time.time()
    for step in range(args.steps):
        index = torch.randint(n_factual, (args.batch,), generator=generator)
        z = factual["z"][index].to(device)
        actions = factual["actions"][index].to(device)
        loss = base = factual_loss(world, z, actions)
        extra = None
        if args.arm != "A_control":
            pick = torch.randint(n_fork, (extra_roots,), generator=generator)
            zh = fork["z_history"][pick].to(device)
            past = fork["past_actions"][pick].to(device)
            if args.arm == "Ap_data":
                target = fork["z_branch"][pick, fork["factual_action"][pick]].to(device)[:, None]
                act = fork["factual_action"][pick].to(device)[:, None]
            else:
                target = fork["z_branch"][pick].to(device)
                act = torch.arange(17, device=device).expand(extra_roots, 17)
            extra = branch_loss(world, bundle, zh, past, target, act)
            loss = (1 - args.mass) * base + args.mass * extra
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world.parameters(), j.grad_clip)
        optimizer.step()
        if (step + 1) % 100 == 0:
            row = {"step": step + 1, "loss": float(loss.detach()),
                   "factual": float(base.detach()),
                   "extra": None if extra is None else float(extra.detach()),
                   "seconds": round(time.time() - started, 1)}
            curve.append(row)
            print(json.dumps({"arm": args.arm, **row}), flush=True)
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            torch.save({"world": world.state_dict(), "step": step + 1, "arm": args.arm,
                        "extra_roots_per_update": extra_roots,
                        "extra_targets_per_update": extra_roots * (1 if args.arm == "Ap_data" else 17),
                        "mass": args.mass, "parent": _sha256(RAW)},
                       out / f"world_{step + 1:06d}.pt")
    (out / "curve.json").write_text(json.dumps(
        {"schema": "d4mj_cf_arm_curve_v1", "arm": args.arm, "steps": args.steps,
         "frozen_batchnorm": frozen_bn,
         "batch": args.batch, "extra_roots_per_update": extra_roots,
         "extra_targets_per_update": extra_roots * (1 if args.arm == "Ap_data" else 17),
         "mass": args.mass, "parent_sha256": _sha256(RAW), "curve": curve}, indent=2) + "\n")
    print(json.dumps({"status": "arm_complete", "arm": args.arm}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
