"""Is the state x action interaction present under Direct's action prior, or absent?

Every terminal arm ended up ranking successors mostly by which action was taken: the
action-identity share of within-root ranking runs 32-42% against 12-16% for the controls
and 6.8% in the truth, and the arms lose exactly on the roots where the marginal is
wrong. That measurement was made on the death probe's scalar output, which cannot tell
two very different failures apart:

  buried   the interaction is there, faithfully, but a much larger action component sits
           on top of it and the probe reads that instead
  absent   the interaction was never learned, and there is nothing underneath

The distinction decides what to do next, and the two answers point opposite ways -- the
first is a scaling problem, the second is a representation problem.

On a balanced grid of R roots by all 17 actions, the two-way decomposition is orthogonal,
so the latent splits exactly:

    X[r,a] = m + beta[r] + alpha[a] + gamma[r,a]
             grand   root    action    interaction

Shares are sums of squared norms across coordinates, never a scalar mean -- a scalar
grand mean is not the least-squares centre of a vector-valued cell and silently
misattributes between-cell variance to the residual.

Three readings:

  1  fidelity   cosine between predicted and true alpha, and between predicted and true
                gamma, with their norm ratios. High alpha cosine beside near-zero gamma
                cosine is `absent`. Good gamma cosine with a small norm ratio is `buried`.

  2  localise   the same shares at four depths inside `predict`, which is where the
                action first enters and where it could come to dominate:

                  pooled     the pooled root state, before the action token exists.
                             Its action share must be exactly zero -- it is computed
                             without the action -- which makes it a free correctness
                             check on the whole decomposition.
                  mixed      spatial states straight after the action-token mixer
                  pre_tanh   the readout output before squashing
                  final      the predicted successor

  3  causal     the variance shares say how big each component is, not whether the small
                one is usable. So rebuild the prediction with its action marginal removed,
                and again with the TRUE marginal grafted in place of the predicted one,
                and re-run the frozen death probe on each. If escape-rich AUC rises, the
                interaction carries real signal that the prior was swamping. If it does
                not move, there is nothing under there to recover.

Read-only: no training, no new data, one forward pass per arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from evaluate_death_transfer import DEVICE, ENCODER, REPORT
from reevaluate_phase1b_delta import fit_probe, within_state
from train_phase1b_fork import seed_split

from d4mj.config import Config
from d4mj.transition import World, commit_inputs

N_ACTIONS = 17


def decompose(x: torch.Tensor) -> dict:
    """Two-way ANOVA of a balanced (roots, actions, dims) grid, per coordinate."""
    x = x.double()
    grand = x.mean((0, 1), keepdim=True)
    root = x.mean(1, keepdim=True) - grand
    action = x.mean(0, keepdim=True) - grand
    inter = x - grand - root - action
    ss = lambda t, n: float((t.pow(2).sum() * n))
    r, a = x.shape[0], x.shape[1]
    parts = {"root": ss(root, a), "action": ss(action, r), "interaction": ss(inter, 1)}
    total = float((x - grand).pow(2).sum())
    # orthogonality is a property of the balanced design, so it is worth asserting
    assert abs(sum(parts.values()) - total) < 1e-6 * max(total, 1.0), "grid is not balanced"
    return {"shares": {k: v / total for k, v in parts.items()},
            "root": root[:, 0], "action": action[0], "interaction": inter, "grand": grand}


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


def stages(world, config, history, led):
    """`predict`, reimplemented once to hand back what it computes on the way."""
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    committed, conditioning = commit_inputs(history[None].to(DEVICE), rng, config)
    features, _, _ = world(None, led[None].to(DEVICE), committed, conditioning)
    last = features[:, -1:].expand(1, N_ACTIONS, *features.shape[2:])
    action = torch.arange(N_ACTIONS, device=DEVICE)[None]

    world_tokens = torch.cat([last[:, :, world.spatial], last[:, :, world.register]], dim=2)
    pooled = world.pool(world_tokens.transpose(2, 3)).transpose(2, 3)
    b, t, s, d = pooled.shape
    tokens = torch.cat([world.action_embed(action)[:, :, None], pooled], dim=2)
    mixed = world.direct_mixer(tokens.reshape(b * t, s + 1, d)).view(b, t, s + 1, d)
    normed = world.direct_norm(mixed[:, :, 1:])
    pre_tanh = world.readout(normed)
    return {"pooled": pooled.flatten(2)[0].cpu(), "mixed": normed.flatten(2)[0].cpu(),
            "pre_tanh": pre_tanh.flatten(2)[0].cpu(),
            "final": torch.tanh(pre_tanh).flatten(2)[0].cpu()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--milestone", type=int, default=0)
    args = parser.parse_args()

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    config = replace(base, transition="direct", time_mixer="attention")
    name = f"world_{args.milestone:06d}.pt" if args.milestone else "world.pt"
    world = World(config).to(DEVICE)
    if args.milestone:
        from legacy import open_checkpoint
        world = open_checkpoint(args.folder / name, config, "promoted")
    else:
        world.load_state_dict(torch.load(args.folder / name, weights_only=False)["world"])
    world.eval()
    for parameter in world.parameters():
        parameter.requires_grad_(False)

    cached = torch.load(HERE / "death_transfer_true.pt", weights_only=False)
    true_z, histories, death = cached["true_z"].float(), cached["histories"], cached["death"]
    rows = torch.load(HERE / "fork_histories" / "branched_965.pt", weights_only=False)
    import glob
    keys = set()
    for path in sorted(glob.glob(str(HERE / "fork_successors" / "shard-*.pt"))):
        for row in torch.load(path, weights_only=False):
            keys.add((int(row["seed"]), int(row["step"])))
    rows = [r for r in rows if (int(r["seed"]), int(r["step"])) in keys]
    assert len(rows) == len(true_z), f"{len(rows)} rows against {len(true_z)} latents"

    splits = np.array([seed_split(int(r["seed"])) for r in rows])
    fit, tune, test = (torch.from_numpy(splits == s) for s in ("fit", "tune", "test"))
    depth = {k: [] for k in ("pooled", "mixed", "pre_tanh", "final")}
    with torch.no_grad():
        for h, r in zip(histories, rows):
            for key, value in stages(world, config, h, r["led_to_action"].long()).items():
                depth[key].append(value)
    depth = {k: torch.stack(v).float() for k, v in depth.items()}
    print(f"{args.tag}: {len(rows)} roots, {int(test.sum())} held out\n", flush=True)

    # ---- 2, localisation: where the action component takes over
    truth = decompose(true_z[test])
    report = {"tag": args.tag, "milestone": args.milestone,
              "shares": {"TRUE successors": truth["shares"]}}
    print(f"{'variance share':<26}{'root':>10}{'action':>10}{'interaction':>14}")
    def show(label, parts):
        s = parts["shares"]
        print(f"  {label:<24}{s['root']:>10.1%}{s['action']:>10.1%}{s['interaction']:>14.1%}")
    show("TRUE successors", truth)
    parts = {}
    for key in ("pooled", "mixed", "pre_tanh", "final"):
        parts[key] = decompose(depth[key][test])
        report["shares"][key] = parts[key]["shares"]
        show(key, parts[key])
    assert parts["pooled"]["shares"]["action"] < 1e-9, (
        "the pooled root state is computed without the action, so its action share must "
        "be zero; a non-zero value means the decomposition is wrong")
    print("  (pooled carries no action by construction -- a check on the decomposition)\n")

    # ---- 1, fidelity: is the interaction faithful, merely small, or absent?
    final = parts["final"]
    print(f"{'against the truth':<26}{'cosine':>10}{'norm ratio':>14}")
    for part in ("action", "interaction"):
        c = cosine(final[part], truth[part])
        ratio = float(final[part].double().norm() / truth[part].double().norm())
        report[f"{part}_cosine"], report[f"{part}_norm_ratio"] = c, ratio
        print(f"  {part:<24}{c:>10.3f}{ratio:>14.3f}")
    print()

    # The latent shares can look right while the probe output is action-dominated, if the
    # prediction ERROR is itself action-structured in whatever subspace the probe reads.
    # So decompose the residual as well, which the shares of the prediction cannot show.
    error = decompose(depth["final"][test] - true_z[test])
    report["error_shares"] = error["shares"]
    report["error_norm_ratio"] = float(
        (depth["final"][test] - true_z[test]).double().norm() / true_z[test].double().norm())
    show("PREDICTION ERROR", error)
    print(f"  error is {report['error_norm_ratio']:.1%} of the true successor norm\n")

    # ---- 3, causal: remove the predicted marginal, then graft the true one
    width = true_z.shape[-1]
    probe, _ = fit_probe(true_z[fit].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(death[fit.numpy()].reshape(-1)).float().to(DEVICE),
                         true_z[tune].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(death[tune.numpy()].reshape(-1)).float().to(DEVICE),
                         seed=11)
    lethal = death[test.numpy()].sum(1).astype(int)
    escape, trap = lethal <= 2, lethal >= 14

    def read(x):
        with torch.no_grad():
            s = torch.cat([probe(x[lo:lo + 64].reshape(-1, width).to(DEVICE)).cpu()
                           for lo in range(0, len(x), 64)]).numpy().reshape(-1, N_ACTIONS)
        return within_state(s, death[test.numpy()])

    # Removing the action marginal moves the input off the distribution the raw probe was
    # fitted on, so reading residualised latents with it is not interpretable. A residual
    # test needs a probe fitted on equivalently residualised TRUE fit/tune successors.
    def residualise(z):
        return z - (z.mean(0, keepdim=True) - z.mean((0, 1), keepdim=True))

    r_probe, _ = fit_probe(
        residualise(true_z[fit]).reshape(-1, width).to(DEVICE),
        torch.from_numpy(death[fit.numpy()].reshape(-1)).float().to(DEVICE),
        residualise(true_z[tune]).reshape(-1, width).to(DEVICE),
        torch.from_numpy(death[tune.numpy()].reshape(-1)).float().to(DEVICE), seed=11)

    def read_with(p, x):
        with torch.no_grad():
            s = torch.cat([p(x[lo:lo + 64].reshape(-1, width).to(DEVICE)).cpu()
                           for lo in range(0, len(x), 64)]).numpy().reshape(-1, N_ACTIONS)
        return within_state(s, death[test.numpy()])

    pred = depth["final"][test]
    variants = {"as predicted": (probe, pred),
                # both rows below are read by the residual-fitted probe, and the
                # residualised truth is the only valid ceiling for them
                "action marginal removed": (r_probe, residualise(pred)),
                "residualised TRUE (ceiling)": (r_probe, residualise(true_z[test])),
                "true marginal grafted": (probe, pred - final["action"].float()
                                          + truth["action"].float()),
                "TRUE successors": (probe, true_z[test])}
    print(f"{'death AUC on held-out roots':<30}{'overall':>9}{'escape-rich':>13}{'trap-heavy':>12}")
    report["auc"] = {}
    for label, (which, x) in variants.items():
        v = read_with(which, x)
        report["auc"][label] = {"overall": float(v.mean()),
                                "escape_rich": float(v[escape].mean()),
                                "trap_heavy": float(v[trap].mean()),
                                "per_root": v.tolist()}
        print(f"  {label:<28}{v.mean():>9.3f}{v[escape].mean():>13.3f}{v[trap].mean():>12.3f}")
    print(f"\n  escape-rich n={int(escape.sum())}, trap-heavy n={int(trap.sum())}")

    tag = f"_{args.milestone:06d}" if args.milestone else ""
    (HERE / f"action_interaction_{args.tag}{tag}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
