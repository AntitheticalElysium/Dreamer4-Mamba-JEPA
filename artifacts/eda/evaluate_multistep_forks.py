"""Does the consequence of a chosen action survive the steps after it?

One-step successor fidelity is now well measured and broad-uniform improves it at two
seeds. Compounding is not: it sits at 1.75-1.77x for every model ever trained here and no
intervention has moved it. This scores the axis that actually matters for imagination --
whether a first action's effect persists when the model rolls its own predictions
forward.

Rollout uses the production path, `advance`, the same call `_direct_loss` trains through.
That loss trains exactly two generated states and its docstring caps imagination there,
so depth 3 and 4 are deliberately outside the trained horizon and are reported as such.

Three readings per depth, all on the 197 sealed test roots:

  fidelity   MSE and cosine of the predicted latent against the true one
  contrast   NSE of the CENTRED action effect. Centring across the 17 actions removes
             everything common to the root, leaving only what the choice of action did.
             This is the counterfactual quantity: a model can track the trajectory while
             losing the between-action difference entirely, and MSE would not show it.
The action-effect signal is also measured on the TRUE latents at each depth, which is the
ceiling: if the true contrast itself decays under NOOP continuation, a model tracking it
is not failing.
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

from evaluate_death_transfer import DEVICE, ENCODER, REPORT, encode_root
from train_phase1b_fork import seed_split

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.representation import Encoder, pack
from d4mj.state import WorldState
from d4mj.transition import World, advance, commit_inputs

N_ACTIONS = 17


@torch.no_grad()
def encode_depths(encoder, config, history, successors, depth):
    """True latent of every branch at every depth, each read as the final block of the
    history with that branch's frames so far appended -- the same regime the one-step
    encoding uses, extended one frame at a time."""
    out = []
    for k in range(depth):
        rows = []
        for lo in range(0, N_ACTIONS, 2):
            stacked = torch.stack([torch.cat([history, successors[a, : k + 1]])
                                   for a in range(lo, min(lo + 2, N_ACTIONS))])
            z, _, _ = encoder(patchify_(stacked, config))
            rows.append(pack(z, config)[:, -1].flatten(1).cpu())
        out.append(torch.cat(rows))
    return torch.stack(out, 1)                       # (17, depth, width)


def patchify_(frames, config):
    from d4mj.data import patchify
    return patchify(frames, config.patch).to(DEVICE)


@torch.no_grad()
def roll(world, config, history_z, led, depth):
    """`advance` from the root, once per first action, then NOOP to depth."""
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    z = history_z[None].to(DEVICE)
    committed, conditioning = commit_inputs(z, rng, config)
    features, _, memory = world(None, led[None].to(DEVICE), committed, conditioning)

    out = []
    for action in range(N_ACTIONS):
        state = WorldState(z[:, -1:], memory, z.shape[1], features[:, -1:])
        row = []
        for k in range(depth):
            act = torch.full((1, 1), action if k == 0 else 0, dtype=torch.long, device=DEVICE)
            state, _ = advance(world, state, act, rng, config)
            row.append(state.latent.flatten(1)[0].cpu())
        out.append(torch.stack(row))
    return torch.stack(out)                          # (17, depth, width)


def nse(pred, true):
    """Normalised squared error of the centred action effect, per root."""
    p = pred - pred.mean(0, keepdim=True)
    t = true - true.mean(0, keepdim=True)
    return float((p - t).pow(2).sum() / t.pow(2).sum().clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    config = replace(base, transition="direct", time_mixer="attention")
    world = World(config).to(DEVICE)
    world.load_state_dict(torch.load(args.folder / "world.pt", weights_only=False)["world"])
    world.eval()
    for p in world.parameters():
        p.requires_grad_(False)

    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]), part0=encoder)
    encoder.eval()

    rollouts = torch.load(HERE / "multistep_forks" / "rollouts.pt", weights_only=False)
    # both the frames and the action stream come from the same rows the one-step
    # evaluator uses: these histories are 64 blocks, and fork_actions.pt holds the
    # 32-block form, which would silently mismatch the latent length
    saved = {(int(r["seed"]), int(r["step"])): r for r in
             torch.load(HERE / "fork_histories" / "branched_965.pt", weights_only=False)}
    histories = {k: r["frames"] for k, r in saved.items()}
    actions = {k: r["led_to_action"] for k, r in saved.items()}
    depth = int(rollouts[0]["depth"])
    print(f"{args.tag}: {len(rollouts)} sealed roots, depth {depth}", flush=True)

    true, pred, death = [], [], []
    for n, row in enumerate(rollouts):
        key = (int(row["seed"]), int(row["step"]))
        history = histories[key]
        true.append(encode_depths(encoder, base, history, row["successors"], depth))
        z, _ = encode_root(encoder, base, history, row["successors"][:, 0])
        assert len(actions[key]) == z.shape[0], (
            f"{len(actions[key])} actions against {z.shape[0]} history blocks")
        pred.append(roll(world, config, z, actions[key].long(), depth))
        at = row["terminated_at"].numpy()
        death.append(np.stack([(at >= 0) & (at <= k) for k in range(depth)], 1))
        if (n + 1) % 40 == 0:
            print(f"  {n+1}/{len(rollouts)}", flush=True)
    true, pred = torch.stack(true).float(), torch.stack(pred).float()
    death = np.stack(death)                            # (roots, 17, depth)

    splits = np.array([seed_split(int(r["seed"])) for r in rollouts])
    print(f"  all {len(rollouts)} roots are test-split: {(splits == 'test').all()}\n", flush=True)

    print(f"{'depth':<8}{'MSE':>10}{'cosine':>9}{'contrast NSE':>14}{'true |effect|':>15}"
          f"{'pred |effect|':>15}")
    report = {"tag": args.tag, "depth": depth, "roots": len(rollouts), "per_depth": []}
    for k in range(depth):
        p, t = pred[:, :, k], true[:, :, k]
        mse = float((p - t).pow(2).mean())
        cos = float(torch.nn.functional.cosine_similarity(
            p.reshape(-1, p.shape[-1]), t.reshape(-1, t.shape[-1]), dim=-1).mean())
        contrast = float(np.mean([nse(p[i], t[i]) for i in range(len(p))]))
        # the size of the between-action difference itself, true and predicted: a
        # contrast NSE near 1.0 with a collapsing predicted effect means the model has
        # stopped distinguishing the actions rather than distinguishing them wrongly
        effect_t = float((t - t.mean(1, keepdim=True)).norm(dim=-1).mean())
        effect_p = float((p - p.mean(1, keepdim=True)).norm(dim=-1).mean())
        beyond = "   <- beyond the trained horizon" if k >= 2 else ""
        print(f"  {k+1:<6}{mse:>10.5f}{cos:>9.4f}{contrast:>14.4f}{effect_t:>15.4f}"
              f"{effect_p:>15.4f}{beyond}")
        report["per_depth"].append({"depth": k + 1, "mse": mse, "cosine": cos,
                                    "contrast_nse": contrast,
                                    "true_effect_norm": effect_t,
                                    "pred_effect_norm": effect_p})
    (HERE / f"multistep_{args.tag}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
