"""Does Direct preserve the death mechanic in the encoder's own coordinates?

The previous death evaluation was an own-probe reading: the classifier was fitted on
predicted successors, so it measured whether death is *recoverable* from a prediction
with a freshly fitted decoder, not whether the prediction lands where the true successor
lands. It also lacked an action-only control, so part of its 0.711 was the global action
prior rather than state-conditional structure.

Both are fixable, and the claim that true successor latents were unrecoverable was
simply wrong: `fork_successors/` holds exact successor frames for 961 of the 965
branched roots, with termination labels matching the stored `true_death`.

So this is the transfer test, on the same footing as R_delta:

  1. encode the real successor frames with the repaired tokenizer, each appended to its
     own causal history exactly as `encode_fork_dataset` does
  2. fit the death probe on REAL successors from fit roots only
  3. apply that identical frozen probe to real and predicted successors on held-out roots
  4. report true AUC, predicted AUC, R_death, an action-only floor, and paired
     root-bootstrap uncertainty

The 104 policy forks are excluded: they come from a different collection and have no
stored successor frames.

Splitting matters here. These 965 death roots are a subset of the 5,402 damage roots
the fork arm trained on, so scoring `abm0` on an arbitrary split would score it on its
own training data. Both arms therefore use `seed_split` -- the whole-seed split abm0 was
trained under -- which is honest for abm0 and equally valid for the production world,
which saw none of these roots. Identical held-out roots for both.

The true-successor encoding does not depend on the world, so it is cached once and
reused across arms.
"""

from __future__ import annotations

import glob
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
from evaluate_phase1b_fork import Readout
from reevaluate_phase1b_delta import fit_probe, within_state
from train_phase1b_fork import seed_split

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.representation import Encoder, pack
from d4mj.transition import World, commit_inputs
from legacy import open_checkpoint

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"


@torch.no_grad()
def encode_root(encoder, config, history, successors):
    """History latents, and each successor appended to that same history."""
    z, _, _ = encoder(patchify(history[None], config.patch).to(DEVICE))
    z_history = pack(z, config)[0]
    # chunked: these histories run to 64 frames, so all 17 successors at once is
    # ~17x65 frames and does not fit in 6 GB (encode_fork_dataset had 32-frame windows)
    out = []
    for lo in range(0, 17, 4):
        stacked = torch.stack([torch.cat([history, successors[a][None]])
                               for a in range(lo, min(lo + 4, 17))])
        z, _, _ = encoder(patchify(stacked, config.patch).to(DEVICE))
        out.append(pack(z, config)[:, -1].flatten(1).cpu())
    return z_history.cpu(), torch.cat(out)


@torch.no_grad()
def predict_root(world, config, history, led):
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    z = history[None].to(DEVICE)
    committed, conditioning = commit_inputs(z, rng, config)
    features, _, _ = world(None, led[None].to(DEVICE), committed, conditioning)
    last = features[:, -1:]
    actions = torch.arange(17, device=DEVICE)[None]
    return world.predict(last.expand(1, 17, *last.shape[2:]), actions).flatten(2)[0].cpu()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="production",
                        choices=("production", "abm0", "abm1", "factual", "counterfactual"))
    parser.add_argument("--folder", type=Path, default=None,
                        help="checkpoint directory, for arms outside the seed-0 set; the "
                             "world is loaded exactly as the named arms are")
    parser.add_argument("--tag", default=None,
                        help="output name, defaulting to --arm, so a second seed's arms "
                             "do not overwrite the first's readings")
    parser.add_argument("--milestone", type=int, default=0,
                        help="for the terminal arms: score world_XXXXXX.pt instead of the "
                             "final world.pt, so the first complete pass at 13,592 can be "
                             "read separately")
    args = parser.parse_args()
    base = replace(Config(), n_latents=64, d_bottleneck=16)
    # the arm records the mixer it trained with; hardcoding attention here silently
    # fails to load a mamba checkpoint, and the T-versus-M comparison cannot be run at all
    mixer = "attention"
    report_path = (args.folder / "training_report.json") if args.folder else None
    if report_path is not None and report_path.exists():
        mixer = json.loads(report_path.read_text()).get("time_mixer", "attention")
    config = replace(base, transition="direct", time_mixer=mixer)

    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]), part0=encoder)
    encoder.eval()
    if args.folder is not None or args.arm in ("factual", "counterfactual"):
        folder = args.folder or HERE / f"terminal_{args.arm}"
        if args.milestone:
            world = open_checkpoint(folder / f"world_{args.milestone:06d}.pt",
                                    config, "promoted")
        else:
            world = World(config).to(DEVICE)
            world.load_state_dict(torch.load(folder / "world.pt",
                                             weights_only=False)["world"])
            world.eval()
            for parameter in world.parameters():
                parameter.requires_grad_(False)
    elif args.arm == "production":
        world = World(config).to(DEVICE)
        world.load_state_dict(torch.load(HERE / "production_1b" / "world.pt",
                                         weights_only=False)["world"])
        world.eval()
    else:
        world = open_checkpoint(HERE / f"phase1b_{args.arm}_n64" / "world_020000.pt",
                                config, "promoted")

    successors = {}
    for path in sorted(glob.glob(str(HERE / "fork_successors" / "shard-*.pt"))):
        for row in torch.load(path, weights_only=False):
            successors[(int(row["seed"]), int(row["step"]))] = row

    rows = torch.load(HERE / "fork_histories" / "branched_965.pt", weights_only=False)
    rows = [r for r in rows if (int(r["seed"]), int(r["step"])) in successors]
    print(f"{len(rows)} branched roots with exact successor frames", flush=True)

    cache = HERE / "death_transfer_true.pt"
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        true_z, histories, death = blob["true_z"], blob["histories"], blob["death"]
        print("reused the cached true-successor encoding", flush=True)
    else:
        true_z, histories, death = [], [], []
        for row in rows:
            key = (int(row["seed"]), int(row["step"]))
            history, branch = encode_root(encoder, base, row["frames"],
                                          successors[key]["successors"])
            true_z.append(branch)
            histories.append(history)
            death.append(successors[key]["terminated"].float())
        true_z = torch.stack(true_z).float()
        death = torch.stack(death).numpy()
        torch.save({"true_z": true_z, "histories": histories, "death": death}, cache)
    pred_z = torch.stack([predict_root(world, config, h, r["led_to_action"].long())
                          for h, r in zip(histories, rows)]).float()
    print(f"encoded {tuple(true_z.shape)}; death rate {death.mean():.3f}", flush=True)

    # the split abm0 was trained under, so it is not scored on its own fit roots
    n = len(rows)
    splits = np.array([seed_split(int(r["seed"])) for r in rows])
    fit = torch.from_numpy(splits == "fit")
    tune = torch.from_numpy(splits == "tune")
    test = torch.from_numpy(splits == "test")
    print(f"seed_split: fit {int(fit.sum())}, tune {int(tune.sum())}, "
          f"test {int(test.sum())}", flush=True)
    width = true_z.shape[-1]

    # the probe never sees a prediction during fitting
    probe, _ = fit_probe(true_z[fit].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(death[fit.numpy()].reshape(-1)).float().to(DEVICE),
                         true_z[tune].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(death[tune.numpy()].reshape(-1)).float().to(DEVICE),
                         seed=11)

    # the raw (root, action) scores are kept, not just the per-root AUC they collapse
    # into, so how much of a reading is explained by action identity alone can be
    # measured rather than asserted
    surface = {}

    def read(x, key):
        with torch.no_grad():
            s = torch.cat([probe(x[lo:lo+64].reshape(-1, width).to(DEVICE)).cpu()
                           for lo in range(0, len(x), 64)]).numpy().reshape(-1, 17)
        surface[key] = s
        return within_state(s, death[test.numpy()])

    v_true, v_pred = read(true_z[test], "true"), read(pred_z[test], "pred")

    # action-only floor, same split and probe family
    onehot = torch.eye(17).expand(n, 17, 17).contiguous()
    floor, _ = fit_probe(onehot[fit].reshape(-1, 17).to(DEVICE),
                         torch.from_numpy(death[fit.numpy()].reshape(-1)).float().to(DEVICE),
                         onehot[tune].reshape(-1, 17).to(DEVICE),
                         torch.from_numpy(death[tune.numpy()].reshape(-1)).float().to(DEVICE),
                         seed=11)
    with torch.no_grad():
        s = floor(onehot[test].reshape(-1, 17).to(DEVICE)).cpu().numpy().reshape(-1, 17)
    v_floor = within_state(s, death[test.numpy()])

    a_true, _ = interval(v_true, 17)
    a_pred, (plo, phi) = interval(v_pred, 17)
    a_floor, (flo, fhi) = interval(v_floor, 17)

    generator = np.random.default_rng(20260825)
    k = len(v_true)
    draws = np.array([[(v_pred[i].mean() - 0.5) / (v_true[i].mean() - 0.5),
                       v_pred[i].mean() - v_floor[i].mean()]
                      for i in (generator.integers(0, k, k) for _ in range(10000))])
    band = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    result = {"arm": args.arm, "milestone": args.milestone,
              "per_root_pred": v_pred.tolist(), "per_root_true": v_true.tolist(),
              "per_root_floor": v_floor.tolist(),
              # how many of each test root's seventeen successors are fatal, so the
              # escape-rich and trap-heavy strata can be read off the saved values
              # without re-encoding; the training roots are bimodal on this axis
              "per_root_lethal": death[test.numpy()].sum(1).astype(int).tolist(),
              "scores_true": surface["true"].tolist(),
              "scores_pred": surface["pred"].tolist(),
              "roots": n, "test_roots": int(test.sum()), "scored_roots": int(k),
              "death_rate": float(death.mean()),
              "auc_true": a_true, "auc_pred": a_pred, "auc_pred_ci": [plo, phi],
              "auc_action_only": a_floor, "auc_action_only_ci": [flo, fhi],
              "R_death": (a_pred - 0.5) / (a_true - 0.5),
              "R_death_ci": band(draws[:, 0]),
              "pred_minus_action_only": a_pred - a_floor,
              "pred_minus_action_only_ci": band(draws[:, 1])}
    # `--arm` still reads "production" when a folder was named, which would mislabel
    # every seed-2 reading in the logs
    print(f"\n{args.tag or args.arm}: death transfer, probe on TRUE successors only, "
          f"{k} scored roots")
    print(f"  true successors    {a_true:.4f}")
    print(f"  predicted          {a_pred:.4f} [{plo:.4f}, {phi:.4f}]")
    print(f"  action-only floor  {a_floor:.4f} [{flo:.4f}, {fhi:.4f}]")
    print(f"  R_death            {result['R_death']:.3f} "
          f"[{result['R_death_ci'][0]:.3f}, {result['R_death_ci'][1]:.3f}]")
    print(f"  predicted - floor  {result['pred_minus_action_only']:+.4f} "
          f"[{result['pred_minus_action_only_ci'][0]:+.4f}, "
          f"{result['pred_minus_action_only_ci'][1]:+.4f}]")
    name = f"death_transfer_{args.tag or args.arm}"
    (HERE / f"{name}{'_%06d' % args.milestone if args.milestone else ''}.json").write_text(
        json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
