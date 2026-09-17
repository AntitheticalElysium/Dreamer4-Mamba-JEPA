"""Separate local prediction, feedback drift, history extrapolation and decoder mismatch.

The closed-loop run confounded at least two things. Its worlds were trained on windows of
**three** action-pairs with memory reset every batch, but evaluation prefilled three pairs
and then advanced eight more, so *every* scored prediction sat beyond the trained history
length -- the first one included. Feedback drift and history extrapolation were mixed.

It also scored only three labels: the sampler asked for one frame and one action it never
used, which excluded episode-ending transitions, so `termination` had no positives and was
silently dropped. That is fixed here.

A third confound is measurement, not recurrence. The world trains on raw-coordinate MSE
while the probe standardizes every coordinate, and in the saved TRAIN cache the ten
highest-variance coordinates hold **93.1%** of `u`'s variance against **12.9%** of `z`'s.
Raw MSE on `u` is therefore almost entirely about ten numbers, while the probe weights all
192 equally. Errors are reported split by that partition.

Factorial, on identical windows:

    feed    teacher (true latent in at every step) | generated (its own prediction)
    memory  training (reset every 3 pairs, as trained) | persistent (carry throughout)

Diagnostic over frozen checkpoints and the worlds from `20260917_state_transition`.
Changes no objective, trains nothing, authorizes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))

from d4mj.config import load_recipe
from d4mj.data import _sha256, atomic_manifest, load_joint_corpus
from d4mj.m03.gate import M03Settings, _binary_metrics, _fit_probe_many, load_m03_bundle
from d4mj.state import PredictiveState

CHECKPOINT = ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/joint/step-010000.pt"
TRANSITION = ROOT / "artifacts/experiments/20260917_state_transition/evidence"
CONTEXT = 4
TRAIN_PAIRS = 3
LABELS = ("positive_reward", "negative_reward", "achievement_event", "termination")


def sample_windows(episodes, split, depth, count, seed):
    """CONTEXT frames, then `depth` scored steps -- and no unused tail frame.

    The closed-loop sampler took one frame and one action beyond what it scored, which
    silently excluded every episode-ending transition and left `termination` untestable.
    """
    span = CONTEXT + depth
    pool = [e for e in episodes if e.split == split and e.uniform_eligible and len(e) + 1 >= span]
    rng = torch.Generator().manual_seed(seed)
    frames, actions, truth, owners = [], [], [], []
    for _ in range(count):
        which = int(torch.randint(len(pool), (1,), generator=rng))
        episode = pool[which]
        start = int(torch.randint(len(episode) + 2 - span, (1,), generator=rng))
        frames.append(episode.observations[start:start + span])
        actions.append(episode.actions_taken[start:start + span - 1])
        rows = []
        for k in range(depth):
            i = start + CONTEXT - 1 + k
            event = False if episode.events is None else bool(episode.events[i])
            rows.append([bool(episode.rewards[i] > 0), bool(episode.rewards[i] < 0),
                         event, bool(episode.terminated[i])])
        truth.append(torch.tensor(rows)); owners.append(which)
    return (torch.stack(frames), torch.stack(actions), torch.stack(truth), torch.tensor(owners))


@torch.no_grad()
def encode(bundle, pca, frames, batch=16):
    z_rows, u_rows = [], []
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch].to(bundle.device)
        z, _, patch = bundle.encoder.export(chunk, grid=4)
        z_rows.append(z[:, :, 0].cpu())
        flat = patch.flatten(2).cpu()
        out = torch.zeros(*flat.shape[:-1], pca["components"])
        out[..., :pca["rank"]] = (flat - pca["mean"]) @ pca["basis"]
        u_rows.append(out)
    return torch.cat(z_rows), torch.cat(u_rows)


@torch.no_grad()
def unroll(bundle, latents, actions, depth, *, feed, memory, batch=32):
    """One cell of the factorial. `feed` sets what goes in; `memory` sets when it resets."""
    world = bundle.world
    rows = []
    for start in range(0, len(latents), batch):
        end = min(len(latents), start + batch)
        true = latents[start:end].to(bundle.device)
        state = bundle.prefill(true[:, :CONTEXT].unsqueeze(2),
                               actions[start:end, :CONTEXT - 1].to(bundle.device))
        carried, steps = CONTEXT - 1, []
        for k in range(depth):
            if memory == "training" and carried >= TRAIN_PAIRS:
                # Before the first advance there is no prediction to restart from, so both
                # feeds anchor on the last *observed* latent; afterwards they diverge.
                if feed == "teacher" or not steps:
                    anchor = true[:, CONTEXT + k - 1]
                else:
                    anchor = steps[-1].to(bundle.device)
                state = world.start(anchor.unsqueeze(1).unsqueeze(2))
                carried = 0
            action = actions[start:end, CONTEXT - 1 + k].to(bundle.device)[:, None]
            predicted, _ = world.advance(state, action)
            steps.append(predicted.latent[:, 0, 0].cpu())
            carried += 1
            if feed == "teacher":
                # Same accepted pair update, truth swapped in for the predicted successor.
                state = PredictiveState(true[:, CONTEXT + k].unsqueeze(1).unsqueeze(2).clone(),
                                        predicted.memory, predicted.history, predicted.step)
            else:
                state = predicted
        rows.append(torch.stack(steps, 1))
    return torch.cat(rows)


def errors(predicted, true, order):
    """Normalized MSE per depth, split by the variance partition that raw MSE optimizes."""
    top, rest = order[:10], order[10:]
    out = []
    for k in range(predicted.shape[1]):
        p, t = predicted[:, k].double(), true[:, k].double()
        def norm(idx):
            d = (p[:, idx] - t[:, idx]).square().mean()
            return float(d / t[:, idx].var(0).mean().clamp_min(1e-12))
        out.append({"depth": k + 1, "all": norm(slice(None)), "top10": norm(top), "rest": norm(rest)})
    return out


def decode(predicted, truth, actions, owners, settings, device, depth, names, keep):
    """Native readouts: the probe is fitted on this condition's own latents."""
    out = []
    for k in range(depth):
        y = {s: truth[s][:, k][:, keep].float().to(device) for s in ("train", "dev")}
        act = {s: torch.nn.functional.one_hot(actions[s][:, CONTEXT - 1 + k].long(), 17).float().to(device)
               for s in ("train", "dev")}
        def pack(t, s):
            return torch.cat((t[:, k].float().to(device), act[s]), 1)
        fitted = _fit_probe_many(pack(predicted["train"], "train"), y["train"],
                                 {"dev": pack(predicted["dev"], "dev")}, settings, hidden=True, binary=True)
        out.append({"depth": k + 1,
                    "native": _binary_metrics(fitted["dev"].cpu(), y["dev"].bool().cpu(),
                                              owners["dev"], names, settings)})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/craftax_support_v2")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--train-windows", type=int, default=2048)
    parser.add_argument("--dev-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "drift.json"
    if destination.exists():
        raise FileExistsError(f"drift: refusing to replace {destination}")

    settings = M03Settings()
    cache = torch.load(TRANSITION / "state_cache.pt", map_location="cpu", weights_only=False)
    pca = cache["pca"]
    variance_order = {n: cache[n].reshape(-1, cache[n].shape[-1]).double().var(0)
                      .sort(descending=True).indices for n in ("z", "u")}
    config = load_recipe(ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/resolved_recipe.json")
    stored = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    bundle, _, _ = load_m03_bundle(CHECKPOINT, device=args.device,
                                   dataset_sha256=stored["dataset"]["sha256"])
    del stored
    episodes, _ = load_joint_corpus(args.dataset, config)
    frames, actions, truth, owners = {}, {}, {}, {}
    for split, count in (("train", args.train_windows), ("dev", args.dev_windows)):
        f, a, t, o = sample_windows(episodes, split, args.depth, count, args.seed + len(split))
        frames[split], actions[split], truth[split], owners[split] = f, a, t, o
    del episodes
    keep = [i for i in range(4) if bool(truth["train"][:, :, i].any() and (~truth["train"][:, :, i]).any())]
    names = [LABELS[i] for i in keep]
    positives = {LABELS[i]: int(truth["dev"][:, :, i].sum()) for i in range(4)}
    print(json.dumps({"stage": "sampled", "scored": names, "dev_positives": positives}), flush=True)

    latents = {}
    for split in ("train", "dev"):
        z, u = encode(bundle, pca, frames[split])
        latents[split] = {"z": z, "u": u}
    del frames
    print(json.dumps({"stage": "encoded"}), flush=True)

    report = {"schema": "d4mj_drift_decomposition_v1",
              "question": "separate local prediction, feedback drift, history extrapolation "
                          "and decoder mismatch",
              "trained_pairs": TRAIN_PAIRS, "context": CONTEXT, "depth": args.depth,
              "scored_labels": names, "dev_label_positives": positives,
              "variance_share_top10": {n: float(cache[n].reshape(-1, cache[n].shape[-1]).double()
                                                .var(0).sort(descending=True).values[:10].sum()
                                                / cache[n].reshape(-1, cache[n].shape[-1]).double().var(0).sum())
                                       for n in ("z", "u")},
              "checkpoint_sha256": _sha256(CHECKPOINT), "script_sha256": _sha256(Path(__file__)),
              "cells": {}, "m4_authorized": False}
    for source in ("z", "u"):
        path = TRANSITION / f"world_{source}_{source}.pt"
        bundle.world.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
        bundle.world.eval()
        true = {s: latents[s][source][:, CONTEXT:] for s in ("train", "dev")}
        root = {s: latents[s][source][:, CONTEXT - 1] for s in ("train", "dev")}
        hold = {s: root[s].unsqueeze(1).expand(-1, args.depth, -1) for s in ("train", "dev")}
        report["cells"][f"{source}:persistence"] = {
            "errors": errors(hold["dev"], true["dev"], variance_order[source]),
            "decoding": decode(hold, truth, actions, owners, settings, args.device,
                               args.depth, names, keep)}
        for feed in ("teacher", "generated"):
            for memory in ("training", "persistent"):
                rolled = {s: unroll(bundle, latents[s][source], actions[s], args.depth,
                                    feed=feed, memory=memory) for s in ("train", "dev")}
                report["cells"][f"{source}:{feed}:{memory}"] = {
                    "errors": errors(rolled["dev"], true["dev"], variance_order[source]),
                    "decoding": decode(rolled, truth, actions, owners, settings, args.device,
                                       args.depth, names, keep)}
                print(json.dumps({"stage": "cell", "arm": source, "feed": feed,
                                  "memory": memory}), flush=True)
    atomic_manifest(destination, report)
    print(json.dumps({"status": "complete", "report": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
