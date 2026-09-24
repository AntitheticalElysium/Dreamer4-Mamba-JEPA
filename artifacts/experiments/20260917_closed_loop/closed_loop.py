"""Does the richer-state gain survive repeated prediction, or only one step?

The 2x2 showed `u->u` beats the matched `z->z` baseline one step out. A world model is
only useful if that survives feeding its own output back in, so this rolls both arms
closed-loop -- each step consumes the previous *prediction*, never a fresh image -- and
measures two things at every depth:

  fidelity     normalized MSE against the truly-encoded latent, divided by that latent's
               own variance, so `z` space and `u` space are on one scale
  decodability a probe fitted on TRUE latents at that depth, read on the PREDICTED latent

against three baselines that need no world model at all: persistence (hold the root
latent), root-plus-action, and action-only. Persistence is the one that matters: a latent
that simply stops moving scores well on fidelity while predicting nothing.

Frozen encoder, the four worlds trained by `20260917_state_transition`. Diagnostic only.
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
from d4mj.m03.gate import (M03Settings, _binary_metrics, _fit_probe_many, load_m03_bundle)

CHECKPOINT = ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/joint/step-010000.pt"
TRANSITION = ROOT / "artifacts/experiments/20260917_state_transition/evidence"
CONTEXT = 4
LABELS = ("positive_reward", "negative_reward", "achievement_event", "termination")


def sample_windows(episodes, split, depth, count, seed):
    """Fixed windows: CONTEXT frames to prefill, then `depth` closed-loop steps."""
    span = CONTEXT + depth
    pool = [e for e in episodes if e.split == split and e.uniform_eligible and len(e) + 1 >= span + 1]
    rng = torch.Generator().manual_seed(seed)
    frames, actions, truth, owners = [], [], [], []
    for _ in range(count):
        which = int(torch.randint(len(pool), (1,), generator=rng))
        episode = pool[which]
        owners.append(which)
        start = int(torch.randint(len(episode) + 1 - span, (1,), generator=rng))
        frames.append(episode.observations[start:start + span + 1])
        actions.append(episode.actions_taken[start:start + span])
        rows = []
        for k in range(depth):
            i = start + CONTEXT - 1 + k
            event = False if episode.events is None else bool(episode.events[i])
            rows.append([bool(episode.rewards[i] > 0), bool(episode.rewards[i] < 0),
                         event, bool(episode.terminated[i])])
        truth.append(torch.tensor(rows))
    # Windows are drawn with replacement, so several can share an episode. Bootstrap
    # clusters must follow the episode, not the window, or the intervals are too tight.
    return (torch.stack(frames), torch.stack(actions), torch.stack(truth),
            torch.tensor(owners), episodes[0].events is not None)


@torch.no_grad()
def encode(bundle, pca, frames, batch=16):
    z_rows, u_rows = [], []
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch].to(bundle.device)
        z, _, patch = bundle.encoder.export(chunk, grid=4)
        z_rows.append(z[:, :, 0].cpu())
        flat = patch.flatten(2)
        out = torch.zeros(*flat.shape[:-1], pca["components"])
        out[..., :pca["rank"]] = (flat.cpu() - pca["mean"]) @ pca["basis"]
        u_rows.append(out)
    return torch.cat(z_rows), torch.cat(u_rows)


@torch.no_grad()
def rollout(bundle, latents, actions, depth, batch=32):
    """Closed loop: every step consumes the previous prediction, never a fresh frame."""
    rows = []
    for start in range(0, len(latents), batch):
        end = min(len(latents), start + batch)
        context = latents[start:end, :CONTEXT].unsqueeze(2).to(bundle.device)
        past = actions[start:end, :CONTEXT - 1].to(bundle.device)
        state = bundle.prefill(context, past)
        steps = []
        for k in range(depth):
            action = actions[start:end, CONTEXT - 1 + k].to(bundle.device)[:, None]
            state, _ = bundle.advance(state, action)
            steps.append(state.latent[:, 0, 0].cpu())
        rows.append(torch.stack(steps, 1))
    return torch.cat(rows)


def fidelity(predicted, true):
    """Normalized MSE per depth: divided by the true latent's own variance at that depth."""
    out = []
    for k in range(predicted.shape[1]):
        p, t = predicted[:, k].double(), true[:, k].double()
        out.append(float((p - t).square().mean() / t.var(0).mean().clamp_min(1e-12)))
    return out


def decode(predicted, true, root, truth, actions, owners, settings, device, depth, supported):
    """Probe fitted on TRUE latents at a depth, read on the PREDICTED latent there."""
    from d4mj.m03.gate import _one_hot_actions
    names = [n for n, ok in zip(LABELS, supported) if ok]
    keep = [i for i, ok in enumerate(supported) if ok]
    out = []
    for k in range(depth):
        y = {s: truth[s][:, k][:, keep].float().to(device) for s in ("train", "dev")}
        act = {s: torch.nn.functional.one_hot(actions[s][:, CONTEXT - 1 + k].long(), 17).float().to(device)
               for s in ("train", "dev")}
        def pack(t, s):
            return torch.cat((t.float().to(device), act[s]), 1)
        fitted = _fit_probe_many(pack(true["train"][:, k], "train"), y["train"],
                                 {"predicted": pack(predicted["dev"][:, k], "dev"),
                                  "true": pack(true["dev"][:, k], "dev"),
                                  "persistence": pack(root["dev"], "dev")},
                                 settings, hidden=True, binary=True)
        root_fit = _fit_probe_many(pack(root["train"], "train"), y["train"],
                                   {"dev": pack(root["dev"], "dev")}, settings, hidden=True, binary=True)
        action_only = _fit_probe_many(act["train"], y["train"], {"dev": act["dev"]},
                                      settings, hidden=True, binary=True)
        clusters = owners["dev"]
        row = {"depth": k + 1}
        for name, logits in (("predicted", fitted["predicted"]), ("true_upper_bound", fitted["true"]),
                             ("persistence", fitted["persistence"]), ("root_action", root_fit["dev"]),
                             ("action_only", action_only["dev"])):
            row[name] = _binary_metrics(logits.cpu(), y["dev"].bool().cpu(), clusters, names, settings)
        out.append(row)
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
    destination = args.out / "closed_loop.json"
    if destination.exists():
        raise FileExistsError(f"closed_loop: refusing to replace {destination}")

    settings = M03Settings()
    cache = torch.load(TRANSITION / "state_cache.pt", map_location="cpu", weights_only=False)
    pca = cache["pca"]
    config = load_recipe(ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/resolved_recipe.json")
    stored = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    bundle, _, _ = load_m03_bundle(CHECKPOINT, device=args.device,
                                   dataset_sha256=stored["dataset"]["sha256"])
    del stored
    episodes, _ = load_joint_corpus(args.dataset, config)

    frames, actions, truth, owners, has_events = {}, {}, {}, {}, True
    for split, count in (("train", args.train_windows), ("dev", args.dev_windows)):
        f, a, t, o, has_events = sample_windows(episodes, split, args.depth, count, args.seed + len(split))
        frames[split], actions[split], truth[split], owners[split] = f, a, t, o
    del episodes
    supported = [bool(truth["train"][:, :, i].any() and (~truth["train"][:, :, i]).any()) for i in range(4)]
    print(json.dumps({"stage": "sampled", "supported": [n for n, s in zip(LABELS, supported) if s]}), flush=True)

    latents = {}
    for split in ("train", "dev"):
        z, u = encode(bundle, pca, frames[split])
        latents[split] = {"z": z, "u": u}
        print(json.dumps({"stage": "encoded", "split": split, "windows": len(z)}), flush=True)
    del frames

    report = {"schema": "d4mj_closed_loop_v1",
              "question": "does the richer-state gain survive repeated prediction?",
              "scope": "closed loop from CONTEXT frames; each step consumes the previous prediction",
              "depth": args.depth, "context": CONTEXT,
              "windows": {"train": args.train_windows, "dev": args.dev_windows},
              "labels": [n for n, s in zip(LABELS, supported) if s],
              "checkpoint_sha256": _sha256(CHECKPOINT), "script_sha256": _sha256(Path(__file__)),
              "arms": {}, "m4_authorized": False}
    for source, target in (("z", "z"), ("u", "u")):
        path = TRANSITION / f"world_{source}_{target}.pt"
        bundle.world.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
        bundle.world.eval()
        predicted = {s: rollout(bundle, latents[s][source], actions[s], args.depth) for s in ("train", "dev")}
        true = {s: latents[s][target][:, CONTEXT:] for s in ("train", "dev")}
        report["arms"][f"{source}->{target}"] = {
            "world_sha256": _sha256(path),
            "normalized_mse_by_depth": fidelity(predicted["dev"], true["dev"]),
            "persistence_normalized_mse_by_depth": fidelity(
                latents["dev"][target][:, CONTEXT - 1:CONTEXT].expand(-1, args.depth, -1), true["dev"]),
            "decoding": decode(predicted, true,
                               {s: latents[s][target][:, CONTEXT - 1] for s in ("train", "dev")},
                               truth, actions, owners, settings, args.device, args.depth, supported)}
        print(json.dumps({"stage": "scored", "arm": f"{source}->{target}"}), flush=True)
    atomic_manifest(destination, report)
    print(json.dumps({"status": "complete", "report": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
