"""Do the 2x2 worlds move the rows the gate actually fails on?

The one-step 2x2 already scored the M03 successor-state panel and `u->u` led it. What it
never scored is the rest of the gate-facing panel: coarse outcomes including death and
damage, within-root action ranking, action-effect equivalence, and the matched
root-plus-action / action-only / shuffled-action floors. Those are the rows the capability
gate reads, and a successor-state win does not imply them.

So run the gate's own `_outcome_report` over the 2x2 arms' features. Nothing is
reimplemented: the same function that produces the sealed panel produces these numbers, so
they sit on one scale with the existing Direct and TC results. Direct features are reused
from the published run rather than recomputed.

Because the worlds were trained on three action-pairs, generated successors are produced at
three memory lengths and the panel is run at each:

    standard        prefill 4 frames, advance once -> 4 pairs (one beyond training)
    trained_length  prefill 3 frames, advance once -> 3 pairs (exactly as trained)
    reset           start from the last observed latent -> 1 pair (M03's own control)

Frozen encoder, frozen worlds. Trains nothing, changes no objective, authorizes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))

from d4mj.data import _sha256, atomic_manifest
from d4mj.m03.cache import ArtifactCache, resolve_payload, use_cache
from d4mj.m03.gate import M03Settings, _outcome_report, load_m03_bundle

CHECKPOINT = ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/joint/step-010000.pt"
TRANSITION = ROOT / "artifacts/experiments/20260917_state_transition/evidence"
RUN = ROOT / "artifacts/lewm_gates_20260916/m03_window"
# prefill frames -> pairs of memory carried into the single advance
CONDITIONS = {"standard": 4, "trained_length": 3, "reset": 1}


@torch.no_grad()
def encode(bundle, pca, rows, settings, batch=16):
    """Root and observed-successor latents in both z and u space."""
    context, successors = rows["context"], rows["successors"]
    rz, ru, sz, su = [], [], [], []
    for start in range(0, len(context), batch):
        end = min(len(context), start + batch)
        frame = context[start:end, -settings.lewm_context:].to(bundle.device)
        z, _, patch = bundle.encoder.export(frame, grid=4)
        rz.append(z[:, :, 0].cpu())
        ru.append(project(pca, patch.flatten(2).cpu()))
        flat = successors[start:end].reshape(-1, 1, *successors.shape[-3:]).to(bundle.device)
        z2, _, patch2 = bundle.encoder.export(flat, grid=4)
        sz.append(z2[:, 0, 0].reshape(end - start, 17, -1).cpu())
        su.append(project(pca, patch2.flatten(2).cpu())[:, 0].reshape(end - start, 17, -1))
    return {"root_z": torch.cat(rz), "root_u": torch.cat(ru),
            "observed_z": torch.cat(sz), "observed_u": torch.cat(su)}


def project(pca, flat):
    out = torch.zeros(*flat.shape[:-1], pca["components"])
    out[..., :pca["rank"]] = (flat - pca["mean"]) @ pca["basis"]
    return out


@torch.no_grad()
def generate(bundle, roots, past_actions, frames, batch=16):
    """Fan every action off a root, at a chosen prefill depth."""
    all_actions = torch.arange(bundle.n_actions, device=bundle.device, dtype=torch.long)
    out = []
    for start in range(0, len(roots), batch):
        end = min(len(roots), start + batch)
        context = roots[start:end, -frames:].unsqueeze(2).to(bundle.device)
        if frames == 1:
            state = bundle.world.start(context)
        else:
            state = bundle.prefill(context, past_actions[start:end, -(frames - 1):].to(bundle.device))
        branches = bundle.repeat_state(state, bundle.n_actions)
        action = all_actions.repeat(end - start)[:, None]
        predicted, _ = bundle.advance(branches, action)
        out.append(predicted.latent[:, 0].reshape(end - start, 17, -1).cpu())
    return torch.cat(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "gate_facing.json"
    if destination.exists():
        raise FileExistsError(f"gate_facing: refusing to replace {destination}")

    settings = M03Settings()
    cache = torch.load(TRANSITION / "state_cache.pt", map_location="cpu", weights_only=False)
    pca = cache["pca"]
    sidecar = torch.load(RUN / "sidecar/sidecar.probe_only.pt", map_location="cpu", weights_only=False)
    stored = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    bundle, _, _ = load_m03_bundle(CHECKPOINT, device=args.device,
                                   dataset_sha256=stored["dataset"]["sha256"])
    del stored

    # Full context latents, so a shallower prefill is a slice rather than a re-encode.
    latents = {}
    for split in ("train", "dev"):
        rows = sidecar["splits"][split]
        wide = dict(rows)
        latents[split] = encode(bundle, pca, wide, settings)
        latents[split]["past_actions"] = rows["past_actions"][:, -settings.lewm_context + 1:]
        print(json.dumps({"stage": "encoded", "split": split}), flush=True)

    # Direct arms, reused from the published run rather than recomputed.
    direct = {}
    store = ArtifactCache(ROOT / "artifacts/lewm_gates_20260906/cache.sqlite3",
                          settings=settings, device="cpu")
    with use_cache(store):
        for arm in ("direct_mamba", "direct_attention"):
            paths = {s: RUN / f"features/{arm}.{s}.pt" for s in ("train", "dev")}
            if not all(p.exists() for p in paths.values()):
                continue
            f = {s: resolve_payload(torch.load(p, map_location="cpu", weights_only=False))["features"]
                 for s, p in paths.items()}
            direct[arm] = {"train_projected": f["train"]["projected"], "dev_projected": f["dev"]["projected"],
                           "train_observed_successor": f["train"]["observed_successor"],
                           "dev_observed_successor": f["dev"]["observed_successor"],
                           "dev_generated_successor": f["dev"]["generated_successor"]}
            print(json.dumps({"stage": "reused_direct", "arm": arm}), flush=True)

    report = {"schema": "d4mj_gate_facing_v1",
              "question": "do the 2x2 worlds move the rows the gate fails on?",
              "scope": "the gate's own _outcome_report over frozen 2x2 arms; Direct reused",
              "trained_pairs": 3, "conditions": CONDITIONS,
              "checkpoint_sha256": _sha256(CHECKPOINT), "script_sha256": _sha256(Path(__file__)),
              "panels": {}, "m4_authorized": False}

    for condition, frames in CONDITIONS.items():
        arms = dict(direct)
        for space in ("z", "u"):
            path = TRANSITION / f"world_{space}_{space}.pt"
            bundle.world.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
            bundle.world.eval()
            entry = {"train_projected": latents["train"][f"root_{space}"][:, -1],
                     "dev_projected": latents["dev"][f"root_{space}"][:, -1],
                     "train_observed_successor": latents["train"][f"observed_{space}"],
                     "dev_observed_successor": latents["dev"][f"observed_{space}"],
                     "dev_generated_successor": generate(
                         bundle, latents["dev"][f"root_{space}"], latents["dev"]["past_actions"], frames)}
            arms[f"{space}->{space}"] = entry
            print(json.dumps({"stage": "generated", "condition": condition, "arm": space}), flush=True)
        report["panels"][condition] = _outcome_report(
            sidecar["splits"]["train"], sidecar["splits"]["dev"], arms, settings, device=args.device)
        print(json.dumps({"stage": "panel", "condition": condition}), flush=True)
    atomic_manifest(destination, report)
    print(json.dumps({"status": "complete", "report": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
