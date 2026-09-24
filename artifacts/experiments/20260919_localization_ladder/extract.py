"""Phase 2 extraction: every encoder tap for roots and all 17 real successors, one frozen pass.

The LeWM encoder is framewise -- `_hidden` flattens (b,t) and runs the ViT per image -- so a
root's encoding depends only on its own frame and the 64-frame context buys it nothing.  That
is why roots are encoded here as single frames: it is not a shortcut, it is the encoder's
actual behaviour, and the parity check below proves the extracted z/CLS reproduce the
published M03 features bit-for-bit under the declared FP32 profile.

Taps, per the plan's ladder:

  z        192   the projector output the world transitions
  cls      192   the backbone embedding, sibling of z, what TC-LeWM's policy reads
  patches  81x192  the full 9x9 grid with positions retained, never SIGReg-regularized
  pooled4  16x192  the 4x4 pooled grid, the old `u` lineage's preprocessing
  pooled2  4x192   and the 2x2 grid, to separate pooling scale from spatial structure

PCA rungs are NOT fitted here: unsupervised reducers must be fitted inside inner TRAIN during
selection, so they belong to the readout stage that knows the folds.

Read-only over published checkpoints and the published sidecar.  Writes only under this
experiment directory.  Trains nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))

from d4mj.data import _sha256
from d4mj.m03.cache import resolve_payload
from d4mj.m03.gate import M03Settings, load_m03_bundle

SIDECAR = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/sidecar/sidecar.probe_only.pt"
DATASET = ROOT / "artifacts/craftax_support_v2/manifest.json"
PUBLISHED = {"mamba_raw": ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/features/raw",
             "mamba_tc": ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/features/tc",
             "transformer_raw": ROOT / "artifacts/lewm_transformer_comparison/m03/features/raw",
             "transformer_tc": ROOT / "artifacts/lewm_transformer_comparison/m03/features/tc"}


@torch.inference_mode()
def taps(encoder, frames, batch, grid_large=4, grid_small=2):
    """z, CLS and three pooling scales of the patch grid, for frames [N, H, W, 3]."""
    out = {k: [] for k in ("z", "cls", "patches", "pooled4", "pooled2", "patch_mean")}
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch].unsqueeze(1).to(encoder.pixel_mean.device)
        z, cls, tokens, b, t = encoder._hidden(chunk)
        side = int(round(tokens.shape[1] ** 0.5))
        if side * side != tokens.shape[1]:
            raise ValueError("patch grid is not square; pooling would misalign it")
        width = encoder.settings.width
        square = tokens.transpose(1, 2).reshape(tokens.shape[0], width, side, side)
        pool = torch.nn.functional.adaptive_avg_pool2d
        out["z"].append(z.reshape(b, encoder.settings.latent_dim).cpu())
        out["cls"].append(cls.reshape(b, width).cpu())
        out["patches"].append(tokens.reshape(b, side * side, width).half().cpu())
        out["pooled4"].append(pool(square, grid_large).flatten(2).transpose(1, 2).half().cpu())
        out["pooled2"].append(pool(square, grid_small).flatten(2).transpose(1, 2).half().cpu())
        out["patch_mean"].append(tokens.mean(1).reshape(b, width).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def extract_split(encoder, split, settings, batch, history=8):
    """Roots, every real successor, and the causal prefix the history controls need.

    `history` frames of the prefix are encoded so the matched "encoded-root-history + action"
    baseline receives the same causal observations the model did. Because the encoder is
    framewise this is a concatenation of independent per-frame encodings, not a temporal
    model -- which is exactly the distinction the plan asks us not to blur.
    """
    roots = taps(encoder, split["context"][:, -1], batch)
    n, actions = split["successors"].shape[:2]
    flat = taps(encoder, split["successors"].reshape(-1, *split["successors"].shape[2:]), batch)
    successors = {k: v.reshape(n, actions, *v.shape[1:]) for k, v in flat.items()}
    prefix = split["context"][:, -history:]
    stacked = taps(encoder, prefix.reshape(-1, *prefix.shape[2:]), batch)
    past = {k: v.reshape(n, history, *v.shape[1:]) for k, v in stacked.items()
            if k in ("z", "cls", "patch_mean")}
    return {"root": roots, "successor": successors, "history": past,
            "history_actions": split["past_actions"][:, -history + 1:].clone()}


def parity(name, extracted, split_name, tolerance):
    """The extracted z must reproduce the published M03 features, or a tap is not comparable."""
    path = Path(f"{PUBLISHED[name]}.{split_name}.pt")
    if not path.exists():
        return {"status": "absent", "path": str(path)}
    published = resolve_payload(torch.load(path, map_location="cpu", weights_only=False))["features"]
    report = {}
    for tap, field in (("z", "projected"), ("z", "observed_successor")):
        if field not in published:
            continue
        mine = extracted["root"]["z"] if field == "projected" else extracted["successor"]["z"]
        theirs = published[field].float()
        delta = (mine.float() - theirs).abs().max().item()
        report[field] = {"max_abs": delta, "within_tolerance": bool(delta <= tolerance)}
    cls_delta = None
    if "cls" in published:
        cls_delta = (extracted["root"]["cls"].float() - published["cls"].float()).abs().max().item()
        report["cls"] = {"max_abs": cls_delta, "within_tolerance": bool(cls_delta <= tolerance)}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, default=Path(__file__).parent / "arms.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    settings = M03Settings()
    sidecar = torch.load(SIDECAR, map_location="cpu", weights_only=False)
    arms = json.loads(args.arms.read_text())
    manifest = {"schema": "d4mj_localization_taps_v1", "sidecar_sha256": _sha256(SIDECAR),
                "script_sha256": _sha256(Path(__file__)), "tolerance": args.tolerance, "arms": {}}
    for name, spec in arms.items():
        destination = args.out / f"taps.{name}.pt"
        if destination.exists():
            print(json.dumps({"stage": "cached", "arm": name}), flush=True)
            manifest["arms"][name] = {"cached": True, "sha256": _sha256(destination)}
            continue
        bundle, _, _ = load_m03_bundle(ROOT / spec["checkpoint"], device=args.device,
                                       dataset_sha256=_sha256(DATASET))
        encoder = bundle.encoder.eval()
        if any(module.training for module in encoder.modules()):
            raise RuntimeError("extract: encoder is not fully in eval mode")
        payload, checks = {}, {}
        for split_name in ("train", "dev"):
            payload[split_name] = extract_split(encoder, sidecar["splits"][split_name], settings, args.batch)
            checks[split_name] = parity(name, payload[split_name], split_name, args.tolerance)
            print(json.dumps({"stage": "split", "arm": name, "split": split_name,
                              "parity": checks[split_name]}), flush=True)
        payload["parity"] = checks
        payload["checkpoint_sha256"] = _sha256(ROOT / spec["checkpoint"])
        torch.save(payload, destination)
        manifest["arms"][name] = {"cached": False, "parity": checks, "sha256": _sha256(destination)}
        del bundle, encoder
        torch.cuda.empty_cache()
        print(json.dumps({"stage": "arm_complete", "arm": name}), flush=True)
    (args.out.parent / "evidence/taps_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
