"""Run the complete M03 suite over the matched 10k `z->z` and `u->u` worlds.

The 2x2 compared a joint-trained export `z` against a label-free patch PCA `u` at 4,000
updates on a one-step panel. This evaluates the same comparison at the full budget through
the gate's *own* suite -- primary, memory supplement and all four historical panels -- so
the richer-state intervention is measured on the rows the gate actually reports.

`u` cannot be expressed as a LeWM checkpoint: `z` is the projection of CLS while `u` comes
off the pooled patch grid, so no encoder state dict represents it. The adapter is therefore
an **encoder shim** presenting the ordinary interface -- `projected_and_cls` and `__call__`
-- while emitting `u`. Every gate encode path (`_encode_lewm`, `encode_memory`, the
historical loop) then works unchanged, which is the point: nothing about the evaluation is
reimplemented for this arm.

`load_m03_bundle` is patched to build (frozen 10k encoder [+ shim], fresh 10k world) from a
*world* file. The gate keys its feature cache on `_sha256(checkpoint_path)`, and the two
world files hash differently, so the arms never share cache entries.

**This is a diagnostic, not a sealed gate result.** The patched loader bypasses the
checkpoint identity, recipe and source checks that make a gate run sealed, and the arms are
frozen-encoder worlds rather than jointly trained systems. It is not a Raw-versus-TC
comparison: both arms share one TC-consecutive encoder and train on MSE alone.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))

from d4mj.m03 import gate as gate_module

ENCODER = ROOT / "artifacts/lewm_gates_20260916/paired_window/raw/joint/step-010000.pt"
EVIDENCE = Path(__file__).parent / "evidence"


class PatchPCAEncoder(nn.Module):
    """The frozen encoder's interface, emitting the fixed patch-PCA state instead of `z`.

    Only the projected slot changes. CLS is passed through untouched, so a static panel
    still reports this arm's CLS exactly as it would for any other.
    """

    def __init__(self, encoder, pca):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("mean", pca["mean"].clone())
        self.register_buffer("basis", pca["basis"].clone())
        self.components, self.rank = pca["components"], pca["rank"]
        self.settings = encoder.settings

    def project(self, flat):
        out = flat.new_zeros(*flat.shape[:-1], self.components)
        out[..., :self.rank] = (flat - self.mean.to(flat.device)) @ self.basis.to(flat.device)
        return out

    def projected_and_cls(self, frames):
        _, cls, patch = self.encoder.export(frames, grid=4)
        u = self.project(patch.flatten(2))
        return u.unsqueeze(2), cls

    def forward(self, frames):
        return self.projected_and_cls(frames)[0]

    def freeze(self):
        self.encoder.freeze()
        return self

    @property
    def backbone(self):
        return self.encoder.backbone


def install(world_files, pca, device):
    """Patch the bundle loader so a *world* file selects the arm."""
    real_load = gate_module.load_m03_bundle
    state = {}

    def patched(path, *, device, dataset_sha256, frozen_eval_proof=None):
        path = Path(path)
        if "bundle" not in state:
            bundle, payload, source = real_load(ENCODER, device=device, dataset_sha256=dataset_sha256)
            state.update(bundle=bundle, payload=payload, source=source, encoder=bundle.encoder,
                         shim=PatchPCAEncoder(bundle.encoder, pca).to(device).eval())
        bundle, payload, source = state["bundle"], state["payload"], state["source"]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        bundle.world.load_state_dict(saved["state_dict"])
        bundle.world.eval()
        space = saved["source"]
        bundle.encoder = state["shim"] if space == "u" else state["encoder"]
        bundle.encoder.eval()
        # Distinguish the arms in every identity the gate derives from the payload; the
        # cache additionally keys on the world file's own hash through `_sha256(path)`.
        marked = dict(payload)
        marked["recipe_id"] = hashlib.sha256(
            (payload["recipe_id"] + ":" + space).encode()).hexdigest()
        return bundle, marked, source

    gate_module.load_m03_bundle = patched
    return real_load


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EVIDENCE / "m03")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/craftax_support_v2/manifest.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/lewm_gates_20260906/cache.sqlite3")
    parser.add_argument("--cache-compatibility", type=Path,
                        default=ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/compatibility.json")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE,
                        help="directory holding state_cache.pt and the two world files")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)

    cache = torch.load(args.evidence / "state_cache.pt", map_location="cpu", weights_only=False)
    worlds = {s: args.evidence / f"world_{s}_{s}.pt" for s in ("z", "u")}
    for space, path in worlds.items():
        if not path.exists():
            raise FileNotFoundError(f"matched_10k: {path} is missing; train the worlds first")
    install(worlds, cache["pca"], args.device)
    argv = ["--raw-checkpoint", str(worlds["z"]), "--tc-checkpoint", str(worlds["u"]),
            "--dataset", str(args.dataset), "--cache", str(args.cache),
            "--cache-compatibility", str(args.cache_compatibility),
            "--out", str(args.out), "--device", args.device, *args.extra]
    print(json.dumps({"stage": "suite_start", "raw": "z->z world", "tc": "u->u world",
                      "note": "diagnostic; patched loader bypasses sealed identity checks"}), flush=True)
    return gate_module.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
