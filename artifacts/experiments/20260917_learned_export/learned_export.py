"""Can a 192-D export off this frozen encoder carry successor-state information?

The readout refit showed LeWM's successor-state weakness is upstream of the decoder:
refitting rescued Direct and did nothing for either TC arm, and the gap is already there
on *observed* successor z.  That leaves a question the diagnostics cannot answer: is 192
dimensions from this backbone simply too few, or did joint training choose a poor mapping
into them?

This is a positive control, not a proposed architecture.  The ViT stays frozen and every
export is exactly 192-D, so width is held constant and only the *mapping* varies:

  current       the joint-trained projection z, frozen
  cls_learned   a trained map from the frozen CLS  (192 -> 192)
  patch_learned a trained map from the frozen 4x4 pooled grid (16*192 -> 192)

Each learned export is trained on TRAIN roots to predict successor labels from the root
plus the action, then **frozen**, after which the standard M03 probe and metrics run on it
unchanged -- so every number is comparable to the readout experiment's table.  The learned
exports get extra trainable parameters in the mapping; that asymmetry is the point of an
upper bound, and is why this is a control rather than a comparison.

Read-only over published checkpoints and sidecars.  Trains no world model, changes no gate
artifact, authorizes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))

from d4mj.data import _sha256, atomic_manifest
from d4mj.m03.gate import (STATIC_BINARY, STATIC_CONTINUOUS, M03Settings, _binary_metrics,
                           _fit_probe_many, _one_hot_actions, _regression_metrics,
                           load_m03_bundle)

LATENT = 192


class Export(nn.Module):
    """A 192-D bottleneck over frozen backbone features, plus the head that trains it."""

    def __init__(self, source_dim: int, targets: int, width: int):
        super().__init__()
        self.project = nn.Linear(source_dim, LATENT)
        self.head = nn.Sequential(nn.Linear(LATENT + 17, width), nn.GELU(), nn.Linear(width, targets))

    def forward(self, source, action):
        export = self.project(source)
        return export, self.head(torch.cat((export, action), 1))


@torch.no_grad()
def encode_roots(bundle, context, batch):
    """Frozen z, CLS and pooled patch grid for each root's final context frame."""
    z_rows, cls_rows, patch_rows = [], [], []
    for start in range(0, len(context), batch):
        frames = context[start:start + batch, -1:].to(bundle.device)
        z, cls, patch = bundle.encoder.export(frames, grid=4)
        z_rows.append(z[:, 0, 0].cpu())
        cls_rows.append(cls[:, 0].cpu())
        patch_rows.append(patch[:, 0].flatten(1).cpu())
    return (torch.cat(z_rows), torch.cat(cls_rows), torch.cat(patch_rows))


def unsupervised_export(source_train, source_dev, *, components=LATENT):
    """A 192-D summary with no label supervision: the TRAIN principal subspace.

    The learned maps below are trained on the evaluation task's label family, so a gain
    could be information the backbone holds *or* supervision leaking in. This carries no
    labels at all, so if it also beats the current export the information is really there.
    """
    mean = source_train.mean(0, keepdim=True)
    centred = (source_train - mean).double()
    # Economy SVD on the TRAIN rows only; DEV is projected through, never fitted.
    _, singular, v = torch.linalg.svd(centred, full_matrices=False)
    basis = v[:components].T
    # Whiten. An unwhitened principal basis hands the ridge a null tail with almost no
    # variance, which it then amplifies -- the same defect that produced nonsense R^2 in
    # the predictability bridge. Scaling by the singular values keeps the directions and
    # removes the pathological conditioning.
    scale = singular[:components] / max(len(centred) - 1, 1) ** 0.5
    # Drop the null tail rather than rescaling it. `_fit_probe_many` standardizes every
    # input dimension with a clamp_min, so a near-zero principal direction is stretched to
    # unit variance and injects pure noise -- the defect that produced R^2 of -164 here and
    # nonsense R^2 in the predictability bridge. Zero-filled columns survive that
    # standardization as exact zeros, so the export keeps its declared 192-D width.
    keep = int((scale > scale[0] * 1e-3).sum())
    basis = basis[:, :keep].float()
    mean = mean.float()
    def project(x):
        out = torch.zeros(len(x), components)
        out[:, :keep] = (x - mean) @ basis
        return out
    return {"train": project(source_train), "dev": project(source_dev), "rank": keep}


def train_export(source_train, action_train, truth_train, settings, device, *, steps, width, seed):
    # Seed initialization as well as batching: leaving init to the ambient RNG made every
    # export and arm start from a different draw, which is not a reproducible measurement.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device == "cuda" else []):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = Export(source_train.shape[1], truth_train.shape[1], width).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(source_train)
    for _ in range(steps):
        index = torch.randint(n, (min(256, n),), generator=generator).to(device)
        _, logits = model(source_train[index], action_train[index])
        loss = loss_fn(logits, truth_train[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_gates_20260916/m03_window")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--encode-batch", type=int, default=32)
    parser.add_argument("--seeds", default="0,1,2", help="repeats of every learned export")
    parser.add_argument("--save-tensors", action="store_true", default=True,
                        help="persist exports and probe logits for independent re-scoring")
    parser.add_argument("--limit", action="store_true", help="tiny structural smoke; never a result")
    args = parser.parse_args(argv)
    if args.limit:
        args.steps = 20
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / ("export.smoke.json" if args.limit else "export.json")
    if destination.exists():
        raise FileExistsError(f"learned_export: refusing to replace {destination}")

    settings = M03Settings()
    sidecar_path = args.run / "sidecar/sidecar.probe_only.pt"
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    splits = sidecar["splits"]
    device = args.device
    report = {"schema": "d4mj_learned_export_v1",
              "question": "is 192-D from this frozen backbone too few, or did joint training "
                          "choose a poor mapping into it?",
              "scope": "positive control; frozen ViT, matched 192-D width, learned mapping only",
              "asymmetry": "learned exports hold extra trainable parameters in the mapping; that "
                           "is what makes this an upper bound rather than a matched comparison",
              "sidecar_sha256": _sha256(sidecar_path), "script_sha256": _sha256(Path(__file__)),
              "settings": {"steps": args.steps, "head_width": args.width, "latent": LATENT},
              "arms": {}, "m4_authorized": False}

    action = {s: _one_hot_actions(len(splits[s]["next_binary"]), device=device) for s in ("train", "dev")}
    truth = {s: splits[s]["next_binary"].flatten(0, 1).float().to(device) for s in ("train", "dev")}
    continuous = {s: splits[s]["next_continuous"].flatten(0, 1).float() for s in ("train", "dev")}

    keep: dict = {}
    for arm, slot in (("consecutive", "raw"), ("strided", "tc")):
        checkpoint = ROOT / f"artifacts/lewm_gates_20260916/paired_window/{slot}/joint/step-010000.pt"
        stored = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        bundle, _, _ = load_m03_bundle(checkpoint, device=device,
                                       dataset_sha256=stored["dataset"]["sha256"])
        del stored
        sources = {}
        for split in ("train", "dev"):
            z, cls, patch = encode_roots(bundle, splits[split]["context"], args.encode_batch)
            sources[split] = {"current": z, "cls": cls, "patch": patch}
        del bundle
        if device == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"stage": "encoded", "arm": arm}), flush=True)

        exports = {"current": {s: sources[s]["current"] for s in ("train", "dev")}}
        ranks: dict = {}
        # No labels touch these two; they bound how much of the gain is supervision.
        for name, field in (("cls_pca", "cls"), ("patch_pca", "patch")):
            built = unsupervised_export(sources["train"][field].float(), sources["dev"][field].float())
            ranks[name] = built.pop("rank")
            exports[name] = built
            print(json.dumps({"stage": "unsupervised_export", "arm": arm, "export": name,
                              "retained_components": ranks[name]}), flush=True)
        for name, field in (("cls_learned", "cls"), ("patch_learned", "patch")):
            src_train = sources["train"][field].to(device).repeat_interleave(17, 0)
            for seed_index, seed in enumerate(int(x) for x in args.seeds.split(",")):
                model = train_export(src_train, action["train"], truth["train"], settings, device,
                                     steps=args.steps, width=args.width, seed=settings.seed + seed)
                with torch.no_grad():
                    exports[f"{name}:seed{seed_index}"] = {
                        s: model.project(sources[s][field].to(device)).cpu() for s in ("train", "dev")}
                del model
                print(json.dumps({"stage": "export_trained", "arm": arm, "export": name,
                                  "seed": seed}), flush=True)
            del src_train

        entry = {"checkpoint_sha256": _sha256(checkpoint), "retained_components": ranks, "exports": {}}
        for name, tensors in exports.items():
            train_x = torch.cat((tensors["train"].to(device).float().repeat_interleave(17, 0),
                                 action["train"]), 1)
            dev_x = torch.cat((tensors["dev"].to(device).float().repeat_interleave(17, 0),
                               action["dev"]), 1)
            block = {}
            for hidden, family in ((False, "linear"), (True, "mlp")):
                binary = _fit_probe_many(train_x, truth["train"], {"dev": dev_x}, settings,
                                         hidden=hidden, binary=True)
                regress = _fit_probe_many(train_x, continuous["train"].to(device), {"dev": dev_x},
                                          settings, hidden=hidden, binary=False)
                roots = splits["dev"]["episode"]
                fork = roots.reshape(-1, 1).expand(len(roots), 17).reshape(-1)
                block[family] = {
                    "successor_binary": _binary_metrics(binary["dev"], splits["dev"]["next_binary"],
                                                        roots, STATIC_BINARY, settings),
                    "successor_continuous": _regression_metrics(regress["dev"], continuous["dev"],
                                                                fork, STATIC_CONTINUOUS, settings)}
            entry["exports"][name] = block
            if args.save_tensors:
                keep.setdefault(arm, {})[name] = {
                    "export_dev": tensors["dev"].cpu(),
                    "logits_dev_binary_mlp": binary["dev"].detach().cpu()}
            print(json.dumps({"stage": "probed", "arm": arm, "export": name}), flush=True)
        report["arms"][arm] = entry

    if args.limit:
        report["mode"] = "structural_smoke_not_a_result"
    if args.save_tensors and keep:
        tensor_path = args.out / ("export.smoke.tensors.pt" if args.limit else "export.tensors.pt")
        torch.save(keep, tensor_path)
        report["tensors"] = {"path": str(tensor_path.resolve()), "sha256": _sha256(tensor_path)}
    atomic_manifest(destination, report)
    print(json.dumps({"status": "complete", "report": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
