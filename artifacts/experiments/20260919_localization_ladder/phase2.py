"""Phase 2: the dimension-matched encoder/export ladder.

Phase 1 found that successor health -- the quantity `death` is literally thresholded from --
is decodable from the ViT patch grid at R^2 0.92 and from CLS at 0.21. That is suggestive and
not yet a finding, because the patch rung has 15552 dimensions against CLS's 192. A wider
representation can win for no interesting reason.

So every rung here is scored at a MATCHED width. PCA bases are fitted on TRAIN only -- a basis
fitted on DEV would leak the evaluation set into the representation it is meant to judge -- and
each basis is saved so the rung is reproducible.

  patches_pca192    TRAIN-fitted PCA-192 of the full 9x9 grid, positions retained before PCA
  pooled4_pca192    TRAIN-fitted PCA-192 of the 4x4 pooled grid (the old `u` lineage's shape)
  cls               the 192-d backbone embedding, already at width
  z                 the 192-d projector output the world model actually transitions
  patches / pooled4 the unreduced rungs, reported with their parameter counts as context

If PCA-192 of patches keeps health while CLS-192 does not, width is excluded and the loss is
specifically at the CLS/pooling step. If both fall together, the story is width and the patch
result proves nothing about pooling.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.m03.cache import resolve_payload
from readout import DEATH, Fit, evaluate, fit_head, standardize

SIDECAR = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/sidecar/sidecar.probe_only.pt"
DIRECT = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/features/direct_mamba"
SEEDS = (0, 1, 2)
WIDTH = 192


def load_taps(name):
    return torch.load(HERE / f"cache/taps.{name}.pt", map_location="cpu", weights_only=False)


def fit_pca(train, components):
    """TRAIN-only PCA. Returns (mean, basis) and the retained variance ratio."""
    flat = train.reshape(-1, train.shape[-1]).double()
    mean = flat.mean(0)
    centred = flat - mean
    # Economy SVD on the centred matrix is the stable route to the basis at this size.
    _, values, vectors = torch.linalg.svd(centred, full_matrices=False)
    basis = vectors[:components]
    variance = (values ** 2)
    retained = float(variance[:components].sum() / variance.sum())
    return mean.float(), basis.float().T, retained


def apply_pca(x, mean, basis):
    lead = x.shape[:-1]
    return ((x.reshape(-1, x.shape[-1]) - mean) @ basis).reshape(*lead, basis.shape[1])


def rungs_for(arm, taps, basis_dir=None):
    """Successor rungs per split, as {name: {split: tensor}} plus the SAVED PCA contracts.

    The mean and components are written to disk, not merely described: a retained-variance
    number does not let anyone reconstruct the rung.
    """
    out, contracts, bases = {}, {}, {}

    def flat(split, tap):
        value = taps[arm][split]["successor"][tap].float()
        return value.flatten(2) if value.dim() == 4 else value

    for tap in ("z", "cls", "patch_mean"):
        out[tap] = {s: flat(s, tap) for s in ("train", "dev")}
    for tap in ("pooled4", "pooled2", "patches"):
        out[tap] = {s: flat(s, tap) for s in ("train", "dev")}
        mean, basis, retained = fit_pca(out[tap]["train"], WIDTH)
        name = f"{tap}_pca{WIDTH}"
        out[name] = {s: apply_pca(out[tap][s], mean, basis) for s in ("train", "dev")}
        contracts[name] = {"source": tap, "components": WIDTH, "retained_variance": retained,
                           "source_dim": int(out[tap]["train"].shape[-1]),
                           "basis_file": f"pca.{arm}.{name}.pt"}
        bases[name] = {"mean": mean, "basis": basis}
    if basis_dir is not None:
        basis_dir.mkdir(parents=True, exist_ok=True)
        for name, value in bases.items():
            torch.save(value, basis_dir / f"pca.{arm}.{name}.pt")
    return out, contracts


def health_r2(xtr, htr, xdv, hdv, *, family, seed, spec, device):
    from readout import build_head
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = build_head(family, xtr.shape[2:], 1, generator, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate,
                                  weight_decay=spec.weight_decay)
    mean, scale = htr.mean(), htr.std().clamp_min(1e-6)
    for _ in range(spec.steps):
        index = torch.randint(len(xtr), (min(spec.batch, len(xtr)),), generator=generator)
        loss = torch.nn.functional.mse_loss(
            model(xtr[index].to(device))[..., 0], ((htr[index] - mean) / scale).to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        pred = torch.cat([model(xdv[i:i + 128].to(device))[..., 0].cpu()
                          for i in range(0, len(xdv), 128)]) * scale + mean
    residual = ((pred - hdv) ** 2).mean()
    total = ((hdv - hdv.mean()) ** 2).mean()
    return float(1 - residual / total), sum(p.numel() for p in model.parameters())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence/phase2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--arms", default="mamba_raw,mamba_tc")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "ladder.json"
    if destination.exists():
        print(json.dumps({"status": "cached"}), flush=True)
        return 0

    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    y = {s: side[s]["outcomes"].float() for s in ("train", "dev")}
    health = {s: side[s]["next_continuous"][:, :, 0].float() for s in ("train", "dev")}
    spec = Fit(steps=args.steps)
    report = {"schema": "d4mj_phase2_ladder_v1", "width": WIDTH, "steps": args.steps,
              "pca_contract": {}, "rows": []}

    for arm in args.arms.split(","):
        taps = {arm: load_taps(arm)}
        rungs, contracts = rungs_for(arm, taps, basis_dir=args.out / "pca")
        report["pca_contract"][arm] = contracts
        for name, value in rungs.items():
            xtr, xdv = standardize(value["train"], value["dev"])
            r2, r2_params = health_r2(xtr, health["train"], xdv, health["dev"],
                                      family="logistic", seed=0, spec=spec, device=args.device)
            choices = []
            for seed in SEEDS:
                model, _, params = fit_head(xtr, y["train"], family="mlp128", objective="rank",
                                            seed=seed, spec=spec, device=args.device)
                metrics = evaluate(model, xdv, y["dev"], args.device)
                choices.append(metrics)
            row = {"arm": arm, "rung": name, "dim": int(xtr.shape[-1]),
                   "health_r2_linear": r2, "health_parameters": r2_params,
                   "choice_parameters": params,
                   "safe_choice": [m["safe_choice"] for m in choices],
                   "opportunity_roots": choices[0]["opportunity_roots"],
                   "within_root_auc": [round(m["within_root_auc"], 4) for m in choices],
                   "selected_actions_seed0": choices[0]["selected_actions"]}
            report["rows"].append(row)
            print(json.dumps({"stage": "rung", "arm": arm, "rung": name, "dim": row["dim"],
                              "health_r2": round(r2, 3), "safe": row["safe_choice"],
                              "auc": row["within_root_auc"]}), flush=True)
        del taps, rungs

    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "phase2_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
