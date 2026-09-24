"""Does coordinate-uniform MSE under-allocate to the directions that carry semantics?

The decomposition showed effect R^2 rising .518 -> .78 while every semantic AUC stayed flat. Two
explanations survive that, and they imply different next experiments:

  ALLOCATION   the semantic directions occupy a small share of delta-z variance, so MSE -- which
               weights all 192 coordinates equally -- spends its capacity elsewhere. Then the
               improvement landed OUTSIDE the semantic subspace, and no exposure regime fixes it.
  DOWNSTREAM   the improvement landed INSIDE the semantic subspace too, and the failure is
               somewhere after delta-z geometry entirely.

The test distinguishes them directly. Fit a linear probe per outcome on the REAL effect
(successor z minus root z), take those weight vectors as the semantic directions, orthonormalize
them into a subspace S, then for each arm decompose the predicted effect:

  R2_total     over all 192 coordinates
  R2_in_S      inside the semantic subspace
  R2_out_S     in its orthogonal complement
  share_S      what fraction of true delta-z variance lies in S at all

If R2_out_S improved across arms while R2_in_S did not, allocation is confirmed. If R2_in_S
improved too, the bottleneck is downstream and a from-scratch exposure run is the right next move.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
LADDER = ROOT / "artifacts/experiments/20260919_localization_ladder"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LADDER))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.m03.gate import M03Settings, load_m03_bundle
from evaluate_arms import RAW, DATASET, SIDECAR, generated

OUTCOMES = ("death", "damage", "reward_positive", "achievement_event",
            "inventory_changed", "tile_changed")
ARMS = ("A_control", "Ap_data", "B_sibling")


@torch.inference_mode()
def encode_all(encoder, frames, device, batch=256):
    n, a = frames.shape[:2]
    flat = frames.reshape(-1, *frames.shape[2:])
    out = []
    for i in range(0, len(flat), batch):
        z, _ = encoder.projected_and_cls(flat[i:i + batch].unsqueeze(1).to(device))
        out.append(z[:, 0, 0].cpu())
    return torch.cat(out).reshape(n, a, -1)


def semantic_directions(delta, labels, steps=1500, lr=1e-2, device="cuda"):
    """One linear direction per outcome, fitted on the REAL effect, with a standardized input."""
    x = delta.reshape(-1, delta.shape[-1]).to(device)
    mean, scale = x.mean(0, keepdim=True), x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / scale
    directions, kept = [], []
    for index, name in enumerate(OUTCOMES):
        t = labels[..., index].reshape(-1).float().to(device)
        if t.sum() < 10 or (1 - t).sum() < 10:
            continue
        w = torch.zeros(x.shape[1], 1, device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        for _ in range(steps):
            loss = torch.nn.functional.binary_cross_entropy_with_logits((x @ w)[:, 0] + b, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        directions.append(w.detach()[:, 0] / w.detach().norm().clamp_min(1e-9))
        kept.append(name)
    raw = torch.stack(directions, 1)                      # [D, k] in STANDARDIZED coordinates
    q, _ = torch.linalg.qr(raw)                            # orthonormal basis for the SPAN
    # NOTE: after QR, column i is a Gram-Schmidt residual, NOT "the i-th outcome's direction".
    # The R^2 decomposition below only needs the SPAN, so it is basis-invariant and unaffected.
    # The per-outcome control must use `raw`, not `q` -- indexing q per outcome is meaningless.
    return q.cpu(), kept, mean.cpu(), scale.cpu(), raw.cpu()


def control_auc(raw, delta_dev, labels, mean, scale, kept):
    """Are these directions genuinely predictive on HELD-OUT real effects?

    Without this the whole decomposition is unfalsifiable: a subspace that predicts nothing would
    also show a world failing inside it.
    """
    x = ((delta_dev.reshape(-1, delta_dev.shape[-1]) - mean) / scale)
    y = labels.reshape(-1, labels.shape[-1])
    out = {}
    for index, name in enumerate(kept):
        column = OUTCOMES.index(name)
        score, lab = x @ raw[:, index], y[:, column].bool()
        if lab.all() or (~lab).all():
            continue
        out[name] = {"heldout_auc": round(float((score[lab][:, None] > score[~lab][None, :]).float().mean()), 4),
                     "positives": int(lab.sum()), "rows": int(len(lab))}
    return out


def decompose(true_delta, pred_delta, basis, mean, scale):
    """R^2 overall, inside the semantic subspace, and in its complement -- all standardized."""
    t = ((true_delta.reshape(-1, true_delta.shape[-1]) - mean) / scale)
    p = ((pred_delta.reshape(-1, pred_delta.shape[-1]) - mean) / scale)
    def r2(a, b):
        resid = (a - b).square().sum()
        total = (a - a.mean(0, keepdim=True)).square().sum()
        return float(1 - resid / total)
    t_in, p_in = t @ basis, p @ basis
    t_out, p_out = t - t_in @ basis.T, p - p_in @ basis.T
    share = float(t_in.square().sum() / t.square().sum())
    return {"r2_total": round(r2(t, p), 4), "r2_in_semantic": round(r2(t_in, p_in), 4),
            "r2_out_semantic": round(r2(t_out, p_out), 4),
            "semantic_variance_share": round(share, 5),
            "semantic_dims": int(basis.shape[1]), "total_dims": int(basis.shape[0])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    settings = M03Settings()
    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    bundle, _, _ = load_m03_bundle(RAW, device=args.device, dataset_sha256=_sha256(DATASET))
    encoder = bundle.encoder.eval()
    real, root = {}, {}
    for split in ("train", "dev"):
        real[split] = encode_all(encoder, side[split]["successors"], bundle.device)
        with torch.inference_mode():
            z, _ = encoder.projected_and_cls(
                side[split]["context"][:, -1:].to(bundle.device))
        root[split] = z[:, 0, 0].cpu()
    delta_real = {s: real[s] - root[s][:, None] for s in ("train", "dev")}
    basis, kept, mean, scale, raw = semantic_directions(
        delta_real["train"], side["train"]["outcomes"], device=args.device)
    control = control_auc(raw, delta_real["dev"], side["dev"]["outcomes"], mean, scale, kept)
    print(json.dumps({"stage": "control", **{k: v["heldout_auc"] for k, v in control.items()}}), flush=True)
    print(json.dumps({"stage": "directions", "kept": kept,
                      "dims": int(basis.shape[1])}), flush=True)

    report = {"schema": "d4mj_cf_allocation_v1",
              "semantic_targets": kept,
              "control_heldout_auc": control,
              "what": "semantic directions fitted on the REAL effect; each arm's predicted effect "
                      "decomposed inside vs outside that subspace",
              "arms": {}}
    for arm in ARMS:
        path = args.runs / arm / "world_010000.pt"
        b2, _, _ = load_m03_bundle(RAW, device=args.device, dataset_sha256=_sha256(DATASET))
        b2.world.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["world"])
        b2.encoder.eval(); b2.world.eval()
        gdv, _ = generated(b2, side["dev"], settings)
        entry = decompose(delta_real["dev"], gdv - root["dev"][:, None], basis, mean, scale)
        report["arms"][arm] = entry
        print(json.dumps({"stage": "arm", "arm": arm, **entry}), flush=True)
        del b2
        torch.cuda.empty_cache()

    a, b = report["arms"]["A_control"], report["arms"]["B_sibling"]
    report["verdict"] = {
        "delta_r2_total": round(b["r2_total"] - a["r2_total"], 4),
        "delta_r2_in_semantic": round(b["r2_in_semantic"] - a["r2_in_semantic"], 4),
        "delta_r2_out_semantic": round(b["r2_out_semantic"] - a["r2_out_semantic"], 4),
        "semantic_variance_share": a["semantic_variance_share"],
        "reading": "ALLOCATION CONFIRMED if the out-of-subspace R^2 improved while the in-subspace "
                   "R^2 did not; DOWNSTREAM if the in-subspace R^2 improved too."}
    (args.out / "allocation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "allocation_complete", **report["verdict"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
