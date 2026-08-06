"""Latent spectrum: is training collapsing the encoder output to a low-rank subspace?

Both trained objectives fall below the random-init floor on most targets
(inventory mean: random 0.661, T-BASE 0.521, T-JEPA 0.320), so something common
to training degrades the latent on top of the objective effect. Dimensional
collapse is the obvious candidate and our `online_std` monitor CANNOT see it: it
measures per-dimension standard deviation, which stays healthy while all the
variance concentrates in a few directions.

Measures, on the same expert probe frames, per encoder:
  * singular value spectrum of the centred latent
  * effective rank  exp(H(p)) with p = normalized singular values (Roy & Vetterli)
  * participation ratio  (sum s^2)^2 / sum s^4
  * fraction of variance in the top 1/5/10/50 directions
  * per-dimension std and the fraction of near-dead dimensions

A random encoder is the reference: it has no reason to be low rank. If trained
encoders show sharply lower effective rank, "training removes information" has a
mechanism, and it is the mechanism anti-collapse regularizers (VICReg/SIGReg)
exist to prevent. Read-only; no training, no checkpoints written.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from d4_mamba_jepa.checkpoint import load_checkpoint
from d4_mamba_jepa.craftax_oracle import encode_latents, load_probe_data
from d4_mamba_jepa.craftax_runners import craftax_jepa_config
from d4_mamba_jepa.model import D4LiteWorld

PROBE = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/expert_probe_v1.probe_only.pt"
SEED = 20260727


def spectrum_stats(latent: np.ndarray) -> dict:
    x = latent.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    energy = s ** 2
    total = float(energy.sum())
    if total <= 0:
        return {"degenerate": True}
    p = energy / total
    nz = p[p > 1e-12]
    effective_rank = float(np.exp(-(nz * np.log(nz)).sum()))
    participation = float(total ** 2 / float((energy ** 2).sum()))
    cum = np.cumsum(p)
    dim_std = latent.std(axis=0)
    return {
        "dims": int(latent.shape[1]),
        "samples": int(latent.shape[0]),
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / latent.shape[1],
        "participation_ratio": participation,
        "var_top1": float(cum[0]),
        "var_top5": float(cum[min(4, len(cum) - 1)]),
        "var_top10": float(cum[min(9, len(cum) - 1)]),
        "var_top50": float(cum[min(49, len(cum) - 1)]),
        "mean_dim_std": float(dim_std.mean()),
        "dead_dim_fraction": float((dim_std < 1e-3).mean()),
        "top_singular_values": [float(v) for v in s[:10]],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, default=PROBE)
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "reviews/artifacts/craftax_latent_rank.json")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    probe = load_probe_data(args.probe)

    runs = REPO_ROOT / "outputs/d4_mamba_jepa"
    trained = {
        "T-JEPA n16": runs / "craftax_expert_v1/t_jepa/world.pt",
        "M-JEPA n16": runs / "craftax_expert_v1/m_jepa/world.pt",
        "T-JEPA n64": runs / "craftax_capacity/n_latents_64_d_bottleneck_16/world.pt",
        "T-JEPA n256": runs / "craftax_capacity/n_latents_256_d_bottleneck_16/world.pt",
        "T-BASE n16": runs / "craftax_tbase/t_base/world.pt",
        "T-BASE n64": runs / "craftax_tbase/t_base_n_latents_64/world.pt",
    }
    results = {}

    for label, (n_latents, d_bottleneck) in {
        "RANDOM n16": (16, 16), "RANDOM n64": (64, 16), "RANDOM n256": (256, 16),
    }.items():
        cfg = replace(craftax_jepa_config("transformer"),
                      n_latents=n_latents, d_bottleneck=d_bottleneck)
        torch.manual_seed(SEED)
        world = D4LiteWorld(cfg).to(device).eval()
        results[label] = spectrum_stats(encode_latents(world, probe.frames))
        del world
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for label, path in trained.items():
        if not path.is_file():
            print(f"skip {label}: {path} not present yet", flush=True)
            continue
        world, _, _ = load_checkpoint(path, device=device)
        world = world.to(device)
        results[label] = spectrum_stats(encode_latents(world, probe.frames))
        del world
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"{'encoder':<14}{'dims':>6}{'eff.rank':>10}{'frac':>8}"
          f"{'PR':>9}{'var@1':>8}{'var@10':>8}{'dead':>7}", flush=True)
    for label, r in results.items():
        if r.get("degenerate"):
            print(f"{label:<14} DEGENERATE", flush=True)
            continue
        print(f"{label:<14}{r['dims']:>6}{r['effective_rank']:>10.1f}"
              f"{r['effective_rank_fraction']:>8.3f}{r['participation_ratio']:>9.1f}"
              f"{r['var_top1']:>8.3f}{r['var_top10']:>8.3f}"
              f"{r['dead_dim_fraction']:>7.2f}", flush=True)

    args.output.write_text(json.dumps(results, indent=2, sort_keys=True,
                                      default=float) + "\n")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
