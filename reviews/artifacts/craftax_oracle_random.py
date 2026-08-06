"""Random-encoder floor for the representation oracle.

The oracle compares the latent against constant, timestep and raw-pixel
references, but NOT against an untrained encoder. That floor matters: a random
projection of 12,288 pixels into a few hundred dimensions preserves a great deal
linearly (Johnson-Lindenstrauss), so part of every ``degraded`` R^2 we have
reported may be random projection rather than anything training produced.

This encodes the same expert probe frames with UNTRAINED encoders at each
geometry we have trained, and runs the identical oracle. Interpretation:

  * trained >> random  -> training concentrated the target
  * trained ~= random  -> training neither added nor destroyed it; the value we
                          reported is the projection floor
  * trained <  random  -> training actively discarded it

No training and no new data: this is an instrument control, and it changes no
result already recorded -- it calibrates them.
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

from d4_mamba_jepa.craftax_oracle import load_probe_data, representation_oracle
from d4_mamba_jepa.craftax_runners import craftax_jepa_config
from d4_mamba_jepa.model import D4LiteWorld

PROBE = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/expert_probe_v1.probe_only.pt"
SEED = 20260727


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", type=Path, default=PROBE)
    p.add_argument(
        "--grid", default="16:16,64:16,256:16",
        help="n_latents:d_bottleneck geometries to build untrained",
    )
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "reviews/artifacts/craftax_oracle_random.json")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    probe = load_probe_data(args.probe)
    print(f"probe: {probe.frames.shape[0]} frames / "
          f"{len(np.unique(probe.episode_id))} episodes", flush=True)

    reports = {}
    for item in args.grid.split(","):
        n_latents, d_bottleneck = (int(x) for x in item.split(":"))
        cfg = replace(craftax_jepa_config("transformer"),
                      n_latents=n_latents, d_bottleneck=d_bottleneck)
        torch.manual_seed(SEED)
        world = D4LiteWorld(cfg).to(device).eval()   # UNTRAINED
        label = f"random_n_latents_{n_latents}_d_bottleneck_{d_bottleneck}"
        report = representation_oracle(world, probe)
        report["config"] = {"n_latents": n_latents, "d_bottleneck": d_bottleneck,
                            "latent_dims": cfg.n_spatial * cfg.d_spatial,
                            "trained": False}
        reports[label] = report
        print(f"\n=== {label} (UNTRAINED) ===", flush=True)
        print(f"  audit pass={report['audit']['pass']}", flush=True)
        for group in ("vitals", "inventory"):
            counts: dict[str, int] = {}
            for entry in report[group]["per_target"].values():
                counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
            print(f"  {group}: {counts}", flush=True)
            for name, entry in report[group]["per_target"].items():
                print(f"    {name:<16} {entry['verdict']:<20} "
                      f"latent lin/non {entry['latent_linear_r2']:+.3f}/"
                      f"{entry['latent_nonlinear_r2']:+.3f}", flush=True)
        del world
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.write_text(json.dumps(reports, indent=2, sort_keys=True,
                                      default=float) + "\n")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
