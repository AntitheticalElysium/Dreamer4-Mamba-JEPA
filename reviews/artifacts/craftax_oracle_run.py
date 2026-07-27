"""Run the representation oracle on both frozen Craftax worlds.

Stage-2 question, per privileged target: does the trained encoder's latent still
carry the simulator state a competent policy needs (vitals, inventory,
achievements), or has the representation discarded it?

Probe frames come from ``expert.probe`` -- the SAME expert policy that produced
the training replay -- so a ``lost`` or ``degraded`` verdict cannot be explained
away as distribution shift.

The oracle scores every target individually against three references (constant,
timestep-only, and a raw-pixel ceiling in both linear and nonlinear form) and
self-audits on perfect/constant/misaligned/shifted inputs. Read the audit block
FIRST: if it does not pass, no verdict below it means anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from d4_mamba_jepa.checkpoint import load_checkpoint
from d4_mamba_jepa.craftax_oracle import load_probe_data, representation_oracle

PROBE = REPO_ROOT / "d4_mamba_jepa/artifacts/expert/expert_probe_v1.probe_only.pt"
RUN_DIR = REPO_ROOT / "outputs/d4_mamba_jepa/craftax_expert_v1"


def _verdict_counts(group: dict) -> dict:
    counts: dict[str, int] = {}
    for entry in group["per_target"].values():
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=PROBE)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "reviews/artifacts/craftax_oracle.json")
    parser.add_argument("--arms", default="t_jepa,m_jepa")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    probe = load_probe_data(args.probe)
    print(f"probe: {probe.frames.shape[0]} frames / "
          f"{len(np.unique(probe.episode_id))} episodes", flush=True)

    reports = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        started = time.perf_counter()
        world, _, _ = load_checkpoint(args.run_dir / arm / "world.pt", device=device)
        world = world.to(device)
        report = representation_oracle(world, probe)
        # Key by directory: capacity rungs all share one `arm_id`.
        arm_id = f"{arm} ({world.cfg.arm_id}, d_bottleneck={world.cfg.d_bottleneck})"
        report["config"] = {"arm_id": world.cfg.arm_id,
                            "d_bottleneck": world.cfg.d_bottleneck,
                            "latent_dims": world.cfg.n_spatial * world.cfg.d_spatial,
                            "representation_objective": world.cfg.representation_objective}
        reports[arm_id] = report
        audit = report["audit"]
        print(f"\n=== {arm_id} ({time.perf_counter() - started:.0f}s) ===", flush=True)
        print(f"  audit pass={audit['pass']} perfect={audit['perfect_r2']:.3f} "
              f"constant={audit['constant_r2']:.3f} "
              f"misaligned={audit['misaligned_r2']:.3f} "
              f"shift={audit['timestep_shift_r2']:.3f}", flush=True)
        for group in ("vitals", "inventory"):
            print(f"  {group}: {_verdict_counts(report[group])}", flush=True)
            for name, entry in report[group]["per_target"].items():
                print(f"    {name:<16} {entry['verdict']:<20} "
                      f"latent lin/non {entry['latent_linear_r2']:+.3f}/"
                      f"{entry['latent_nonlinear_r2']:+.3f}  "
                      f"pixel lin/non {entry['pixel_linear_r2']:+.3f}/"
                      f"{entry['pixel_nonlinear_r2']:+.3f}  "
                      f"time {entry['timestep_r2']:+.3f}", flush=True)
        aucs = [v["auroc"] for v in report["achievements"].values()
                if not np.isnan(v["auroc"])]
        print(f"  achievements: {len(aucs)} scorable, "
              f"mean AUROC {np.mean(aucs):.3f}" if aucs else
              "  achievements: none scorable", flush=True)

    args.output.write_text(json.dumps(
        {"probe": {"path": str(args.probe),
                   "frames": int(probe.frames.shape[0]),
                   "episodes": int(len(np.unique(probe.episode_id)))},
         "arms": reports}, indent=2, sort_keys=True, default=float) + "\n")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
