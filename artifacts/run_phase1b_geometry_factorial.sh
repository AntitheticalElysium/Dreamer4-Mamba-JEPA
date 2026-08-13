#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
root=artifacts/phase1b_geometry_factorial
reuse=artifacts/stage_a_terminalfix
control=artifacts/phase1b_causal_diagnostics/long_direct
prepared=artifacts/phase1b_archive_geometry/preparation/prepared.pt

mkdir -p "$root"

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for checkpoint/config fidelity"'

"$python_bin" -m artifacts.train_phase1b_geometry_factorial \
  --phase1a "$reuse/phase1a.pt" \
  --reference-phase1b "$reuse/direct-attention.1b.pt" \
  --prepared "$prepared" \
  --cell whitened ordinary \
  --cell ordinary terminal \
  --cell whitened terminal \
  --out "$root/training"

"$python_bin" -m artifacts.evaluate_phase1b_archive_geometry \
  --prepared "$prepared" \
  --world ordinary_ordinary_005k "$control/world_005000.pt" \
  --world ordinary_ordinary_020k "$control/world_020000.pt" \
  --world whitened_ordinary_005k "$root/training/whitened_ordinary/world_005000.pt" \
  --world whitened_ordinary_020k "$root/training/whitened_ordinary/world_020000.pt" \
  --world ordinary_terminal_005k "$root/training/ordinary_terminal/world_005000.pt" \
  --world ordinary_terminal_020k "$root/training/ordinary_terminal/world_020000.pt" \
  --world whitened_terminal_005k "$root/training/whitened_terminal/world_005000.pt" \
  --world whitened_terminal_020k "$root/training/whitened_terminal/world_020000.pt" \
  --out "$root/archive_evaluation"

"$python_bin" -m artifacts.probe_paired_trajectory_forks \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 artifacts/stage_a_s76_terminal_only/direct-attention.2.pt \
  --forks artifacts/phase1b_causal_diagnostics/paired_trajectory_actions/paired_trajectory_forks.pt \
  --world ordinary_ordinary_005k "$control/world_005000.pt" \
  --world ordinary_ordinary_020k "$control/world_020000.pt" \
  --world whitened_ordinary_005k "$root/training/whitened_ordinary/world_005000.pt" \
  --world whitened_ordinary_020k "$root/training/whitened_ordinary/world_020000.pt" \
  --world ordinary_terminal_005k "$root/training/ordinary_terminal/world_005000.pt" \
  --world ordinary_terminal_020k "$root/training/ordinary_terminal/world_020000.pt" \
  --world whitened_terminal_005k "$root/training/whitened_terminal/world_005000.pt" \
  --world whitened_terminal_020k "$root/training/whitened_terminal/world_020000.pt" \
  --out "$root/policy_trajectory_evaluation"

"$python_bin" -m artifacts.summarize_phase1b_geometry_factorial \
  --training "$root/training/training_report.json" \
  --archive "$root/archive_evaluation/report.json" \
  --policy "$root/policy_trajectory_evaluation/report.json" \
  --out "$root/report.json"

echo "Phase-1B geometry factorial complete: $root/report.json"
