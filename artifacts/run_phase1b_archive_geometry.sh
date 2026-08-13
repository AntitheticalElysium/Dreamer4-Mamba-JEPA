#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
root=artifacts/phase1b_archive_geometry
reuse=artifacts/stage_a_terminalfix
ordinary=artifacts/phase1b_causal_diagnostics/long_direct

mkdir -p "$root"

"$python_bin" -m artifacts.prepare_phase1b_archive_geometry \
  --phase1a "$reuse/phase1a.pt" \
  --out "$root/preparation"

"$python_bin" -m artifacts.evaluate_phase1b_archive_geometry \
  --prepared "$root/preparation/prepared.pt" \
  --world step_005k "$ordinary/world_005000.pt" \
  --world step_020k "$ordinary/world_020000.pt" \
  --world step_080k "$ordinary/world_080000.pt" \
  --out "$root/ordinary_evaluation"

"$python_bin" -m artifacts.train_whitened_direct_dynamics \
  --phase1a "$reuse/phase1a.pt" \
  --prepared "$root/preparation/prepared.pt" \
  --gate "$root/ordinary_evaluation/whitening_gate.json" \
  --ordinary-control-report "$ordinary/training_report.json" \
  --out "$root/whitened_training"

if "$python_bin" -c \
  'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("eligible") else 1)' \
  "$root/ordinary_evaluation/whitening_gate.json"
then
  "$python_bin" -m artifacts.evaluate_phase1b_archive_geometry \
    --prepared "$root/preparation/prepared.pt" \
    --world step_005k "$root/whitened_training/world_005000.pt" \
    --world step_020k "$root/whitened_training/world_020000.pt" \
    --world step_080k "$root/whitened_training/world_080000.pt" \
    --out "$root/whitened_evaluation"

  "$python_bin" -m artifacts.probe_paired_trajectory_forks \
    --phase1a "$reuse/phase1a.pt" \
    --trajectory-phase2 artifacts/stage_a_s76_terminal_only/direct-attention.2.pt \
    --forks artifacts/phase1b_causal_diagnostics/paired_trajectory_actions/paired_trajectory_forks.pt \
    --world whitened_005k "$root/whitened_training/world_005000.pt" \
    --world whitened_020k "$root/whitened_training/world_020000.pt" \
    --world whitened_080k "$root/whitened_training/world_080000.pt" \
    --out "$root/whitened_policy_trajectories"
else
  echo "Whitening was not run because the preregistered geometry gate failed."
fi

echo "Phase-1B archive geometry queue complete: $root"
