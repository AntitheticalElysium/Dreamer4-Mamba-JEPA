#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
root=artifacts/terminal_diversity_diagnostics
reuse=artifacts/stage_a_terminalfix
factorial=artifacts/phase1b_geometry_factorial
prepared=artifacts/phase1b_archive_geometry/preparation/prepared.pt
trajectory_phase2=artifacts/stage_a_s76_terminal_only/direct-attention.2.pt
forks=artifacts/phase1b_causal_diagnostics/paired_trajectory_actions/paired_trajectory_forks.pt

mkdir -p "$root" "$root/matplotlib-cache"
export MPLCONFIGDIR="$root/matplotlib-cache"
exec > >(tee -a "$root/run.log") 2>&1

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for checkpoint/config fidelity"'

factorial_worlds=(
  ordinary_ordinary_005k ordinary_ordinary_020k
  whitened_ordinary_005k whitened_ordinary_020k
  ordinary_terminal_005k ordinary_terminal_020k
  whitened_terminal_005k whitened_terminal_020k
)
factorial_archive_args=()
factorial_policy_args=()
for name in "${factorial_worlds[@]}"; do
  factorial_archive_args+=(--archive-world "$name")
  factorial_policy_args+=(--policy-world "$name")
done

"$python_bin" -m artifacts.evaluate_fatality_direction_delta \
  --prepared "$prepared" \
  --archive-features "$factorial/archive_evaluation/features" \
  --policy-features "$factorial/policy_trajectory_evaluation/features" \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$trajectory_phase2" \
  --forks "$forks" \
  --fork-starts "$root/fork_start_latents.pt" \
  "${factorial_archive_args[@]}" \
  "${factorial_policy_args[@]}" \
  --out "$root/status_quo_factorial"

"$python_bin" -m artifacts.train_terminal_diversity_scaling \
  --phase1a "$reuse/phase1a.pt" \
  --reference-phase1b "$reuse/direct-attention.1b.pt" \
  --out "$root/training"

cells=(k0032_r0 k0032_r1 k0096_r0 k0096_r1 k0192_r0 k0192_r1 k0300_r0)
archive_args=()
archive_delta_args=()
policy_args=()
policy_delta_args=()
for cell in "${cells[@]}"; do
  for step in 005 010 020; do
    name="${cell}_${step}k"
    checkpoint="$root/training/$cell/world_$(printf '%06d' "$((10#$step * 1000))").pt"
    archive_args+=(--world "$name" "$checkpoint")
    archive_delta_args+=(--archive-world "$name")
  done
  name="${cell}_020k"
  policy_args+=(--world "$name" "$root/training/$cell/world_020000.pt")
  policy_delta_args+=(--policy-world "$name")
done

"$python_bin" -m artifacts.evaluate_phase1b_archive_geometry \
  --prepared "$prepared" \
  "${archive_args[@]}" \
  --out "$root/archive_evaluation"

"$python_bin" -m artifacts.probe_paired_trajectory_forks \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$trajectory_phase2" \
  --forks "$forks" \
  "${policy_args[@]}" \
  --out "$root/policy_trajectory_evaluation"

"$python_bin" -m artifacts.evaluate_fatality_direction_delta \
  --prepared "$prepared" \
  --archive-features "$root/archive_evaluation/features" \
  --policy-features "$root/policy_trajectory_evaluation/features" \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$trajectory_phase2" \
  --forks "$forks" \
  --fork-starts "$root/fork_start_latents.pt" \
  "${archive_delta_args[@]}" \
  "${policy_delta_args[@]}" \
  --out "$root/status_quo_scaling"

"$python_bin" -m artifacts.summarize_terminal_diversity_scaling \
  --training "$root/training/training_report.json" \
  --prepared "$prepared" \
  --archive "$root/archive_evaluation/report.json" \
  --archive-features "$root/archive_evaluation/features" \
  --policy "$root/policy_trajectory_evaluation/report.json" \
  --delta "$root/status_quo_scaling/report.json" \
  --out "$root/report.json"

echo "Terminal-diversity diagnostics complete: $root/report.json"
