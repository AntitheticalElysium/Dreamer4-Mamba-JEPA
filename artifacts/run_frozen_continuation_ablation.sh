#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
out=artifacts/frozen_continuation_ablation
phase1a=artifacts/stage_a_terminalfix/phase1a.pt
world_phase2=artifacts/stage_a_s76_paired/direct-attention.2.pt
trajectory_phase2=artifacts/stage_a_s76_terminal_only/direct-attention.2.pt
forks=artifacts/action_conditioning_diagnostics/expanded_forks/direct-attention.outcome_forks.pt
reference=artifacts/stage_a_s76_paired/matched_direct/matched_report.json
reference_features=artifacts/stage_a_s76_paired/matched_direct/localization_features.pt

required=(
  "$phase1a"
  "$world_phase2"
  "$trajectory_phase2"
  "$forks"
  "$reference"
  "$reference_features"
)
for path in "${required[@]}"; do
  test -f "$path" || { echo "missing prerequisite: $path" >&2; exit 1; }
done

"$python_bin" -m artifacts.ablate_frozen_continuation_heads \
  --phase1a "$phase1a" \
  --world-phase2 "$world_phase2" \
  --trajectory-phase2 "$trajectory_phase2" \
  --forks "$forks" \
  --reference-report "$reference" \
  --reference-features "$reference_features" \
  --checkpoint "$out/training.pt" \
  --out "$out/report.json"

echo "complete: $out/report.json"
