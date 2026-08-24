#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
reuse=artifacts/stage_a_terminalfix
trajectory=artifacts/stage_a_s76_terminal_only/direct-attention.2.pt
forks=artifacts/action_conditioning_diagnostics/expanded_forks/direct-attention.outcome_forks.pt
paired_features=artifacts/stage_a_s76_paired/matched_direct/localization_features.pt
out=artifacts/direct_transition_stages

required=(
  "$reuse/phase1a.pt"
  "$reuse/direct-attention.1b.pt"
  "$trajectory"
  "$forks"
  "$paired_features"
  "artifacts/stage_a_s76_paired/direct-attention.2.pt"
  "artifacts/stage_a_terminal_dynamics/direct-attention.2.pt"
)
for path in "${required[@]}"; do
  test -f "$path" || { echo "missing prerequisite: $path" >&2; exit 1; }
done

"$python_bin" -m artifacts.localize_direct_transition_stages \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$trajectory" \
  --forks "$forks" \
  --phase1b "$reuse/direct-attention.1b.pt" \
  --phase2 terminal_only "$trajectory" \
  --phase2 paired artifacts/stage_a_s76_paired/direct-attention.2.pt \
  --phase2 terminal_dynamics artifacts/stage_a_terminal_dynamics/direct-attention.2.pt \
  --reference-stage paired \
  --reference-features "$paired_features" \
  --out "$out"

echo "Direct transition localization complete: $out/report.json"
