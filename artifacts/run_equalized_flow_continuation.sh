#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
reuse=artifacts/stage_a_terminalfix
out=artifacts/equalized_flow_continuation
trajectory=artifacts/stage_a_s76_terminal_only/direct-attention.2.pt
forks=artifacts/action_conditioning_diagnostics/expanded_forks/direct-attention.outcome_forks.pt

required=(
  "$reuse/phase1a.pt"
  "$reuse/flow-attention.1b.pt"
  "$trajectory"
  "$forks"
)
for path in "${required[@]}"; do
  test -f "$path" || { echo "missing prerequisite: $path" >&2; exit 1; }
done

"$python_bin" -m artifacts.run_stage_a \
  --out "$out" \
  --reuse "$reuse" \
  --arms flow-attention \
  --phase2-only

"$python_bin" -m artifacts.evaluate_matched_counterfactual \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$trajectory" \
  --trajectory-arm direct-attention \
  --eval-phase2 "$out/flow-attention.2.pt" \
  --eval-arm flow-attention \
  --forks "$forks" \
  --out-dir "$out/matched_flow"

echo "equalized Flow complete: $out/matched_flow/matched_report.json"
