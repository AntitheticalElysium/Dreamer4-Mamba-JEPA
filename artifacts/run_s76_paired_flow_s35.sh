#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
reuse=artifacts/stage_a_terminalfix
out=artifacts/stage_a_s76_paired
baseline_phase2=artifacts/stage_a_s76_terminal_only/direct-attention.2.pt
baseline_forks=artifacts/action_conditioning_diagnostics/expanded_forks/direct-attention.outcome_forks.pt

required=(
  "$reuse/phase1a.pt"
  "$reuse/direct-attention.1b.pt"
  "$reuse/flow-attention.1b.pt"
  "$baseline_phase2"
  "$baseline_forks"
)
for path in "${required[@]}"; do
  test -f "$path" || { echo "missing prerequisite: $path" >&2; exit 1; }
done

# Fresh Phase 2 for both attention arms. The Direct arm uses S76's paired
# observed/generated x alive/dead continuation objective; Flow is the matched
# stochastic control and retains its ordinary tail likelihood.
"$python_bin" -m artifacts.run_stage_a \
  --out "$out" \
  --reuse "$reuse" \
  --arms direct-attention flow-attention \
  --phase2-only

# Re-evaluate both worlds at the exact environment states and actions saved from
# the fixed Direct baseline policy. Flow averages its configured generated draws.
"$python_bin" -m artifacts.evaluate_matched_counterfactual \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$baseline_phase2" \
  --trajectory-arm direct-attention \
  --eval-phase2 "$out/direct-attention.2.pt" \
  --eval-arm direct-attention \
  --forks "$baseline_forks" \
  --out-dir "$out/matched_direct"

"$python_bin" -m artifacts.evaluate_matched_counterfactual \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$baseline_phase2" \
  --trajectory-arm direct-attention \
  --eval-phase2 "$out/flow-attention.2.pt" \
  --eval-arm flow-attention \
  --forks "$baseline_forks" \
  --out-dir "$out/matched_flow"

# Localize any residual Direct failure without comparing its new predictions to
# the old checkpoint's saved predictions; simulator truth must still replay exactly.
"$python_bin" -m artifacts.localize_matched_counterfactual \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$baseline_phase2" \
  --trajectory-arm direct-attention \
  --eval-phase2 "$out/direct-attention.2.pt" \
  --eval-arm direct-attention \
  --forks "$baseline_forks" \
  --features "$out/matched_direct/localization_features.pt" \
  --out "$out/matched_direct/localization.json"

# S35 uses pre-head Phase-1B worlds. For every saved terminal-opportunity state,
# it holds state and action fixed, varies only simulator RNG, and measures the
# empirical successor geometry for Direct and Flow.
"$python_bin" -m artifacts.diagnose_s35_multimodality \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$baseline_phase2" \
  --direct-world "$reuse/direct-attention.1b.pt" \
  --flow-world "$reuse/flow-attention.1b.pt" \
  --forks "$baseline_forks" \
  --draws 64 \
  --flow-samples 64 \
  --out "$out/s35_multimodality.json"

echo "complete: $out"
