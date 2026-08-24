#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
reuse=artifacts/stage_a_terminalfix
out=artifacts/stage_a_terminal_dynamics
expanded="$out/expanded_forks"
baseline_phase2=artifacts/stage_a_s76_terminal_only/direct-attention.2.pt
baseline_forks=artifacts/action_conditioning_diagnostics/expanded_forks/direct-attention.outcome_forks.pt

"$python_bin" -m artifacts.run_stage_a \
  --out "$out" \
  --reuse "$reuse" \
  --arms direct-attention \
  --terminal-dynamics-mass 0.3333333333333333 \
  --phase2-only

"$python_bin" -m artifacts.action_shuffle_dev \
  --phase1a "$reuse/phase1a.pt" \
  --phase2 "$out/direct-attention.2.pt" \
  --out "$out/action_shuffle_dev.json"

"$python_bin" -m artifacts.evaluate_matched_counterfactual \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$baseline_phase2" \
  --eval-arm direct-attention \
  --eval-phase2 "$out/direct-attention.2.pt" \
  --forks "$baseline_forks" \
  --out-dir "$expanded"
