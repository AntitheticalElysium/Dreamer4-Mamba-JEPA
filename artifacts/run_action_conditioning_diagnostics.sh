#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
phase1a=artifacts/stage_a_terminalfix/phase1a.pt
phase2=artifacts/stage_a_s76_terminal_only/direct-attention.2.pt
out_dir=artifacts/action_conditioning_diagnostics
expanded_dir="$out_dir/expanded_forks"

"$python_bin" -m artifacts.action_shuffle_dev \
  --phase1a "$phase1a" \
  --phase2 "$phase2" \
  --out "$out_dir/action_shuffle_dev.json"

"$python_bin" -m artifacts.collect_expanded_counterfactual \
  --phase1a "$phase1a" \
  --phase2 "$phase2" \
  --seed-start 13000 \
  --seeds 128 \
  --out-dir "$expanded_dir"

"$python_bin" -m artifacts.localize_counterfactual \
  --phase1a "$phase1a" \
  --phase2 "$phase2" \
  --forks "$expanded_dir/direct-attention.outcome_forks.pt" \
  --out "$expanded_dir/localization.json"

"$python_bin" -m artifacts.localize_counterfactual_interaction \
  --phase1a "$phase1a" \
  --phase2 "$phase2" \
  --forks "$expanded_dir/direct-attention.outcome_forks.pt" \
  --out "$expanded_dir/interaction.json"
