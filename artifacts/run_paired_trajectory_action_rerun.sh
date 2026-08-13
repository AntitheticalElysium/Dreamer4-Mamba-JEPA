#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
reuse=artifacts/stage_a_terminalfix
trajectory=artifacts/stage_a_s76_terminal_only/direct-attention.2.pt
reference_forks=artifacts/action_conditioning_diagnostics/expanded_forks/direct-attention.outcome_forks.pt
root=artifacts/phase1b_causal_diagnostics
out="$root/paired_trajectory_actions"

"$python_bin" -m artifacts.collect_paired_trajectory_forks \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$trajectory" \
  --reference-forks "$reference_forks" \
  --out "$out"

"$python_bin" -m artifacts.probe_paired_trajectory_forks \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$trajectory" \
  --forks "$out/paired_trajectory_forks.pt" \
  --world baseline_020k "$reuse/direct-attention.1b.pt" \
  --world consequence_world_gradient "$root/consequence_gradient/world_gradient.world.pt" \
  --world consequence_stopped "$root/consequence_gradient/stopped_world_gradient.world.pt" \
  --world ordinary_080k "$root/long_direct/world_080000.pt" \
  --out "$out/probe"

echo "Paired trajectory-action rerun complete: $out/probe/report.json"

