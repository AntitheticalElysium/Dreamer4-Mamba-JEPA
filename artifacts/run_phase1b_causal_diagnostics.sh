#!/usr/bin/env bash
set -euo pipefail

python_bin=.venv/bin/python
reuse=artifacts/stage_a_terminalfix
trajectory=artifacts/stage_a_s76_terminal_only/direct-attention.2.pt
flow_phase2=artifacts/equalized_flow_continuation/flow-attention.2.pt
forks=artifacts/action_conditioning_diagnostics/expanded_forks/direct-attention.outcome_forks.pt
root=artifacts/phase1b_causal_diagnostics

required=(
  "$reuse/phase1a.pt"
  "$reuse/direct-attention.1b.pt"
  "$trajectory"
  "$flow_phase2"
  "$forks"
  "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt.manifest.json"
  "artifacts/craftax_support_v1.pt.manifest.json"
)
for path in "${required[@]}"; do
  test -f "$path" || { echo "missing prerequisite: $path" >&2; exit 1; }
done

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for checkpoint-compatible diagnostics"'

"$python_bin" -m artifacts.ablate_phase1b_consequence_gradient \
  --phase1a "$reuse/phase1a.pt" \
  --out "$root/consequence_gradient"

"$python_bin" -m artifacts.probe_direct_phase1b_worlds \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$trajectory" \
  --forks "$forks" \
  --world world_gradient "$root/consequence_gradient/world_gradient.world.pt" \
  --world stopped_world_gradient "$root/consequence_gradient/stopped_world_gradient.world.pt" \
  --out "$root/consequence_gradient_probe"

"$python_bin" -m artifacts.train_long_direct_dynamics \
  --phase1a "$reuse/phase1a.pt" \
  --reference-phase1b "$reuse/direct-attention.1b.pt" \
  --out "$root/long_direct"

"$python_bin" -m artifacts.probe_direct_phase1b_worlds \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$trajectory" \
  --forks "$forks" \
  --world step_005k "$root/long_direct/world_005000.pt" \
  --world step_010k "$root/long_direct/world_010000.pt" \
  --world step_020k "$root/long_direct/world_020000.pt" \
  --world step_040k "$root/long_direct/world_040000.pt" \
  --world step_060k "$root/long_direct/world_060000.pt" \
  --world step_080k "$root/long_direct/world_080000.pt" \
  --out "$root/long_direct_probe"

mkdir -p "$root/flow_localization/probes"
"$python_bin" -m artifacts.localize_flow_phase1b \
  --phase1a "$reuse/phase1a.pt" \
  --trajectory-phase2 "$trajectory" \
  --flow-phase2 "$flow_phase2" \
  --forks "$forks" \
  --out "$root/flow_localization"

echo "Phase-1B causal diagnostics complete: $root"
