#!/usr/bin/env bash
set -euo pipefail

root=artifacts/phase1b_causal_diagnostics

.venv/bin/python -m artifacts.report_flow_phase1b_features \
  --features "$root/flow_localization/features.pt" \
  --out "$root/flow_localization"

bash artifacts/run_paired_trajectory_action_rerun.sh
