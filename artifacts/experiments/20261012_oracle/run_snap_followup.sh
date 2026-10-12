#!/usr/bin/env bash
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP_DIR=artifacts/experiments/20261012_oracle
SNAP_OUT=artifacts/eda/oracle_v1
SNAP_LOG=artifacts/eda/levers_logs/oracle_snap_followup_20261012.log
{
    date '+START %Y-%m-%d %H:%M:%S %Z'
    .venv/bin/python "$SNAP_DIR/snap_eval.py" TRUE=TRUE \
      "oracle_deltairis=$SNAP_OUT/worlds/oracle_deltairis_from_corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000_u3000_b16.pt" \
      "control_deltairis=$SNAP_OUT/worlds/control_deltairis_from_corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000_u3000_b16.pt"
    .venv/bin/python "$SNAP_DIR/snap_analysis.py" \
      "$SNAP_OUT/stats_TRUE.json" "$SNAP_OUT/stats_oracle_deltairis.json" "$SNAP_OUT/stats_control_deltairis.json" \
      --out "$SNAP_OUT/snap_analysis.json"
    date '+SUCCESS %Y-%m-%d %H:%M:%S %Z'
} >> "$SNAP_LOG" 2>&1
