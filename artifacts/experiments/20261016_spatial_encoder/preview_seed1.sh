#!/usr/bin/env bash
# Seed-1 preview on CPU while seed 2 trains on the GPU (Stage 2 numbers of record come from run_stage2.sh's GPU evaluation).
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)/spatial_encoder_v1"
until [ -f "$OUT/stage2_mop_sig_s1/step-010000.pt" ] && [ -f "$OUT/stage2_rawpred_s1/step-010000.pt" ]; do sleep 60; done
sleep 30
export EVAL_DEVICE=cpu OMP_NUM_THREADS=8
$PY spatial_eval.py spatial_s1=spatial:$OUT/stage2_mop_sig_s1/step-010000.pt
$PY tierb_eval.py \
  rawpred_s1=raw@10000,$OUT/stage2_rawpred_s1/step-010000.pt,$OUT/probes_raw10000.pt \
  spatial_s1=spatial:$OUT/stage2_mop_sig_s1/step-010000.pt,$OUT/stage2_mop_sig_s1/step-010000.pt,$OUT/probes_spatial_s1.pt
STAGE2_SEEDS=1 $PY stage2_analysis.py
echo preview_done
