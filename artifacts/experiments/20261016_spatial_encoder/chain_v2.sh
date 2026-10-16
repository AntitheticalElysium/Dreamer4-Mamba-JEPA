#!/usr/bin/env bash
# Corrected Stage 2 evaluation (Amendment 4/4b), on CPU (never shares the training GPU; all Tier B worlds use the same device):
# seed 1 as soon as both seed-1 checkpoints exist (while seed 2 trains), then seed 2 after the training runner ends.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)/spatial_encoder_v1"
export EVAL_DEVICE=cpu OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
for s in 1 2; do
  until [ -f "$OUT/stage2_mop_sig_s$s/step-010000.pt" ] && [ -f "$OUT/stage2_rawpred_s$s/step-010000.pt" ]; do sleep 60; done
  sleep 60
  $PY spatial_eval_v2.py spatial_s$s=spatial:$OUT/stage2_mop_sig_s$s/step-010000.pt
  $PY tierb_eval_v2.py \
    rawpred_s$s=raw@10000,$OUT/stage2_rawpred_s$s/step-010000.pt,$OUT/probes2_raw10000.pt \
    spatial_s$s=spatial:$OUT/stage2_mop_sig_s$s/step-010000.pt,$OUT/stage2_mop_sig_s$s/step-010000.pt,$OUT/probes2_spatial_s$s.pt
  if [ "$s" = 1 ]; then $PY stage2_analysis_v2.py --seeds 1 --tag _seed1; fi
done
$PY stage2_analysis_v2.py --seeds 1,2
echo chain_v2_done
