#!/usr/bin/env bash
# Stage 2 (SPATIAL_ENCODER_PREREG.md): the selected arm at 2 encoder seeds and the Raw-predictor at 2 predictor seeds, 10,000
# updates each, strictly one GPU job at a time (the 6 GB GPU cannot hold two; 10-16 OOM), then Tier A and Tier B for every world.
# Usage: run_stage2.sh <arm>
set -euo pipefail
cd "$(dirname "$0")"
ARM="$1"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)/spatial_encoder_v1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 1 2; do
  seed=$((20260730 + s))
  [ -f "$OUT/stage2_${ARM}_s$s/step-010000.pt" ] || $PY spatial_encoder.py "$ARM" --out "$OUT/stage2_${ARM}_s$s" --stop-at 10000 --seed $seed
  [ -f "$OUT/stage2_rawpred_s$s/step-010000.pt" ] || $PY raw_predictor.py --out "$OUT/stage2_rawpred_s$s" --stop-at 10000 --seed $seed
done
$PY spatial_eval.py raw10000=raw@10000 spatial_s1=spatial:$OUT/stage2_${ARM}_s1/step-010000.pt spatial_s2=spatial:$OUT/stage2_${ARM}_s2/step-010000.pt
$PY tierb_eval.py \
  rawpred_s1=raw@10000,$OUT/stage2_rawpred_s1/step-010000.pt,$OUT/probes_raw10000.pt \
  rawpred_s2=raw@10000,$OUT/stage2_rawpred_s2/step-010000.pt,$OUT/probes_raw10000.pt \
  spatial_s1=spatial:$OUT/stage2_${ARM}_s1/step-010000.pt,$OUT/stage2_${ARM}_s1/step-010000.pt,$OUT/probes_spatial_s1.pt \
  spatial_s2=spatial:$OUT/stage2_${ARM}_s2/step-010000.pt,$OUT/stage2_${ARM}_s2/step-010000.pt,$OUT/probes_spatial_s2.pt
echo stage2_done
