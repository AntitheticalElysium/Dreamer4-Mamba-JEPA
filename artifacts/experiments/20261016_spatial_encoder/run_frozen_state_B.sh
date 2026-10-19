#!/usr/bin/env bash
# FROZEN_STATE_CONTROL_PREREG.md stage B (triggered by the pre-declared rule: frozen projected states also erase): fresh predictors on
# the frozen full-arm encoders' BACKBONE-LN tokens, seeds 1 then 2 (same init / data stream / schedule as the Raw-predictor and FD).
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)/spatial_encoder_v1"
for s in 1 2; do
  [ -f "$OUT/frozen_fb_s$s/step-010000.pt" ] || $PY raw_predictor.py --out "$OUT/frozen_fb_s$s" --seed $((20260730 + s)) \
    --encoder "spatial:$OUT/stage2_full_sig_s$s/step-010000.pt" --stage backbone_ln
done
echo frozen_B_done
