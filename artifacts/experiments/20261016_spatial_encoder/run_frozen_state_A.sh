#!/usr/bin/env bash
# FROZEN_STATE_CONTROL_PREREG.md stage A: fresh predictors on the frozen full-arm encoders' DEPLOYED tokens, seeds 1 then 2
# (GPU strictly sequential; same init / data stream / schedule as the Raw-predictor of the same seed). Training only; the
# registered evaluations run afterwards.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)/spatial_encoder_v1"
for s in 1 2; do
  [ -f "$OUT/frozen_fd_s$s/step-010000.pt" ] || $PY raw_predictor.py --out "$OUT/frozen_fd_s$s" --seed $((20260730 + s)) \
    --encoder "spatial:$OUT/stage2_full_sig_s$s/step-010000.pt" --stage deployed
done
echo frozen_A_done
