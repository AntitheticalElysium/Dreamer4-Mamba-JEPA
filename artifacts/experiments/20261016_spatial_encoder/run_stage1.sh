#!/usr/bin/env bash
# Stage 1 screening (SPATIAL_ENCODER_PREREG.md): three arms, 2,000 updates each, sequential (one 6 GB GPU)
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)/spatial_encoder_v1"
for arm in naive mop_sig mop_vic; do
  if [ ! -f "$OUT/${arm}_s1/step-002000.pt" ]; then
    $PY spatial_encoder.py "$arm" --out "$OUT/${arm}_s1" --stop-at 2000
  fi
done
