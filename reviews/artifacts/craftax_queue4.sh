#!/usr/bin/env bash
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
LOGDIR=outputs/d4_mamba_jepa/queue
echo "=== [$(date -u +%H:%M:%S)] START h2_jepaweight0_FIXED ===" | tee -a "$LOGDIR/queue.log"
if .venv/bin/python -u reviews/artifacts/craftax_timecourse.py --jepa-weight 0.0 \
     --output reviews/artifacts/craftax_timecourse_jepaweight0.json \
     > "$LOGDIR/h2_jepaweight0_FIXED.log" 2>&1; then
  echo "=== [$(date -u +%H:%M:%S)] OK    h2_jepaweight0_FIXED ===" | tee -a "$LOGDIR/queue.log"
else
  echo "=== [$(date -u +%H:%M:%S)] FAIL  h2_jepaweight0_FIXED ===" | tee -a "$LOGDIR/queue.log"
fi
