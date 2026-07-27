#!/usr/bin/env bash
# Stage G: waits for queue 1 to exit, then runs the encoder time-course probe.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
LOGDIR=outputs/d4_mamba_jepa/queue
while pgrep -f "craftax_queue.sh" > /dev/null; do sleep 60; done
echo "=== [$(date -u +%H:%M:%S)] START g_timecourse ===" | tee -a "$LOGDIR/queue.log"
if .venv/bin/python -u reviews/artifacts/craftax_timecourse.py > "$LOGDIR/g_timecourse.log" 2>&1; then
  echo "=== [$(date -u +%H:%M:%S)] OK    g_timecourse ===" | tee -a "$LOGDIR/queue.log"
else
  echo "=== [$(date -u +%H:%M:%S)] FAIL  g_timecourse ===" | tee -a "$LOGDIR/queue.log"
fi
echo "=== [$(date -u +%H:%M:%S)] ALL QUEUES COMPLETE ===" | tee -a "$LOGDIR/queue.log"
