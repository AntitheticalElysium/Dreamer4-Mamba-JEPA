source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 health diagnosis 6 controls as committed code (adjacency positive control, layer-wise phase), reproducing the inline runs.
job hcontrols 1100 $PY $L/health_controls.py
echo "$(date '+%F %T') LANE107_DONE" >> $LOGDIR/lanes.log
