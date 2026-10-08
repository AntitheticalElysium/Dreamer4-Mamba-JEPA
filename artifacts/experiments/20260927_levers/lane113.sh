source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 health diagnosis: approach-adjacency control on the full E20 pool (health_controls.py pool), alongside training.
job hcontrols_pool 1500 $PY $L/health_controls.py pool
echo "$(date '+%F %T') LANE113_DONE" >> $LOGDIR/lanes.log
