source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 21:55: LP step of E' = LP-FT (event_preflight.py lp), on its own so lane121's watchers are never blocked by it.
job e21_lp 1500 $PY $L/event_preflight.py lp
echo "$(date '+%F %T') LANE120_DONE" >> $LOGDIR/lanes.log
