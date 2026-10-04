source artifacts/experiments/20260927_levers/lib.sh
# E6c: State Passing / TBTT post-training of the (a) world (Buitrago Ruiz & Gu 2025), with a post-training control
job statepass 2600 $PY $L/statepass.py
echo "$(date '+%F %T') LANE11_DONE" >> $LOGDIR/lanes.log
