source artifacts/experiments/20260927_levers/lib.sh
# E6d: does State Passing make the state distribution stationary (Buitrago Ruiz & Gu's mechanism)?
[ -f $L/statenorm.json ] || job statenorm 1800 $PY $L/statenorm.py
echo "$(date '+%F %T') LANE14_DONE" >> $LOGDIR/lanes.log
