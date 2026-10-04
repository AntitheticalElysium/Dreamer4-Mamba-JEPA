source artifacts/experiments/20260927_levers/lib.sh
# E2 diagnosis: where the categorical world loses (codes: stay / contextual flip / content), snap floors
[ -f $L/catdiag.json ] || JAX_PLATFORMS=cpu job catdiag 2000 $PY $L/catdiag.py
echo "$(date '+%F %T') LANE7_DONE" >> $LOGDIR/lanes.log
