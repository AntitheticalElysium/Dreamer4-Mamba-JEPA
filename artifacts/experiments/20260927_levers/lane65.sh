source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: how much does Craftax reward memory? (check_memory, data only, CPU; readings in its docstring)
job check_memory 0 $PY $L/check_memory.py
echo "$(date '+%F %T') LANE65_DONE" >> $LOGDIR/lanes.log
