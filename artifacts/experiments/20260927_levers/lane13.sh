source artifacts/experiments/20260927_levers/lib.sh
# E4: evaluate the finished TC + LDAD lambda 10 run now (lane 4b re-runs both later for raw lambda 1; both skip done runs)
JAX_PLATFORMS=cpu job ldad_eval_tc10 1800 $PY $L/ldad_eval.py
JAX_PLATFORMS=cpu job ldad_facts_tc10 1800 $PY $L/ldad_facts.py
echo "$(date '+%F %T') LANE13_DONE" >> $LOGDIR/lanes.log
