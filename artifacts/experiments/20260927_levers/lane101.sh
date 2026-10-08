source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 health diagnosis 5: objective / backbone-gradient share of the health token on M16's own training batches.
W=artifacts/eda/levers_tworlds_v1
for N in corrt_rawlong_teacher_s7_fmamba_L16b40_from36000 corrt_rawlong_teacher_s8_fmamba_L16b40_from36000; do
  job hgrad_$N 2500 $PY $L/health_gradient.py $W/$N.pt 20
done
echo "$(date '+%F %T') LANE101_DONE" >> $LOGDIR/lanes.log
