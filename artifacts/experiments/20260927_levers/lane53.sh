source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: risk-suite baselines (check_damage with health-change accuracy and Bayes hit rates; reported) on the attention
# teacher worlds the Mamba (lane52) and 100k (lane51) arms are compared with.
W=artifacts/eda/levers_tworlds_v1
job suite_damage_attention 1300 $PY $L/check_damage.py $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s8_u36000.pt \
  $W/corrt_raw_teacher_s7_u50000.pt $W/corrt_raw_teacher_s8_u50000.pt
echo "$(date '+%F %T') LANE53_DONE" >> $LOGDIR/lanes.log
