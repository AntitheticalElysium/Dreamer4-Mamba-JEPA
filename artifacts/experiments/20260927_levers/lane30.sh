source artifacts/experiments/20260927_levers/lib.sh
# E14a (2026-10-02): frozen-backbone head re-training (cRT) + gradient anatomy, before any end-to-end E14 arm (headfit.py)
W=artifacts/eda/levers_tworlds_v1
for w in corrt_raw_teacher_s7_u18000 corrt_raw_teacher_s8_u18000 categorical_raw_teacher_s7_K4096 direct_raw_suffix_s7; do
  [ -f $L/evals/headfit_$w.json ] || job headfit_$w 2600 $PY $L/headfit.py $W/$w.pt
done
echo "$(date '+%F %T') LANE30_DONE" >> $LOGDIR/lanes.log
