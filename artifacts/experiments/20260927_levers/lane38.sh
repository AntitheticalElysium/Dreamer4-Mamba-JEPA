source artifacts/experiments/20260927_levers/lib.sh
# E14a addendum 5 (2026-10-02; readings in headfit.py's docstring, committed before this lane ran): positives-only dose
W=artifacts/eda/levers_tworlds_v1
for w in corrt_raw_teacher_s7_u18000 corrt_raw_teacher_s8_u18000; do
  [ -f $L/evals/headfit_${w}_pos.json ] || job headfit_pos_$w 1500 $PY $L/headfit.py --arms skip_pos0.17,skip_pos1 --tag pos $W/$w.pt
done
echo "$(date '+%F %T') LANE38_DONE" >> $LOGDIR/lanes.log
