source artifacts/experiments/20260927_levers/lib.sh
# E14a addendum 4 (2026-10-02; readings in headfit.py's docstring, committed before this lane ran): generic calm-frame event dose
W=artifacts/eda/levers_tworlds_v1
for w in corrt_raw_teacher_s7_u18000 corrt_raw_teacher_s8_u18000; do
  [ -f $L/evals/headfit_${w}_event.json ] || job headfit_event_$w 1500 $PY $L/headfit.py --arms event60x2,skip_event60x1,skip_event60x2 --tag event $W/$w.pt
done
echo "$(date '+%F %T') LANE37_DONE" >> $LOGDIR/lanes.log
