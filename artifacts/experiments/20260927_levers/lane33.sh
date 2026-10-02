source artifacts/experiments/20260927_levers/lib.sh
# E14a addendum 2 (2026-10-02; readings in headfit.py's docstring, committed before this lane ran): the head also reads the raw
# local neighbourhood (skip_ arms), after check_allprobe showed h dilutes the consequence conjunction the input carries
W=artifacts/eda/levers_tworlds_v1
for w in corrt_raw_teacher_s7_u18000 corrt_raw_teacher_s8_u18000 direct_raw_suffix_s7 categorical_raw_teacher_s7_K4096; do
  [ -f $L/evals/headfit_${w}_skip.json ] || job headfit_skip_$w 1500 $PY $L/headfit.py --arms skip_uniform,skip_mask1,skip_mask10 --tag skip $W/$w.pt
done
echo "$(date '+%F %T') LANE33_DONE" >> $LOGDIR/lanes.log
