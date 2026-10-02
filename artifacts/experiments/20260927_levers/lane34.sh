source artifacts/experiments/20260927_levers/lib.sh
# E14a addendum 3 (2026-10-02; readings in headfit.py's docstring, committed before this lane ran): the dose from the model's
# own error (hard-token arms), after check_toperr placed the consequences in the per-token error tail
W=artifacts/eda/levers_tworlds_v1
for w in corrt_raw_teacher_s7_u18000 corrt_raw_teacher_s8_u18000 direct_raw_suffix_s7; do
  [ -f $L/evals/headfit_${w}_hard.json ] || job headfit_hard_$w 1500 $PY $L/headfit.py \
    --arms hard1x1,hard1x3,hard0.3x1,mlp_hard1x3,skip_hard1x3 --tag hard $W/$w.pt
done
echo "$(date '+%F %T') LANE34_DONE" >> $LOGDIR/lanes.log
