source artifacts/experiments/20260927_levers/lib.sh
# E14c evaluations (readings in lane35.sh): every world as soon as it is saved
W=artifacts/eda/levers_tworlds_v1
for N in corrt_raw_teacher_s7_mask1_skip_u18000 corrt_raw_teacher_s8_mask1_skip_u18000 corrt_raw_teacher_s7_mask1_u18000 corrt_raw_teacher_s8_mask1_u18000; do
  T0=corrt_raw_teacher_s$(echo $N | sed 's/.*_s\([78]\)_.*/\1/')_u18000
  until [ -f $W/$N.pt ]; do sleep 60; done
  [ -f $L/evals/consfit_$N.json ] || job consfit_$N 2000 $PY $L/consfit.py $W/$N.pt
  [ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
  job compare_$N 0 $PY $L/compare.py $T0:$N
  [ -f $L/evals/subst16_${N}_base.json ] || job subst16_$N 2000 $PY $L/subst16.py $W/$N.pt --arms base --tag base
  [ -f $L/evals/dpanel_$N.json ] || job dpanel_$N 3000 $PY $L/dpanel.py $W/$N.pt
  job dpanel_compare_$N 0 $PY $L/dpanel.py --compare $N:$T0
  [ -f $L/evals/monotone_$N.json ] || job monotone_$N 1500 $PY $L/monotone.py $W/$N.pt
done
echo "$(date '+%F %T') LANE36_DONE" >> $LOGDIR/lanes.log
