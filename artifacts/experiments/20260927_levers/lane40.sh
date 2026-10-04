source artifacts/experiments/20260927_levers/lib.sh
# E14b' (2026-10-02 16:00): the 36k budget worlds through the E14c evaluations. consfit (E14b): s7 held caught 0.002 (18k) ->
# 0.432 (24k) -> 0.561 (30k) -> 0.563 (36k), hallucinated 0.007: the uniform loss learns the consequences LATE, abruptly
# (s8: between 12k and 18k). Readings, declared before any of these evaluations ran (comparator = the same-seed teacher u18000
# world; two-seed rule): b_cost (teval onestep_all ratio <= 1.10), b_position (subst16 ever_position_wrong >= 20% lower),
# b_depth16 (gen_16 not resolved worse), b_decision (dpanel gen1 or gen2 resolved better); monotone reported.
W=artifacts/eda/levers_tworlds_v1
for s in 7 8; do
  N=corrt_raw_teacher_s${s}_u36000
  T0=corrt_raw_teacher_s${s}_u18000
  until [ -f $W/$N.pt ]; do sleep 120; done
  [ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
  job compare_$N 0 $PY $L/compare.py $T0:$N
  [ -f $L/evals/subst16_${N}_base.json ] || job subst16_$N 2000 $PY $L/subst16.py $W/$N.pt --arms base --tag base
  [ -f $L/evals/dpanel_$N.json ] || job dpanel_$N 3000 $PY $L/dpanel.py $W/$N.pt
  job dpanel_compare_$N 0 $PY $L/dpanel.py --compare $N:$T0
  [ -f $L/evals/monotone_$N.json ] || job monotone_$N 1500 $PY $L/monotone.py $W/$N.pt
done
echo "$(date '+%F %T') LANE40_DONE" >> $LOGDIR/lanes.log
