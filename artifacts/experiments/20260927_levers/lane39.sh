source artifacts/experiments/20260927_levers/lib.sh
# E14d (2026-10-02): the dose end-to-end. E14c s7 (mask1 + skip, lambda 1): consequences 0.991 caught, but onestep +72%, gen_16
# +0.160, position failures not reduced; check_e14c_grad: the mask term's backbone gradient is 5.4x the uniform term's (cos 0.05)
# in the trained world -- the dose dominates. A backbone that adapts may need far less than the head-only x304.
# Arms: --weight mask0.1 --skip (dose x31), seeds 7 and 8, otherwise the u18000 teacher recipe. Readings (declared before training;
# comparator = the same-seed teacher u18000 world; two-seed rule), as E14c: d_learned (consfit held caught >= 0.5), d_cost
# (teval onestep_all ratio <= 1.10), d_position (subst16 ever_position_wrong >= 20% lower), d_depth16 (gen_16 not resolved worse),
# d_decision (dpanel gen1 or gen2 resolved better); also d_balanced = check_e14c_grad backbone ratio < 3.
W=artifacts/eda/levers_tworlds_v1
for s in 7 8; do
  N=corrt_raw_teacher_s${s}_mask0.1_skip_u18000
  T0=corrt_raw_teacher_s${s}_u18000
  [ -f $W/$N.pt ] || job tworld_$N 2400 $PY $L/tworld.py --head corrt --loss teacher --seed $s --updates 18000 --weight mask0.1 --skip
  [ -f $L/evals/consfit_$N.json ] || job consfit_$N 2000 $PY $L/consfit.py $W/$N.pt
  [ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
  job compare_$N 0 $PY $L/compare.py $T0:$N
  [ -f $L/evals/subst16_${N}_base.json ] || job subst16_$N 2000 $PY $L/subst16.py $W/$N.pt --arms base --tag base
  [ -f $L/evals/dpanel_$N.json ] || job dpanel_$N 3000 $PY $L/dpanel.py $W/$N.pt
  job dpanel_compare_$N 0 $PY $L/dpanel.py --compare $N:$T0
done
echo "$(date '+%F %T') LANE39_DONE" >> $LOGDIR/lanes.log
