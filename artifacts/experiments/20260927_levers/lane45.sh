source artifacts/experiments/20260927_levers/lib.sh
# E14e (2026-10-02): the 36k budget worlds extended to 50k by warm restart (option A, validated in lane44: warm_ok TRUE).
# Tests the mode-by-mode prediction (Saxe et al. 2019: weaker modes learned later, each in a stage-like step) and whether more
# budget keeps improving the world. Readings, declared before running (comparator = the same-seed 36k world; two-seed rule):
#   placement_learned   consfit held strict caught >= 0.5 for some placement type (7 / 8 / 9) at 50k at BOTH seeds
#   budget_continues    teval onestep_all and gen_16 resolved lower than the 36k world's at both seeds
#   decision            dpanel gen1 / gen2 / transfer1 / transfer2 vs the 36k world, reported
W=artifacts/eda/levers_tworlds_v1
for s in 7 8; do
  [ -f $W/state/warm_s${s}_at36000.state.pt ] || job warmstate_s${s}_36k 0 $PY $L/warmstate.py $W/corrt_raw_teacher_s${s}_u36000.pt 36000 $W/state/warm_s${s}_at36000.state.pt
  [ -f $W/corrt_raw_teacher_s${s}_u50000.pt ] || job tworld_s${s}_36k_to_50k 2400 $PY $L/tworld.py --head corrt --loss teacher --seed $s --updates 50000 --snapshots --resume $W/state/warm_s${s}_at36000.state.pt
done
for s in 7 8; do
  N=corrt_raw_teacher_s${s}_u50000; T0=corrt_raw_teacher_s${s}_u36000
  job consfit_$N 2000 $PY $L/consfit.py $W/${N}_at42000.pt $W/${N}_at48000.pt $W/$N.pt
  [ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
  job compare_$N 0 $PY $L/compare.py $T0:$N
  [ -f $L/evals/subst16_${N}_base.json ] || job subst16_$N 2000 $PY $L/subst16.py $W/$N.pt --arms base --tag base
  [ -f $L/evals/dpanel_$N.json ] || job dpanel_$N 2800 $PY $L/dpanel.py $W/$N.pt
  job dpanel_compare_$N 0 $PY $L/dpanel.py --compare $N:$T0
done
echo "$(date '+%F %T') LANE45_DONE" >> $LOGDIR/lanes.log
