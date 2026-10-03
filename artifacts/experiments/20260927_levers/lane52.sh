source artifacts/experiments/20260927_levers/lib.sh
# E17 stage 1 (2026-10-03, the user's request: Mamba at an equal training budget): the canonical-Mamba-2 factorized backbone
# (--backbone fmamba: per-frame spatial attention + a Mamba-2 time scan per token, tworld docstring) with the corrt head and the
# teacher loss on the same 6-frame windows, batch, optimizer and 36k updates as the attention worlds corrt_raw_teacher_s{7,8}_u36000
# (A6). Measured cost 0.72 s/update (transport_resource_steady.json, B40), ~7.2 h per seed. Snapshots every 6k; resumable.
# Readings, declared before running (A6 = the same-seed attention world; two-seed rule):
#   m6_depth16       compare.py gen_16 (M6 - A6) resolved lower at both seeds
#   m6_onestep       compare.py onestep_all (M6 - A6) resolved lower at both seeds
#   m6_hits          check_damage teacher-forced caught M6 - A6 >= +0.05 at both seeds
#   m6_h16_traj      check_h16_traj trajectory value M6 - A6 >= +0.02 at both seeds (reported against its DEV-B interval scale)
#   m6_consequences  consfit held strict caught, per action, reported against A6's 36k mode set (DO only)
W=artifacts/eda/levers_tworlds_v1
for s in 7 8; do
  N=corrt_raw_teacher_s${s}_fmamba_u36000
  [ -f $W/$N.pt ] || job tworld_m6_s$s 2600 $PY $L/tworld.py --head corrt --loss teacher --seed $s --backbone fmamba --updates 36000 --snapshots \
    $( [ -f $W/state/$N.state.pt ] && echo --resume $W/state/$N.state.pt )
done
for s in 7 8; do
  N=corrt_raw_teacher_s${s}_fmamba_u36000; A=corrt_raw_teacher_s${s}_u36000
  job consfit_$N 2000 $PY $L/consfit.py $W/$N.pt
  job check_damage_$N 1300 $PY $L/check_damage.py $W/$N.pt
  [ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
  job compare_$N 0 $PY $L/compare.py $A:$N
done
echo "$(date '+%F %T') LANE52_DONE" >> $LOGDIR/lanes.log
