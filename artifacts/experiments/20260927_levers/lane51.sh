source artifacts/experiments/20260927_levers/lib.sh
# E14f (2026-10-03, the user's request): the 50k teacher worlds CONTINUED to 100k from their full states (AdamW included; the
# learning rate is constant after warmup, d4mj.train._phase_lr), snapshots every 6k. Does budget keep learning modes, and does a
# hit mode appear? Readings, declared before running (comparator = the same run at 50k; two-seed rule):
#   table_learned     consfit held strict caught for place table (action 8) >= 0.5 at 100k at both seeds
#   hit_mode          check_damage teacher-forced caught >= 0.2 at 100k at both seeds (check_damage_rule ceiling for a 4-frame
#                     input: 47% of hits are drawable by any deterministic world; current worlds 2-10%)
#   fresh_hits        fresh-arrival hits (P 0.97 from the window) caught >= 0.5 at 100k at both seeds
#   position_plateau  subst16 ever_position_wrong at 100k > the 50k value - 0.02 at both seeds (s7 50k: 0.283)
#   budget_continues  compare onestep_all and gen_16 resolved lower than 50k at both seeds
W=artifacts/eda/levers_tworlds_v1
for s in 7 8; do
  N=corrt_raw_teacher_s${s}_u100000
  [ -f $W/$N.pt ] || job tworld_s${s}_50k_to_100k 2400 $PY $L/tworld.py --head corrt --loss teacher --seed $s --updates 100000 --snapshots \
    --resume $( [ -f $W/state/$N.state.pt ] && echo $W/state/$N.state.pt || echo $W/state/corrt_raw_teacher_s${s}_u50000.state.pt )
done
for s in 7 8; do
  N=corrt_raw_teacher_s${s}_u100000; T0=corrt_raw_teacher_s${s}_u50000
  SN="$(for u in 54000 60000 66000 72000 78000 84000 90000 96000; do echo -n "$W/${N}_at$u.pt "; done)$W/$N.pt"
  job consfit_$N 2000 $PY $L/consfit.py $SN
  job check_damage_$N 1300 $PY $L/check_damage.py $SN
  [ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
  job compare_$N 0 $PY $L/compare.py $T0:$N
  [ -f $L/evals/subst16_${N}_base.json ] || job subst16_$N 2000 $PY $L/subst16.py $W/$N.pt --arms base --tag base
done
echo "$(date '+%F %T') LANE51_DONE" >> $LOGDIR/lanes.log
