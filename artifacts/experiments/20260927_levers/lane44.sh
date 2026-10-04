source artifacts/experiments/20260927_levers/lib.sh
# Option A validation (2026-10-02): can a weights-only world be continued (warm restart: fresh AdamW, batch order replayed)?
# s7's 36k run: warm-restart its 24k snapshot to 30k, against the true 30k snapshot. Readings, declared before running:
#   warm_ok = consfit held caught within 0.10 of the true 30k snapshot's (0.561) AND teval onestep_all within 5% of its value
#   (a criterion any rerun must meet: same-seed reruns diverge chaotically, check_determinism). If warm_ok, the 36k worlds can be
#   extended to 50k from their weights; otherwise reruns with the resumable trainer.
W=artifacts/eda/levers_tworlds_v1
until grep -q "LANE43_DONE" $LOGDIR/lanes.log; do sleep 60; done
[ -f $W/state/warm_s7_at24000.state.pt ] || job warmstate_s7_24k 0 $PY $L/warmstate.py $W/corrt_raw_teacher_s7_u36000_at24000.pt 24000 $W/state/warm_s7_at24000.state.pt
[ -f $W/corrt_raw_teacher_s7_u30000.pt ] || job tworld_warm_s7_24k_to_30k 2400 $PY $L/tworld.py --head corrt --loss teacher --seed 7 --updates 30000 --resume $W/state/warm_s7_at24000.state.pt
job consfit_warm 2000 $PY $L/consfit.py $W/corrt_raw_teacher_s7_u30000.pt
[ -f $L/evals/corrt_raw_teacher_s7_u36000_at30000_per_root.pt ] || job teval_true30k 2000 $PY $L/teval.py $W/corrt_raw_teacher_s7_u36000_at30000.pt
[ -f $L/evals/corrt_raw_teacher_s7_u30000_per_root.pt ] || job teval_warm30k 2000 $PY $L/teval.py $W/corrt_raw_teacher_s7_u30000.pt
job compare_warm 0 $PY $L/compare.py corrt_raw_teacher_s7_u36000_at30000:corrt_raw_teacher_s7_u30000
echo "$(date '+%F %T') LANE44_DONE" >> $LOGDIR/lanes.log
