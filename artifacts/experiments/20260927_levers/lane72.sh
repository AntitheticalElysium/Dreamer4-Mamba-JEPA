source artifacts/experiments/20260927_levers/lib.sh
# E18 (2026-10-04, predeclared in NOTEBOOK "2026-10-04"): world-aligned Mamba. fcanvas (fmamba's Mamba-2 time scan over
# world-aligned canvas cells, scroll.estimate offsets; MapNet-style allocentric memory) on the A6 / M6 recipe: corrt head,
# teacher loss, 6-frame windows, 36k updates, seeds 7 and 8, snapshots every 6k; resumable. Waits for the E17 resource smoke
# (lane62) so the smoke's 3.5 GB admission is not starved.
# Readings (judged with check_recall's pool split against the same-seed fmamba 36k world; two-seed rule):
#   c6_moved_recall   fcanvas moved_slot capture >= fmamba's + 0.15 at both seeds
#   c6_same_recall    fcanvas same_slot capture >= fmamba's - 0.05 at both seeds (alignment does not cost the per-slot edge)
#   c6_unseen         fcanvas unseen entering-cell error within 5% of fmamba's at both seeds
W=artifacts/eda/levers_tworlds_v1
until grep -q "LANE62_DONE" $LOGDIR/lanes.log; do sleep 60; done
for s in 7 8; do
  N=corrt_raw_teacher_s${s}_fcanvas_u36000
  [ -f $W/$N.pt ] || job tworld_c6_s$s 2300 $PY $L/tworld.py --head corrt --loss teacher --seed $s --backbone fcanvas --updates 36000 --snapshots \
    $( [ -f $W/state/$N.state.pt ] && echo --resume $W/state/$N.state.pt )
done
job check_recall_c6 2000 $PY $L/check_recall.py $W/corrt_raw_teacher_s7_fcanvas_u36000.pt $W/corrt_raw_teacher_s7_fmamba_u36000.pt \
  $W/corrt_raw_teacher_s8_fcanvas_u36000.pt $W/corrt_raw_teacher_s8_fmamba_u36000.pt
echo "$(date '+%F %T') LANE72_DONE" >> $LOGDIR/lanes.log
