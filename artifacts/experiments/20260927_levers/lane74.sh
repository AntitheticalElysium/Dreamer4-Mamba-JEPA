source artifacts/experiments/20260927_levers/lib.sh
# E18 (predeclared in NOTEBOOK 2026-10-04 and lane72.sh, unchanged): fcanvas on the A6 / M6 recipe, 36k, seeds 7 and 8, then
# check_recall's pool split. Replaces lane72 (stopped 13:44 before it had saved a state): starts after E17's attention runs,
# so it runs beside the Mamba L16 runs (fcanvas 2.4 GB + M16 2.9 GB fit; attention L16 at 4.2 GB does not).
W=artifacts/eda/levers_tworlds_v1
until grep -q "DONE e17_A16_s8" $LOGDIR/lanes.log; do sleep 60; done
for s in 7 8; do
  N=corrt_raw_teacher_s${s}_fcanvas_u36000
  [ -f $W/$N.pt ] || job tworld_c6_s$s 2300 $PY $L/tworld.py --head corrt --loss teacher --seed $s --backbone fcanvas --updates 36000 --snapshots \
    $( [ -f $W/state/$N.state.pt ] && echo --resume $W/state/$N.state.pt )
done
job check_recall_c6 2000 $PY $L/check_recall.py $W/corrt_raw_teacher_s7_fcanvas_u36000.pt $W/corrt_raw_teacher_s7_fmamba_u36000.pt \
  $W/corrt_raw_teacher_s8_fcanvas_u36000.pt $W/corrt_raw_teacher_s8_fmamba_u36000.pt
job check_recall_c6_futures 2600 $PY $L/check_recall.py --futures $W/corrt_raw_teacher_s7_fcanvas_u36000.pt $W/corrt_raw_teacher_s8_fcanvas_u36000.pt
echo "$(date '+%F %T') LANE74_DONE" >> $LOGDIR/lanes.log
