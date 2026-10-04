source artifacts/experiments/20260927_levers/lib.sh
# E17 stage 2 (predeclared in NOTEBOOK 8172e291 + amendments 1-2, 2026-10-04): A16 / M16 continued at L = 16 from their own 36k
# worlds on the 64-frame Raw TRAIN ledger, 40 windows per update (the resource smoke kept the predeclared count: attention 0.468
# s/update, peak 4.20 GB; Mamba 1.436 s/update, 2.87 GB), 6,000 updates, seeds 7 and 8. Attention first (it cannot share the
# GPU); Mamba then runs beside E18 (lane74). Readings: long_hits, long_fresh, mamba_long_edge (8172e291); long_recall_same,
# long_recall_used, moved_unsolved (check_recall docstring, 3ac7e494).
W=artifacts/eda/levers_tworlds_v1
for bb in full fmamba; do
  for s in 7 8; do
    P=corrt_raw_teacher_s${s}_u36000; [ $bb = fmamba ] && P=corrt_raw_teacher_s${s}_fmamba_u36000
    N=corrt_rawlong_teacher_s${s}$( [ $bb = fmamba ] && echo _fmamba )_L16b40_from36000
    tag=A16; [ $bb = fmamba ] && tag=M16; need=4400; [ $bb = fmamba ] && need=3200
    [ -f $W/$N.pt ] || job e17_${tag}_s$s $need $PY $L/tworld.py --head corrt --loss teacher --seed $s --backbone $bb --pool rawlong \
      --frames 16 --windows 40 --updates 6000 --init $W/$P.pt $( [ -f $W/state/$N.state.pt ] && echo --resume $W/state/$N.state.pt )
  done
done
L16="" ; for s in 7 8; do L16="$L16 $W/corrt_rawlong_teacher_s${s}_L16b40_from36000.pt $W/corrt_rawlong_teacher_s${s}_fmamba_L16b40_from36000.pt"; done
job e17_recall_w15 3000 $PY $L/check_recall.py --futures $L16
job e17_recall_w5 3000 $PY $L/check_recall.py --futures --window 5 $L16
job e17_recall_imagined_w15 3000 $PY $L/check_recall.py --futures --imagined $L16
job e17_damage_w16 3000 $PY $L/check_damage.py $L16 --window 16
job e17_damage_w5 3000 $PY $L/check_damage.py $L16
job e17_h16traj_w16 3000 $PY $L/check_h16_traj.py $L16 --window 16
for s in 7 8; do
  for N in corrt_rawlong_teacher_s${s}_L16b40_from36000 corrt_rawlong_teacher_s${s}_fmamba_L16b40_from36000; do
    [ -f $L/evals/${N}__w16_per_root.pt ] || job e17_teval_w16_$N 3000 $PY $L/teval.py $W/$N.pt --window 16
    [ -f $L/evals/${N}_per_root.pt ] || job e17_teval_w5_$N 3000 $PY $L/teval.py $W/$N.pt
  done
  A6=corrt_raw_teacher_s${s}_u36000; M6=corrt_raw_teacher_s${s}_fmamba_u36000
  A16=corrt_rawlong_teacher_s${s}_L16b40_from36000; M16=corrt_rawlong_teacher_s${s}_fmamba_L16b40_from36000
  job e17_compare_s$s 0 $PY $L/compare.py $A6:${A16}__w16 $M6:${M16}__w16 ${A16}__w16:${M16}__w16 $A16:${A16}__w16 $M16:${M16}__w16
done
echo "$(date '+%F %T') LANE73_DONE" >> $LOGDIR/lanes.log
