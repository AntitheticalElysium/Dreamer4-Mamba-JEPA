source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 audit: lane73 never reached its predeclared teval (windows 15 and 5) and compare steps: the 10-06 20:55 reboot
# stopped it after the H16 job, which was later resumed by a separate script; the teval/compare tail was never requeued.
# These are exactly lane73's remaining commands (teval_export reuses M16 s7's completed, hash-verified reports).
W=artifacts/eda/levers_tworlds_v1
for s in 7 8; do
  for N in corrt_rawlong_teacher_s${s}_L16b40_from36000 corrt_rawlong_teacher_s${s}_fmamba_L16b40_from36000; do
    [ -f $L/evals/${N}__w15_per_root.pt ] || job e17_teval_w15_$N 3000 $PY artifacts/experiments/20261005_recovery/teval_export.py $W/$N.pt --window 15
    [ -f $L/evals/${N}_per_root.pt ] || job e17_teval_w5_$N 3000 $PY artifacts/experiments/20261005_recovery/teval_export.py $W/$N.pt
  done
  A6=corrt_raw_teacher_s${s}_u36000; M6=corrt_raw_teacher_s${s}_fmamba_u36000
  A16=corrt_rawlong_teacher_s${s}_L16b40_from36000; M16=corrt_rawlong_teacher_s${s}_fmamba_L16b40_from36000
  job e17_compare_s$s 0 $PY $L/compare.py $A6:${A16}__w15 $M6:${M16}__w15 ${A16}__w15:${M16}__w15 $A16:${A16}__w15 $M16:${M16}__w15
done
echo "$(date '+%F %T') LANE95_DONE" >> $LOGDIR/lanes.log
