source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 E21 (NOTEBOOK "E21", predeclared 10:40): 2 x 2 on M16 s7, each arm 6,000 updates continued from M16 with the M16
# recipe (L16, b40, rawlong, teacher, seed 7), snapshots every 2k, full state every 1k; then the arm's extended hidden states
# (health_chain.py --ext) and teval at window 15. Order: GE, C0, E, G (the main arm and its control first).
W=artifacts/eda/levers_tworlds_v1
M16=$W/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
TAIL=_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000
for ARM in "gl_ev:--gen-loss --event" "c0:" "ev:--event" "gl:--gen-loss"; do
  TAG=${ARM%%:*}; FLAGS=${ARM#*:}
  case $TAG in c0) N=corrt_rawlong_teacher_s7_fmamba$TAIL ;; *) N=corrt_rawlong_teacher_s7_fmamba_$TAG$TAIL ;; esac
  job e21_train_$TAG 3000 $PY $L/tworld.py --head corrt --pool rawlong --loss teacher --seed 7 --updates 6000 --backbone fmamba \
      --frames 16 --windows 40 --init $M16 --snapshots --snapshot-every 2000 --state-every 1000 $FLAGS
  job e21_hext_$TAG 3000 $PY $L/health_chain.py world $W/$N.pt --ext
  [ -f $L/evals/${N}__w15_per_root.pt ] || job e21_teval_$TAG 3000 $PY artifacts/experiments/20261005_recovery/teval_export.py $W/$N.pt --window 15
  echo "$(date '+%F %T') E21_ARM_DONE $TAG" >> $LOGDIR/lanes.log
done
echo "$(date '+%F %T') LANE104_DONE" >> $LOGDIR/lanes.log
