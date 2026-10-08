source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 E21 trends: extended hidden states of each arm's 2k / 4k snapshots as they appear, alongside training
# (health_chain.py world --ext measured at 0.56 GB peak allocated, declared 1,100 MiB).
W=artifacts/eda/levers_tworlds_v1
TAIL=_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000
for TAG in gl_ev c0 ev gl; do
  case $TAG in c0) N=corrt_rawlong_teacher_s7_fmamba$TAIL ;; *) N=corrt_rawlong_teacher_s7_fmamba_$TAG$TAIL ;; esac
  for U in 2000 4000; do
    until [ -f $W/${N}_at$U.pt ]; do sleep 30; done
    sleep 30                                                      # let the snapshot write finish
    job e21_hext_${TAG}_at$U 1100 $PY $L/health_chain.py world $W/${N}_at$U.pt --ext
  done
done
echo "$(date '+%F %T') LANE105_DONE" >> $LOGDIR/lanes.log
