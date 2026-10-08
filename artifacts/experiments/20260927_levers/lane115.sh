source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 E21 row-9 readout (window 10: the same last frames at output row 9, away from the training sampler's row-14 death cue),
# for M16 s7 (done inline 19:05) and each E21 arm as it finishes (G done inline), then D.
W=artifacts/eda/levers_tworlds_v1
TAIL=_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000
for TAG in c0 gl_ev ev; do
  case $TAG in c0) N=corrt_rawlong_teacher_s7_fmamba$TAIL ;; *) N=corrt_rawlong_teacher_s7_fmamba_$TAG$TAIL ;; esac
  until grep -q "E21V2_ARM_DONE $TAG" $LOGDIR/lanes.log; do sleep 30; done
  job e21v2_w10_$TAG 1100 $PY $L/health_chain.py world $W/$N.pt --window 10
done
until [ -f $W/corrt_rawlong_teacher_s7_fmamba_di$TAIL.pt ]; do sleep 60; done
sleep 60
job e21v2_w10_di 1100 $PY $L/health_chain.py world $W/corrt_rawlong_teacher_s7_fmamba_di$TAIL.pt --window 10
echo "$(date '+%F %T') LANE115_DONE" >> $LOGDIR/lanes.log
