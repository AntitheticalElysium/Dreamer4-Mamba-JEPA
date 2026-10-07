source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 health diagnosis 3b: does the hit evidence in h grow with training? M6 s7 snapshots (6k..30k; 36k in lane97) and the
# attention 50k / 100k continuations (6-frame worlds, window 5), extended hidden states.
W=artifacts/eda/levers_tworlds_v1
until grep -q "LANE97_DONE" $LOGDIR/lanes.log; do sleep 30; done
for N in corrt_raw_teacher_s7_fmamba_u36000_at6000 corrt_raw_teacher_s7_fmamba_u36000_at12000 corrt_raw_teacher_s7_fmamba_u36000_at18000 \
         corrt_raw_teacher_s7_fmamba_u36000_at24000 corrt_raw_teacher_s7_fmamba_u36000_at30000 corrt_raw_teacher_s7_u50000 corrt_raw_teacher_s7_u100000; do
  job hext_$N 3000 $PY $L/health_chain.py world $W/$N.pt --ext
done
echo "$(date '+%F %T') LANE98_DONE" >> $LOGDIR/lanes.log
