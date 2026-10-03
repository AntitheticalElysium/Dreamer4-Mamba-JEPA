source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: E17 stage 2 resource smoke (not a result): 500 updates each, L = 16 windows from the 36k worlds (seed 7), attention
# and Mamba, 40 windows per update (predeclared); on OOM the lane logs it and tries 24, then 13 (the only change the smoke may make).
# Each attempt waits for >= 3.5 GB free (admit), so an OOM reflects the configuration, not another job.
W=artifacts/eda/levers_tworlds_v1
for bb in full fmamba; do
  init=$W/corrt_raw_teacher_s7_u36000.pt; [ $bb = fmamba ] && init=$W/corrt_raw_teacher_s7_fmamba_u36000.pt
  for wpu in 40 24 13; do
    admit 3300                                  # >= 3.5 GB free: what a run has beside one co-running 2.2 GB job
    $PY $L/tworld.py --head corrt --loss teacher --seed 7 --backbone $bb --pool rawlong --frames 16 --windows $wpu --updates 500 \
      --init $init >> $LOGDIR/e17smoke_${bb}_w${wpu}.log 2>&1 && { echo "$(date '+%F %T') DONE e17smoke_${bb}_w${wpu}" >> $LOGDIR/lanes.log; break; }
    echo "$(date '+%F %T') FAILED e17smoke_${bb}_w${wpu}" >> $LOGDIR/lanes.log
  done
done
echo "$(date '+%F %T') LANE62_DONE" >> $LOGDIR/lanes.log
