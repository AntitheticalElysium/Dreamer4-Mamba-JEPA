source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: check_damage on the 50k extensions (and their 42k / 48k snapshots): is the hit another stage-like mode (cf. placements)?
W=artifacts/eda/levers_tworlds_v1
job check_damage_50k 1300 $PY $L/check_damage.py $W/corrt_raw_teacher_s7_u50000_at42000.pt $W/corrt_raw_teacher_s7_u50000_at48000.pt \
  $W/corrt_raw_teacher_s7_u50000.pt $W/corrt_raw_teacher_s8_u50000_at42000.pt $W/corrt_raw_teacher_s8_u50000_at48000.pt $W/corrt_raw_teacher_s8_u50000.pt
echo "$(date '+%F %T') LANE49_DONE" >> $LOGDIR/lanes.log
