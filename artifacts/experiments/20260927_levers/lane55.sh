source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: matched-budget deterministic comparators for E16 (amendment in NOTEBOOK): the 100k continuation's 54k snapshots.
W=artifacts/eda/levers_tworlds_v1
while [ ! -f $W/corrt_raw_teacher_s8_u100000_at54000.pt ]; do sleep 300; done
[ -f $L/evals/h16traj/corrt_raw_teacher_s7_u100000_at54000.json ] || job h16traj_54k 2400 $PY $L/check_h16_traj.py \
  $W/corrt_raw_teacher_s7_u100000_at54000.pt $W/corrt_raw_teacher_s8_u100000_at54000.pt
job suite_damage_54k 1300 $PY $L/check_damage.py $W/corrt_raw_teacher_s7_u100000_at54000.pt $W/corrt_raw_teacher_s8_u100000_at54000.pt
echo "$(date '+%F %T') LANE55_DONE" >> $LOGDIR/lanes.log
