source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: check_h16_traj on M6 s7 / s8 (window 5): the input to E17 stage 1's declared m6_h16_traj reading (lane52.sh) and
# E17 stage 2's parent comparator; lane50 ran it on the attention worlds only.
W=artifacts/eda/levers_tworlds_v1
job h16traj_m6 2400 $PY $L/check_h16_traj.py $W/corrt_raw_teacher_s7_fmamba_u36000.pt $W/corrt_raw_teacher_s8_fmamba_u36000.pt
echo "$(date '+%F %T') LANE76_DONE" >> $LOGDIR/lanes.log
