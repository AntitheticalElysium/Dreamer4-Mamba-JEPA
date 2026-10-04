source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: check_delta on the CPU (DELTA_DEVICE=cpu, Mamba-2 reference scan), same worlds and readings as lane79.
W=artifacts/eda/levers_tworlds_v1
DELTA_DEVICE=cpu job check_delta 0 $PY $L/check_delta.py $W/corrt_raw_teacher_s7_fmamba_u36000.pt $W/corrt_raw_teacher_s8_fmamba_u36000.pt
echo "$(date '+%F %T') LANE80_DONE" >> $LOGDIR/lanes.log
