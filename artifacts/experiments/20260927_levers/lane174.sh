source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-10 10:23: 10-10 PROGRAM B4 (P-C): gradient anatomy by token category under BOTH objectives (l1, deltairis) for the matched
# stage-1 snapshots (M6 L1 / D-scratch D-loss, 6k..36k; all held windows) and the L16 worlds (M16, D continued, D scratch; 60 batches).
V=artifacts/eda/levers_tworlds_v1
job gradanat_stage1 2000 $PY $L/gradanat.py  $V/corrt_raw_teacher_s7_fmamba_u36000_at6000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000_at6000.pt $V/corrt_raw_teacher_s7_fmamba_u36000_at12000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000_at12000.pt $V/corrt_raw_teacher_s7_fmamba_u36000_at18000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000_at18000.pt $V/corrt_raw_teacher_s7_fmamba_u36000_at24000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000_at24000.pt $V/corrt_raw_teacher_s7_fmamba_u36000_at30000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000_at30000.pt $V/corrt_raw_teacher_s7_fmamba_u36000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000.pt
job gradanat_L16 2000 $PY $L/gradanat.py $V/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt $V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt $V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt --batches 60
echo "$(date '+%F %T') LANE174_DONE" >> $LOGDIR/lanes.log
