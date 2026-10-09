source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-10 09:16: 10-10 PROGRAM Part I: exports of the deterministic worlds (M16, D continued, D from scratch), then H16 decisions
# for D from scratch and D continued (window 15; M16's exists).
V=artifacts/eda/levers_tworlds_v1
job sb_export_det 2300 $PY $L/rollout_export.py --world M16=$V/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt \
    --world DC=$V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt \
    --world DS=$V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt
job h16_dscratch 3000 $PY -B $L/check_h16_traj.py $V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt --window 15
job h16_dcont 3000 $PY -B $L/check_h16_traj.py $V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt --window 15
echo "$(date '+%F %T') LANE170_DONE" >> $LOGDIR/lanes.log
