source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-10 10:28: 10-10 PROGRAM C3 (imagination sensitivity): root-frame perturbations sigma 0.001 / 0.01 / 0.1, M16 / D cont / D scratch.
V=artifacts/eda/levers_tworlds_v1
job chaos_L16 2300 $PY $L/chaos.py M16=$V/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt DC=$V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt DS=$V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt
echo "$(date '+%F %T') LANE177_DONE" >> $LOGDIR/lanes.log
