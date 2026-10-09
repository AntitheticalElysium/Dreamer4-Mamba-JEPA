source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 11:50: E22FJ P2 failed with the channel open: where its capacity goes (changed cells split by scroll / entering edges) and P2
# split by scroll, E22FJ vs D (E22F's zeroed prediction = D exactly).
job e22_carries2 1500 $PY $L/e22_channel.py E22FJ=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_L16_di_b16_u2000.pt E22F=artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_frozen_L16_di_b16_u2000.pt --out e22_carries2.json
job e22_p2scroll 1500 $PY $L/health_zombie.py E22FJs=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_L16_di_b16_u2000.pt Ds=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
echo "$(date '+%F %T') LANE150_DONE" >> $LOGDIR/lanes.log
