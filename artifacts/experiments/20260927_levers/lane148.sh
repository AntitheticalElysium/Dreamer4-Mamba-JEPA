source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 11:08: E22F P1 passed (+0.57%), P2 failed: what the channel carries (e22_channel 'carries'), E22F vs the warm and scratch smokes.
job e22_carries 1500 $PY $L/e22_channel.py E22F=artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_frozen_L16_di_b16_u2000.pt WARM=artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_L16_di_b16_u2000.pt SCRATCH=artifacts/eda/levers_tworlds_v1/dworld_a_s7_scratch_revalways_L16_di_b16_u2000.pt --out e22_carries.json
echo "$(date '+%F %T') LANE148_DONE" >> $LOGDIR/lanes.log
