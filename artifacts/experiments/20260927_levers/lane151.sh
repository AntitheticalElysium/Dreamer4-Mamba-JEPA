source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 12:06: does the posterior ENCODE the zombie's random move (no-scroll steps)? CPU probe on the four stage-A posteriors.
export CUDA_VISIBLE_DEVICES=
job e22_zombiecode 0 $PY $L/e22_zombiecode.py E22FJ=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_L16_di_b16_u2000.pt E22F=artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_frozen_L16_di_b16_u2000.pt SCRATCH=artifacts/eda/levers_tworlds_v1/dworld_a_s7_scratch_revalways_L16_di_b16_u2000.pt WARM=artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_L16_di_b16_u2000.pt
echo "$(date '+%F %T') LANE151_DONE" >> $LOGDIR/lanes.log
