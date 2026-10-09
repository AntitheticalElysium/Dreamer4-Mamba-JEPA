source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 12:59: E22M (predeclared 12:57): the E22Z smoke with the label-free moved-content mask; then P2, encoder probe (CPU), carries.
job e22m_smoke 2600 $PY $L/dworld.py --stage a --long --recon deltairis --revival always --seed 7 --init artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_frozen_L16_di_b16_u2000.pt --updates 2000 --windows 16 --motion-weight auto
job e22m_p2 1500 $PY $L/health_zombie.py E22M=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000.pt
CUDA_VISIBLE_DEVICES= job e22m_probe 0 $PY $L/e22_zombiecode.py E22M=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000.pt E22Z=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000.pt E22FJ=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_L16_di_b16_u2000.pt --out e22_zombiecode_m.json
job e22m_carries 1500 $PY $L/e22_channel.py E22M=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000.pt --out e22_carries_m.json
echo "$(date '+%F %T') LANE155_DONE" >> $LOGDIR/lanes.log
