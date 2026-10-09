source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 12:14: E22Z (predeclared 12:13; diagnostic, label-derived): the E22FJ smoke + zombie-cell L1 term (lambda auto); then P2,
# the encoder probe (CPU) and carries.
job e22z_smoke 2600 $PY $L/dworld.py --stage a --long --recon deltairis --revival always --seed 7 --init artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_frozen_L16_di_b16_u2000.pt --updates 2000 --windows 16 --zombie-weight auto
job e22z_p2 1500 $PY $L/health_zombie.py E22Z=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000.pt
CUDA_VISIBLE_DEVICES= job e22z_probe 0 $PY $L/e22_zombiecode.py E22Z=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000.pt E22FJ=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_L16_di_b16_u2000.pt --out e22_zombiecode_z.json
job e22z_carries 1500 $PY $L/e22_channel.py E22Z=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000.pt --out e22_carries_z.json
echo "$(date '+%F %T') LANE152_DONE" >> $LOGDIR/lanes.log
