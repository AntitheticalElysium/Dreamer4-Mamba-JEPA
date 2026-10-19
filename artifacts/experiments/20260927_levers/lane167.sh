source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 19:29: E22MS (predeclared 19:29): the label-free channel recipe on the from-scratch D decoder.
job e22ms_frozen 2200 $PY $L/dworld.py --stage a --long --recon deltairis --revival always --seed 7 --init artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt --updates 2000 --windows 16 --freeze-world --tag dscratch
job e22ms_joint 2600 $PY $L/dworld.py --stage a --long --recon deltairis --revival always --seed 7 --init artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000_revalways_frozen_L16_di_b16_dscratch_u2000.pt --updates 2000 --windows 16 --motion-weight auto --tag dscratch
job e22ms_p2 1500 $PY $L/health_zombie.py E22MS=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_dscratch_u2000.pt
job e22ms_prior 1500 $PY $L/dworld.py --stage b --seed 7 --init artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_dscratch_u2000.pt --updates 20000
job e22ms_imagine_soft 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_dscratch_u2000_prior_u20000.pt --samples 4
job e22ms_imagine_hard 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_dscratch_u2000_prior_u20000.pt --samples 4 --decode interior
echo "$(date '+%F %T') LANE167_DONE" >> $LOGDIR/lanes.log
