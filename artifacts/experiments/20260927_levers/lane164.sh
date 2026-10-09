source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 17:43: E22H (predeclared 17:43): integer cell decisions in sampled imagination, decoding only (itc / interior), E22Z and E22M.
job e22h_z_itc 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000_prior_u20000.pt --samples 4 --decode itc
job e22h_m_itc 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000.pt --samples 4 --decode itc
job e22h_z_interior 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000_prior_u20000.pt --samples 4 --decode interior
job e22h_m_interior 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000.pt --samples 4 --decode interior
echo "$(date '+%F %T') LANE164_DONE" >> $LOGDIR/lanes.log
