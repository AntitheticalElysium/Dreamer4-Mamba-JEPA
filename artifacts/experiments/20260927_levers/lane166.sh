source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 19:02: E22MP (predeclared 19:02): E22M's prior continued 20k -> 60k (its saved state, same run); held fit 20k vs 60k; sampled
# imagination with hard interior decoding on the 60k prior.
job e22mp_held20 1200 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000.pt --prior-held-only
[ -f artifacts/eda/levers_tworlds_v1/state/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u60000.state.pt ] || cp artifacts/eda/levers_tworlds_v1/state/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000.state.pt artifacts/eda/levers_tworlds_v1/state/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u60000.state.pt
job e22mp_prior60 1500 $PY $L/dworld.py --stage b --seed 7 --init artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000.pt --updates 60000
job e22mp_held60 1200 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u60000.pt --prior-held-only
job e22mp_imagine 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u60000.pt --samples 4 --decode interior
echo "$(date '+%F %T') LANE166_DONE" >> $LOGDIR/lanes.log
