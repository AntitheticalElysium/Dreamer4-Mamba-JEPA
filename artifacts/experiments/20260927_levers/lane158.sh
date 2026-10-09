source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 14:48: E22M sufficiency (rule 12:57): prior 20k on the frozen E22M posterior, then e22_imagine (readings 1-2, in-distribution,
# census), 4 samples.
job e22m_prior 1500 $PY $L/dworld.py --stage b --seed 7 --init artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000.pt --updates 20000
job e22m_imagine 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000.pt --samples 4
echo "$(date '+%F %T') LANE158_DONE" >> $LOGDIR/lanes.log
