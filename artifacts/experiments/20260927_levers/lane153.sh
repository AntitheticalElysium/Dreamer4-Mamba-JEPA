source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 12:49: E22Z sufficiency test (predeclared 12:49): stage-B prior on the frozen E22Z posterior, then E22 readings 1-2 with
# prior-sampled Delta (e22_imagine, 4 samples).
until grep -q "LANE152_DONE" $LOGDIR/lanes.log; do sleep 30; done
job e22z_prior 1500 $PY $L/dworld.py --stage b --seed 7 --init artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000.pt --updates 20000
job e22z_imagine 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000_prior_u20000.pt --samples 4
echo "$(date '+%F %T') LANE153_DONE" >> $LOGDIR/lanes.log
