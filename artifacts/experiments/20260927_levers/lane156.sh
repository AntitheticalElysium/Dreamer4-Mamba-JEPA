source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 14:03: E22Z sufficiency, in-distribution readings (beside rate and hit rate given beside, imagined vs true; caught / false
# split by imagined adjacency), after lane153's first run.
until grep -q "LANE153_DONE" $LOGDIR/lanes.log; do sleep 30; done
cp artifacts/eda/levers_logs/e22z_imagine.log artifacts/eda/levers_logs/e22z_imagine_v1.log
job e22z_imagine_v2 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000_prior_u20000.pt --samples 4
echo "$(date '+%F %T') LANE156_DONE" >> $LOGDIR/lanes.log
