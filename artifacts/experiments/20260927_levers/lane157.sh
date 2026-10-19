source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 14:16: E22Z sampled imagination over-produces adjacency (21-22% vs 8.7%): zombie census by depth (count of drawn zombies,
# nearest distance to the player), imagined vs true.
until grep -q "LANE156_DONE" $LOGDIR/lanes.log; do sleep 30; done
cp artifacts/eda/levers_logs/e22z_imagine_v2.log artifacts/eda/levers_logs/e22z_imagine_v2_saved.log
job e22z_imagine_v3 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000_prior_u20000.pt --samples 4
echo "$(date '+%F %T') LANE157_DONE" >> $LOGDIR/lanes.log
