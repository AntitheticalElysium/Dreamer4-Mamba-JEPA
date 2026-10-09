source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 16:16: prior vs decoder in sampled imagination (E22M fades, E22Z multiplies): rollout with the posterior's codes of the TRUE
# transitions (self-fed frames), 1 prior sample for reference; the 4-sample result files are kept as *_s4.json.
until grep -q "LANE158_DONE" $LOGDIR/lanes.log; do sleep 30; done
cp artifacts/eda/health_chain_v1/e22_imagine_dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000.json artifacts/eda/health_chain_v1/e22_imagine_dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000_s4.json; cp artifacts/eda/health_chain_v1/e22_imagine_dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000_prior_u20000.json artifacts/eda/health_chain_v1/e22_imagine_dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000_prior_u20000_s4.json
job e22m_postcodes 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000.pt --samples 1 --postcodes
job e22z_postcodes 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000_prior_u20000.pt --samples 1 --postcodes
echo "$(date '+%F %T') LANE159_DONE" >> $LOGDIR/lanes.log
