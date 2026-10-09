source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 18:51: E22M + hard interior decoding: prior vs decoder (true posterior codes, self-fed frames; 1 prior sample for reference).
cp artifacts/eda/health_chain_v1/e22_imagine_dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000_interior.json artifacts/eda/health_chain_v1/e22_imagine_dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000_interior_s4.json
job e22h_m_interior_postcodes 2000 $PY $L/e22_imagine.py artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000_prior_u20000.pt --samples 1 --postcodes --decode interior
echo "$(date '+%F %T') LANE165_DONE" >> $LOGDIR/lanes.log
