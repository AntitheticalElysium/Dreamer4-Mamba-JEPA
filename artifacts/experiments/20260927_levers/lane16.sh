source artifacts/experiments/20260927_levers/lib.sh
# E5j: re-localize corrt's remaining error at 18k (where / static / substitution), small GPU jobs
W=artifacts/eda/levers_tworlds_v1
JAX_PLATFORMS=cpu job where_corrt18 1500 $PY $L/where.py $W/corrt_raw_suffix_s7_u18000.pt
JAX_PLATFORMS=cpu job static_corrt18 1500 $PY $L/static.py $W/corrt_raw_suffix_s7_u18000.pt
JAX_PLATFORMS=cpu job substitute_corrt18 1500 $PY $L/substitute.py $W/corrt_raw_suffix_s7_u18000.pt
echo "$(date '+%F %T') LANE16_DONE" >> $LOGDIR/lanes.log
