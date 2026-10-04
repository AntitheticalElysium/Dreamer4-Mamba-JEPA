source artifacts/experiments/20260927_levers/lib.sh
# E9 ablation (2026-10-01): is the corrt copy head what makes the one-step decision readable? direct / residual heads
# at 6k (seed 7, suffix) vs corrt 6k (group 1). spatial.py's T, which did not carry the decision, used the direct output.
W=artifacts/eda/levers_tworlds_v1
until grep -qE "DONE dpanel_group2|FAILED dpanel_group2" $LOGDIR/lanes.log; do sleep 60; done
[ -f $L/evals/dpanel_residual_raw_suffix_s7.json ] || job dpanel_group3 3000 $PY $L/dpanel.py $W/direct_raw_suffix_s7.pt $W/residual_raw_suffix_s7.pt
echo "$(date '+%F %T') LANE19C_DONE" >> $LOGDIR/lanes.log
