source artifacts/experiments/20260927_levers/lib.sh
# E10 stage 2 (2026-10-01): deep labels + depth frames for the FIT-train seeds (replayed and checked)
job deeppanel_fit 1200 $PY $L/deeppanel.py --mode replay --split fit --out artifacts/eda/deeppanel_fit_v1
echo "$(date '+%F %T') LANE20A_DONE" >> $LOGDIR/lanes.log
