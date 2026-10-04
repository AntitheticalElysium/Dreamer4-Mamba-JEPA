source artifacts/experiments/20260927_levers/lib.sh
# E10 stage 2 (2026-10-01): FIT-dev seeds (replayed and checked), then the fresh judgement block 62,000-62,399
job deeppanel_dev 1200 $PY $L/deeppanel.py --mode replay --split dev --out artifacts/eda/deeppanel_dev_v1
job deeppanel_judge 1200 $PY $L/deeppanel.py --mode collect --seed-start 62000 --seeds 400 --out artifacts/eda/deeppanel_judge_v1
echo "$(date '+%F %T') LANE20B_DONE" >> $LOGDIR/lanes.log
