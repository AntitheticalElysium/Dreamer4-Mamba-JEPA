source artifacts/experiments/20260927_levers/lib.sh
# E10 stage 1 (2026-10-01): deep-panel label statistics on 10 fresh seeds (69,000-69,009), CPU simulator + GPU BC policy
job deeppanel_smoke 1200 $PY $L/deeppanel.py --seed-start 69000 --seeds 10 --keys 32 --out artifacts/eda/deeppanel_smoke_v1
echo "$(date '+%F %T') LANE19A_DONE" >> $LOGDIR/lanes.log
