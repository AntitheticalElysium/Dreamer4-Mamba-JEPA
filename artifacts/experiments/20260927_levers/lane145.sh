source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 10:06: E22 channel diagnostics at update 0 of both smokes (train_a's exact initialization, warm = D loaded, scratch = not).
I=artifacts/eda/health_chain_v1/e22_init
job e22_channel_u0 1600 $PY $L/e22_channel.py WARM0=$I/dworld_a_s7_warm_u0.pt SCRATCH0=$I/dworld_a_s7_scratch_u0.pt --out e22_channel_u0.json
echo "$(date '+%F %T') LANE145_DONE" >> $LOGDIR/lanes.log
