# 2026-10-10 19:2x: seed-8 replication readouts (L1 side) on lane181's exports, started when sb_export_s8 is DONE.
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export CUDA_VISIBLE_DEVICES=
until grep -q "DONE sb_export_s8" artifacts/eda/levers_logs/lanes.log; do
  grep -q "FAILED sb_export_s8" artifacts/eda/levers_logs/lanes.log && exit 1; sleep 30; done
run() { n=$1; shift; echo "$(date '+%F %T') START $n" >> artifacts/eda/levers_logs/lanes.log
  .venv/bin/python "$@" > artifacts/eda/levers_logs/$n.log 2>&1 && echo "$(date '+%F %T') DONE $n" >> artifacts/eda/levers_logs/lanes.log \
    || echo "$(date '+%F %T') FAILED $n" >> artifacts/eda/levers_logs/lanes.log; }
S="M6s8_6k M6s8_12k M6s8_18k M6s8_24k M6s8_30k M6s8_36k M16s8"
L=artifacts/experiments/20260927_levers
run s8_budget $L/budget.py $S
run s8_currency $L/currency.py $S
run s8_sb_a $L/scoreboard.py M6s8_6k M6s8_12k
run s8_sb_b $L/scoreboard.py M6s8_18k M6s8_24k
run s8_sb_c $L/scoreboard.py M6s8_30k M6s8_36k
run s8_sb_d $L/scoreboard.py M16s8
run s8_statoracle $L/statoracle.py M16 DC DS E22MS_zeroed E22MS_sample0 E22MS_postcodes M16s8
echo "$(date '+%F %T') CPU182_DONE" >> artifacts/eda/levers_logs/lanes.log
