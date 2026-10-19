# 2026-10-10 18:3x: P-D (training dynamics) CPU readouts on the 12 stage-1 snapshot exports (M6 = L1, DS1 = D loss; 6k..36k),
# started automatically when lane173's sb_export_snapshots is DONE (systemd unit, MemoryMax 10G). One process per step.
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export CUDA_VISIBLE_DEVICES=
until grep -q "DONE sb_export_snapshots" artifacts/eda/levers_logs/lanes.log; do
  grep -q "FAILED sb_export_snapshots" artifacts/eda/levers_logs/lanes.log && exit 1; sleep 30; done
run() { n=$1; shift; echo "$(date '+%F %T') START $n" >> artifacts/eda/levers_logs/lanes.log
  .venv/bin/python "$@" > artifacts/eda/levers_logs/$n.log 2>&1 && echo "$(date '+%F %T') DONE $n" >> artifacts/eda/levers_logs/lanes.log \
    || echo "$(date '+%F %T') FAILED $n" >> artifacts/eda/levers_logs/lanes.log; }
S="M6_6k DS1_6k M6_12k DS1_12k M6_18k DS1_18k M6_24k DS1_24k M6_30k DS1_30k M6_36k DS1_36k"
L=artifacts/experiments/20260927_levers
run pd_budget $L/budget.py $S
run pd_currency $L/currency.py $S
run pd_entmem $L/entering_memory.py $S
run pd_sb_6k $L/scoreboard.py M6_6k DS1_6k
run pd_sb_12k $L/scoreboard.py M6_12k DS1_12k
run pd_sb_18k $L/scoreboard.py M6_18k DS1_18k
run pd_sb_24k $L/scoreboard.py M6_24k DS1_24k
run pd_sb_30k $L/scoreboard.py M6_30k DS1_30k
run pd_sb_36k $L/scoreboard.py M6_36k DS1_36k
echo "$(date '+%F %T') CPU180_DONE" >> artifacts/eda/levers_logs/lanes.log
