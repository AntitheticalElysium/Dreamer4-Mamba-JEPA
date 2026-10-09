# 2026-10-10 10:26: Part I/II CPU readouts on the finished exports, one at a time (systemd unit, MemoryMax 10G), after sb_e22ms.
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export CUDA_VISIBLE_DEVICES=
while systemctl --user is-active --quiet sb_e22ms; do sleep 20; done
run() { n=$1; shift; echo "$(date '+%F %T') START $n" >> artifacts/eda/levers_logs/lanes.log
  .venv/bin/python "$@" > artifacts/eda/levers_logs/$n.log 2>&1 && echo "$(date '+%F %T') DONE $n" >> artifacts/eda/levers_logs/lanes.log \
    || echo "$(date '+%F %T') FAILED $n" >> artifacts/eda/levers_logs/lanes.log; }
run sb_det artifacts/experiments/20260927_levers/scoreboard.py DC DS
run budget_det artifacts/experiments/20260927_levers/budget.py DC DS
run entmem_all artifacts/experiments/20260927_levers/entering_memory.py DC DS E22MS_sample0 E22MS_sample1 E22MS_sample2 E22MS_sample3 E22MS_postcodes
run sbcmp_M16 artifacts/experiments/20260927_levers/scoreboard_compare.py M16 DC DS E22MS_zeroed E22MS E22MS_postcodes
echo "$(date '+%F %T') CPU175_DONE" >> artifacts/eda/levers_logs/lanes.log
