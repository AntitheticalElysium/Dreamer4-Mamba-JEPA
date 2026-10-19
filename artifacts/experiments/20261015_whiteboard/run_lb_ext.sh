#!/bin/sh
# LB_PREREG.md declared extension (the 25-30k token loss was still > 5% below 20-25k): train to 60k, evaluate there, verdict.
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
E=../../eda/oracle_v1/transport
LOG=../../eda/levers_logs
$P lb_world.py --seed-set 1 --updates 60000 --extend > $LOG/lb_seed1_ext.log 2>&1 &
$P lb_world.py --seed-set 2 --updates 60000 --extend > $LOG/lb_seed2_ext.log 2>&1 &
wait
$P lb_eval.py 1 60000 lb60k_s1 > $LOG/lb60k_eval.log 2>&1
$P lb_eval.py 2 60000 lb60k_s2 >> $LOG/lb60k_eval.log 2>&1
$P mob_fading.py lb60k_s1 lb60k_s2 >> $LOG/lb60k_eval.log 2>&1
$P census_split.py lb60k_s1 lb60k_s2 >> $LOG/lb60k_eval.log 2>&1
$P pursuit.py lb60k_s1 lb60k_s2 >> $LOG/lb60k_eval.log 2>&1
mv ../../eda/whiteboard/pursuit.json ../../eda/whiteboard/pursuit_lb60k.json
$P decision_proxies.py lb60k_s1 lb60k_s2 >> $LOG/lb60k_eval.log 2>&1
$P entering_memory.py lb60k_s1 lb60k_s2 >> $LOG/lb60k_eval.log 2>&1
$P ../20261012_oracle/transport_track_verdicts.py lb60k >> $LOG/lb60k_eval.log 2>&1
echo LB60DONE >> $LOG/lb60k_eval.log
