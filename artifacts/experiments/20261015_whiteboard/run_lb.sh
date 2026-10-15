#!/bin/sh
# LB_PREREG.md: evaluate LB at 10k / 20k (learning curve) and 30k (verdict), all readings, mechanical verdict.
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
E=../../eda/oracle_v1/transport
LOG=../../eda/levers_logs
for u in 10000 20000 30000; do
  until [ -f $E/lb_seed1/lb_u$(printf %05d $u).pt ] && [ -f $E/lb_seed2/lb_u$(printf %05d $u).pt ]; do sleep 60; done
  if [ "$u" = 30000 ]; then N=""; else N="_u$u"; fi
  $P lb_eval.py 1 $u lb${N}_s1 >> $LOG/lb_eval.log 2>&1
  $P lb_eval.py 2 $u lb${N}_s2 >> $LOG/lb_eval.log 2>&1
done
$P mob_fading.py lb_s1 lb_s2 >> $LOG/lb_eval.log 2>&1
$P census_split.py lb_s1 lb_s2 >> $LOG/lb_eval.log 2>&1
$P pursuit.py lb_s1 lb_s2 >> $LOG/lb_eval.log 2>&1
mv ../../eda/whiteboard/pursuit.json ../../eda/whiteboard/pursuit_lb.json
$P decision_proxies.py lb_s1 lb_s2 >> $LOG/lb_eval.log 2>&1
$P entering_memory.py lb_s1 lb_s2 >> $LOG/lb_eval.log 2>&1
$P move_by_score.py lb_s1 lb_s2 >> $LOG/lb_eval.log 2>&1
$P ../20261012_oracle/transport_track_verdicts.py lb >> $LOG/lb_eval.log 2>&1
echo LBDONE >> $LOG/lb_eval.log
