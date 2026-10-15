#!/bin/sh
# TRANSPORT_SF_PREREG.md: one seed set (continuation training, one-step evaluation, recursion, fading, pursuit, decisions).
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
E=../../eda/oracle_v1/transport
LOG=../../eda/levers_logs
DIR=$E/sf_seed$1
$P transport_sf_train.py --seed-set "$1" > $LOG/sf_seed$1_train.log 2>&1
if [ "$1" = 1 ]; then FREE=$E/eval_u3000/pred_free.pt; else FREE=$E/seed2/eval_u3000/pred_free.pt; fi
for u in 3500 4500; do
  $P transport_eval.py sample $u --train $DIR --eval $DIR/eval_u$u --arms ic,icsf >> $LOG/sf_seed$1_eval.log 2>&1
  $P transport_eval.py analyze $u --eval $DIR/eval_u$u --arms ic,icsf --free $FREE >> $LOG/sf_seed$1_eval.log 2>&1
done
$P transport_recursion.py sf_ic_s$1 sf_icsf_s$1 >> $LOG/sf_seed$1_eval.log 2>&1
cd ../20261015_whiteboard
$P mob_fading.py sf_ic_s$1 sf_icsf_s$1 >> $LOG/sf_seed$1_eval.log 2>&1
$P pursuit.py sf_ic_s$1 sf_icsf_s$1 >> $LOG/sf_seed$1_eval.log 2>&1
mv ../../eda/whiteboard/pursuit.json ../../eda/whiteboard/pursuit_sf_s$1.json
$P decision_proxies.py sf_ic_s$1 sf_icsf_s$1 >> $LOG/sf_seed$1_eval.log 2>&1
echo END >> $LOG/sf_seed$1_eval.log
