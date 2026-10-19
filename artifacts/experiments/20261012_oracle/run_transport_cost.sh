#!/bin/sh
# TRANSPORT_COST_PREREG.md: one seed-set run (train, then sample + analyze at 500/1500/3000). Usage: run_transport_cost.sh 1|2
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
E=../../eda/oracle_v1/transport
LOG=../../eda/levers_logs
if [ "$1" = 1 ]; then ARMS=operator,operator_cost; DIR=$E/seed1_cost; else ARMS=free,operator,operator_cost; DIR=$E/seed2; fi
$P transport_train.py --seed-set "$1" --arms $ARMS --cost 1e-4 --directory $DIR > $LOG/cost_seed$1_train.log 2>&1
for u in 0500 1500 3000; do
  $P transport_eval.py sample $u --train $DIR --eval $DIR/eval_u$u --arms $ARMS >> $LOG/cost_seed$1_eval.log 2>&1
  if [ "$1" = 1 ]; then
    $P transport_eval.py analyze $u --eval $DIR/eval_u$u --arms $ARMS --free $E/eval_u$u/pred_free.pt >> $LOG/cost_seed$1_eval.log 2>&1
  else
    $P transport_eval.py analyze $u --eval $DIR/eval_u$u --arms $ARMS >> $LOG/cost_seed$1_eval.log 2>&1
  fi
done
echo END >> $LOG/cost_seed$1_eval.log
