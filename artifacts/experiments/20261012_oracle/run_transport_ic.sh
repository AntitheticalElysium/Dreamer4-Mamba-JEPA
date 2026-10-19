#!/bin/sh
# TRANSPORT_IC_PREREG.md: one seed set (train operator+ic, then sample + analyze at 500/1500/3000). Usage: run_transport_ic.sh 1|2
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
E=../../eda/oracle_v1/transport
LOG=../../eda/levers_logs
DIR=$E/ic_seed$1
$P transport_ic_train.py --seed-set "$1" --arms operator,ic --directory $DIR > $LOG/ic_seed$1_train.log 2>&1
for u in 0500 1500 3000; do
  if [ "$1" = 1 ]; then FREE=$E/eval_u$u/pred_free.pt; else FREE=$E/seed2/eval_u$u/pred_free.pt; fi
  $P transport_eval.py sample $u --train $DIR --eval $DIR/eval_u$u --arms operator,ic >> $LOG/ic_seed$1_eval.log 2>&1
  $P transport_eval.py analyze $u --eval $DIR/eval_u$u --arms operator,ic --free $FREE >> $LOG/ic_seed$1_eval.log 2>&1
done
echo END >> $LOG/ic_seed$1_eval.log
