#!/bin/sh
# PI_STALE_PREREG.md: both seeds (two arms per process), one-step evaluation, recursion, substitution gap, readings, verdict.
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
E=../../eda/oracle_v1/transport
LOG=../../eda/levers_logs
$P pi_stale_train.py --seed-set 1 > $LOG/stale_seed1_train.log 2>&1 &
$P pi_stale_train.py --seed-set 2 > $LOG/stale_seed2_train.log 2>&1 &
wait
cd ../20261012_oracle
for s in 1 2; do
  if [ "$s" = 1 ]; then FREE=$E/eval_u3000/pred_free.pt; else FREE=$E/seed2/eval_u3000/pred_free.pt; fi
  $P transport_eval.py sample 4000 --train $E/pi_seed$s --eval $E/pi_seed$s/eval_u4000 --arms piplain,pistale >> $LOG/stale_eval.log 2>&1
  $P transport_eval.py analyze 4000 --eval $E/pi_seed$s/eval_u4000 --arms piplain,pistale --free $FREE >> $LOG/stale_eval.log 2>&1
done
$P transport_recursion.py piplain_s1 pistale_s1 piplain_s2 pistale_s2 >> $LOG/stale_eval.log 2>&1
cd ../20261015_whiteboard
for a in piplain pistale; do
  STALE_CKPT="pi_seed{s}/${a}_u4000.pt" STALE_TAG="_$a" $P stale_context.py >> $LOG/stale_eval.log 2>&1
done
$P mob_fading.py piplain_s1 pistale_s1 piplain_s2 pistale_s2 >> $LOG/stale_eval.log 2>&1
$P pursuit.py piplain_s1 pistale_s1 piplain_s2 pistale_s2 >> $LOG/stale_eval.log 2>&1
mv ../../eda/whiteboard/pursuit.json ../../eda/whiteboard/pursuit_stale.json
$P decision_proxies.py piplain_s1 pistale_s1 piplain_s2 pistale_s2 >> $LOG/stale_eval.log 2>&1
$P ../20261012_oracle/transport_track_verdicts.py stale >> $LOG/stale_eval.log 2>&1
echo STALEDONE >> $LOG/stale_eval.log
