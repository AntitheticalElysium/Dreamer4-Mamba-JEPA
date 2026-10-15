#!/bin/sh
# TRANSPORT_PI_PREREG.md (Track B): waits for the self-fed training to finish, smoke, both seeds, evaluation, recursion readings.
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
E=../../eda/oracle_v1/transport
LOG=../../eda/levers_logs
S=/tmp/claude-1000/-home-antithetical-EPITA-PERSO-DynamicHorizons-Mamba-JEPA/1bf3b794-6910-4310-90e3-9290b8496cc1/scratchpad/pi_smoke
until grep -q '"stage": "done"' $LOG/sf_seed1_train.log 2>/dev/null && grep -q '"stage": "done"' $LOG/sf_seed2_train.log 2>/dev/null; do sleep 60; done
rm -rf $S; mkdir -p $S
$P pi_train.py --seed-set 1 --stop-after 3 --directory $S > $LOG/pi_smoke.log 2>&1 || { echo SMOKEFAIL >> $LOG/pi_smoke.log; exit 1; }
for s in 1 2; do
  $P pi_train.py --seed-set $s > $LOG/pi_seed${s}_train.log 2>&1
done
cd ../20261012_oracle
for s in 1 2; do
  if [ "$s" = 1 ]; then FREE=$E/eval_u3000/pred_free.pt; else FREE=$E/seed2/eval_u3000/pred_free.pt; fi
  for u in 0500 1500 3000; do
    $P transport_eval.py sample $u --train $E/pi_seed$s --eval $E/pi_seed$s/eval_u$u --arms pi >> $LOG/pi_eval.log 2>&1
    $P transport_eval.py analyze $u --eval $E/pi_seed$s/eval_u$u --arms pi --free $FREE >> $LOG/pi_eval.log 2>&1
  done
done
$P transport_recursion.py pi_s1 pi_s2 pi_s1_ref >> $LOG/pi_eval.log 2>&1
cd ../20261015_whiteboard
$P mob_fading.py pi_s1 pi_s2 pi_s1_ref >> $LOG/pi_eval.log 2>&1
$P camera_diag.py pi_s1 pi_s2 >> $LOG/pi_eval.log 2>&1
$P pursuit.py pi_s1 pi_s2 >> $LOG/pi_eval.log 2>&1
mv ../../eda/whiteboard/pursuit.json ../../eda/whiteboard/pursuit_pi.json
$P decision_proxies.py pi_s1 pi_s2 >> $LOG/pi_eval.log 2>&1
$P entering_memory.py pi_s1 >> $LOG/pi_eval.log 2>&1
echo PIDONE >> $LOG/pi_eval.log
