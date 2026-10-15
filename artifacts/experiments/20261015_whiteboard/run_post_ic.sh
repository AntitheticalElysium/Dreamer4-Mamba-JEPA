#!/bin/sh
# After TRANSPORT_IC v2 training+evaluation: preregistered diagnostics, recursion readings, then the deferred causal test.
set -u
cd "$(dirname "$0")/../20261012_oracle"
P=../../../.venv/bin/python
E=../../eda/oracle_v1/transport
LOG=../../eda/levers_logs/post_ic.log
until grep -q END ../../eda/levers_logs/ic_seed1_eval.log 2>/dev/null && grep -q END ../../eda/levers_logs/ic_seed2_eval.log 2>/dev/null; do sleep 60; done
for s in 1 2; do
  if [ "$s" = 1 ]; then FREE=$E/eval_u3000/pred_free.pt; else FREE=$E/seed2/eval_u3000/pred_free.pt; fi
  for arm in operator ic; do
    $P transport_diag.py content 3000 $E/ic_seed$s $arm $E/ic_seed$s >> $LOG 2>&1
    $P transport_diag.py swaps 3000 $E/ic_seed$s/eval_u3000 $arm $FREE >> $LOG 2>&1
  done
done
$P transport_recursion.py ic_s1 esrep_s1 ic_s2 esrep_s2 ic_s1_ref >> $LOG 2>&1
$P transport_ic_verdict.py >> $LOG 2>&1
cd ../20261015_whiteboard
$P mob_fading.py ic_s1 esrep_s1 ic_s1_ref >> $LOG 2>&1
$P camera_diag.py ic_s1 esrep_s1 >> $LOG 2>&1
$P entering_memory.py ic_s1 esrep_s1 >> $LOG 2>&1
$P faced_substitution.py op_s1 >> $LOG 2>&1
echo POSTDONE >> $LOG
