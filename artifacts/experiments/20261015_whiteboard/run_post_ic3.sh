#!/bin/sh
# Third post-IC stage: posterior-forced recursion (chance supplied from the real future; deterministic part self-fed).
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
LOG=../../eda/levers_logs/post_ic3.log
until grep -q POST2DONE ../../eda/levers_logs/post_ic2.log 2>/dev/null; do sleep 60; done
$P posterior_forced.py ic_seed1/ic_u3000.pt >> $LOG 2>&1
$P posterior_forced.py ic_seed1/operator_u3000.pt >> $LOG 2>&1
echo POST3DONE >> $LOG
