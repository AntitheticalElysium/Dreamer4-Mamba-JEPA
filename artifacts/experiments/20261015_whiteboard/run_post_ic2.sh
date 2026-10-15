#!/bin/sh
# Second post-IC stage (does not edit the running run_post_ic.sh): pursuit and decision proxies for the IC models.
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
LOG=../../eda/levers_logs/post_ic2.log
until grep -q POSTDONE ../../eda/levers_logs/post_ic.log 2>/dev/null; do sleep 60; done
$P pursuit.py ic_s1 esrep_s1 ic_s1_ref ic_s2 esrep_s2 >> $LOG 2>&1
mv ../../eda/whiteboard/pursuit.json ../../eda/whiteboard/pursuit_ic.json
$P decision_proxies.py ic_s1 esrep_s1 ic_s2 esrep_s2 >> $LOG 2>&1
echo POST2DONE >> $LOG
