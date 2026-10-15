#!/bin/sh
# TRANSPORT_SF_PREREG.md, seed 2: its recursion OOMed (both seeds' recursions started together, ~3.9 GB each on 6 GB) and the
# downstream readings failed on the missing samples. Re-run them after seed 1's pipeline ends, then the SF verdict, then the
# PI pipeline (run_pi.sh, unchanged; its wait condition is already met).
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
LOG=../../eda/levers_logs
until grep -q '^END' $LOG/sf_seed1_eval.log 2>/dev/null; do sleep 60; done
$P transport_recursion.py sf_ic_s2 sf_icsf_s2 > $LOG/sf_seed2_rerun.log 2>&1
cd ../20261015_whiteboard
$P mob_fading.py sf_ic_s2 sf_icsf_s2 >> $LOG/sf_seed2_rerun.log 2>&1
$P pursuit.py sf_ic_s2 sf_icsf_s2 >> $LOG/sf_seed2_rerun.log 2>&1
mv ../../eda/whiteboard/pursuit.json ../../eda/whiteboard/pursuit_sf_s2.json
$P decision_proxies.py sf_ic_s2 sf_icsf_s2 >> $LOG/sf_seed2_rerun.log 2>&1
echo END >> $LOG/sf_seed2_rerun.log
$P ../20261012_oracle/transport_track_verdicts.py sf > $LOG/sf_verdict.log 2>&1
exec sh ./run_pi.sh
