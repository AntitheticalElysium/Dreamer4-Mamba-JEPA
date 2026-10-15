#!/bin/sh
# PI_ENTER_PREREG.md: evaluation-only variant, readings, verdict.
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
LOG=../../eda/levers_logs
$P pi_canvas.py pi_enter_s1 pi_enter_s2 > $LOG/pi_enter.log 2>&1
$P census_split.py pi_enter_s1 pi_enter_s2 >> $LOG/pi_enter.log 2>&1
$P mob_fading.py pi_enter_s1 pi_enter_s2 >> $LOG/pi_enter.log 2>&1
$P seen_split.py pi_enter_s1 pi_enter_s2 >> $LOG/pi_enter.log 2>&1
$P entering_memory.py pi_enter_s1 pi_enter_s2 >> $LOG/pi_enter.log 2>&1
$P decision_proxies.py pi_enter_s1 pi_enter_s2 >> $LOG/pi_enter.log 2>&1
$P ../20261012_oracle/transport_track_verdicts.py enter >> $LOG/pi_enter.log 2>&1
echo ENTERDONE >> $LOG/pi_enter.log
