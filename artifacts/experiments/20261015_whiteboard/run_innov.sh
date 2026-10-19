#!/bin/sh
# PI_INNOV_PREREG.md: head-only content categorical (both seeds in parallel), sampled-content recursion, readings, verdict.
set -u
cd "$(dirname "$0")"
P=../../../.venv/bin/python
LOG=../../eda/levers_logs
$P pi_innov_train.py --seed-set 1 > $LOG/innov_seed1_train.log 2>&1 &
$P pi_innov_train.py --seed-set 2 > $LOG/innov_seed2_train.log 2>&1 &
wait
$P pi_canvas.py pi_innov_s1 pi_innov_s2 > $LOG/pi_innov.log 2>&1
$P census_split.py pi_innov_s1 pi_innov_s2 >> $LOG/pi_innov.log 2>&1
$P mob_fading.py pi_innov_s1 pi_innov_s2 >> $LOG/pi_innov.log 2>&1
$P entering_memory.py pi_innov_s1 pi_innov_s2 >> $LOG/pi_innov.log 2>&1
$P decision_proxies.py pi_innov_s1 pi_innov_s2 >> $LOG/pi_innov.log 2>&1
$P ../20261012_oracle/transport_track_verdicts.py innov >> $LOG/pi_innov.log 2>&1
echo INNOVDONE >> $LOG/pi_innov.log
