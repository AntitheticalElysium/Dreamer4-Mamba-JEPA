#!/usr/bin/env bash
# Waits until the T1 waiter has finished and no T1 lane is running, then launches the three T2c lanes.
set -u
R=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA; X=$R/artifacts/experiments/20261023_system_design
while systemctl --user is-active -q t1-waiter || systemctl --user is-active -q t1-lane0 || systemctl --user is-active -q t1-lane1 || systemctl --user is-active -q t1-lane2 \
      || systemctl --user is-active -q t2-lane0 || systemctl --user is-active -q t2-lane1 || systemctl --user is-active -q t2-lane2; do sleep 30; done
for i in 0 1 2; do systemd-run --user --unit t2c-lane$i -q -p MemoryMax=9G --working-directory=$R -E OMP_NUM_THREADS=3 $X/t2c_lane.sh $X/t2c_lane$i.txt; done
