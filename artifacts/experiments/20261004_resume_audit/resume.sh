#!/usr/bin/env bash
# Finish short, already-declared diagnostics before booking both training lanes.
source artifacts/experiments/20260927_levers/lib.sh
W=artifacts/eda/levers_tworlds_v1
job oct04_check_delta 2800 $PY -B $L/check_delta.py $W/corrt_raw_teacher_s7_fmamba_u36000.pt $W/corrt_raw_teacher_s8_fmamba_u36000.pt
job oct04_h16traj_m6 2800 $PY -B $L/check_h16_traj.py $W/corrt_raw_teacher_s7_fmamba_u36000.pt $W/corrt_raw_teacher_s8_fmamba_u36000.pt
# These units may have been admitted early while the large H16 feature pass runs; never create duplicate trainers.
if [ "$(systemctl --user show d4mj-oct04-e17.service -p LoadState --value)" != loaded ]; then
  systemd-run --user --unit=d4mj-oct04-e17 --property=MemoryHigh=10G --property=MemoryMax=12G --working-directory="$PWD" \
    /bin/bash artifacts/experiments/20260927_levers/lane73.sh
fi
if [ "$(systemctl --user show d4mj-oct04-e18.service -p LoadState --value)" != loaded ]; then
  systemd-run --user --unit=d4mj-oct04-e18 --property=MemoryHigh=10G --property=MemoryMax=12G --working-directory="$PWD" \
    /bin/bash artifacts/experiments/20260927_levers/lane74.sh
fi
echo "$(date '+%F %T') OCT04_RESUME_COORDINATOR_DONE" >> "$LOGDIR/lanes.log"
