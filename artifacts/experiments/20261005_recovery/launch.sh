#!/usr/bin/env bash
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
R=artifacts/experiments/20261005_recovery
.venv/bin/python -B - <<'PY'
import json
from pathlib import Path
r=Path('artifacts/experiments/20261005_recovery')
assert (r/'inputs.json').exists(), 'Inputs not verified'
t=json.loads((r/'gpu_resume.json').read_text())
assert t['start']==13000 and t['parameter_max_abs']<=1e-6 and t['optimizer_tensor_max_abs']<=1e-6
assert t['order_rng_equal'] and t['cpu_rng_equal'] and t['cuda_rng_equal']
PY
for name in lead e17 e18 h16-s8; do
  state=$(systemctl --user show d4mj-oct05-$name.service -p ActiveState --value)
  if [ "$state" = active ] || [ "$state" = activating ]; then
    echo "Active recovery service: d4mj-oct05-$name.service; refusing duplicate launch" >&2
    exit 1
  fi
done
.venv/bin/python -B "$R/verify_restart.py"
systemd-run --user --unit=d4mj-oct05-lead --property=MemoryHigh=20G --property=MemoryMax=24G --working-directory="$PWD" \
  /bin/bash artifacts/experiments/20260927_levers/lane81.sh
if [ ! -f artifacts/experiments/20260927_levers/evals/e19_B_s7_fmamba_from36000__e19_train_mechanism.json ]; then
  state=$(systemctl --user show d4mj-oct05-e19-mechanism.service -p ActiveState --value)
  if [ "$state" != active ] && [ "$state" != activating ]; then
    systemd-run --user --collect --unit=d4mj-oct05-e19-mechanism --property=MemoryHigh=8G --property=MemoryMax=10G --working-directory="$PWD" \
      /bin/bash artifacts/experiments/20260927_levers/lane83.sh
  fi
fi
if [ ! -f artifacts/experiments/20260927_levers/evals/corrt_rawlong_teacher_s8_L16b40_from36000__e17_fixed_clock.json ]; then
  state=$(systemctl --user show d4mj-oct05-e17-clock.service -p ActiveState --value)
  if [ "$state" != active ] && [ "$state" != activating ]; then
    systemd-run --user --collect --unit=d4mj-oct05-e17-clock --property=MemoryHigh=6G --property=MemoryMax=8G --working-directory="$PWD" \
      /bin/bash artifacts/experiments/20260927_levers/lane85.sh
  fi
fi
state=$(systemctl --user show d4mj-oct06-diagnostic-replicas.service -p ActiveState --value)
if [ "$state" != active ] && [ "$state" != activating ]; then
  systemd-run --user --unit=d4mj-oct06-diagnostic-replicas --property=MemoryHigh=8G --property=MemoryMax=10G --working-directory="$PWD" \
    /bin/bash artifacts/experiments/20260927_levers/lane86.sh
fi
state=$(systemctl --user show d4mj-oct06-cpu-recurrence.service -p ActiveState --value)
if [ "$state" != active ] && [ "$state" != activating ]; then
  if [ ! -f artifacts/experiments/20260927_levers/evals/corrt_rawlong_teacher_s8_fmamba_L16b40_from36000__e17_recurrence_cpu.json ]; then
    systemd-run --user --unit=d4mj-oct06-cpu-recurrence --property=MemoryHigh=4G --property=MemoryMax=6G --working-directory="$PWD" \
      /bin/bash artifacts/experiments/20260927_levers/lane87.sh
  fi
fi
state=$(systemctl --user show d4mj-oct06-cpu-recurrence-full.service -p ActiveState --value)
if [ "$state" != active ] && [ "$state" != activating ]; then
  if [ ! -f artifacts/experiments/20260927_levers/evals/corrt_rawlong_teacher_s8_fmamba_L16b40_from36000__e17_recurrence_full_cpu.json ]; then
    systemd-run --user --unit=d4mj-oct06-cpu-recurrence-full --property=MemoryHigh=4G --property=MemoryMax=6G --working-directory="$PWD" \
      /bin/bash artifacts/experiments/20260927_levers/lane88.sh
  fi
fi
state=$(systemctl --user show d4mj-oct06-cpu-conv-channels.service -p ActiveState --value)
if [ "$state" != active ] && [ "$state" != activating ]; then
  if [ ! -f artifacts/experiments/20260927_levers/evals/corrt_rawlong_teacher_s8_fmamba_L16b40_from36000__e17_conv_channels_cpu.json ]; then
    systemd-run --user --unit=d4mj-oct06-cpu-conv-channels --property=MemoryHigh=4G --property=MemoryMax=6G --working-directory="$PWD" \
      /bin/bash artifacts/experiments/20260927_levers/lane89.sh
  fi
fi
for item in 'cpu-endpoints lane90 e18_s7_at36000__endpoint_cpu' \
            'cpu-canvas36k lane94 e18_reroute_s7_at36000'; do
  read -r unit lane result <<< "$item"
  state=$(systemctl --user show "d4mj-oct06-$unit.service" -p ActiveState --value)
  if [ "$state" != active ] && [ "$state" != activating ] && \
     [ ! -f "artifacts/experiments/20260927_levers/evals/$result.json" ]; then
    systemd-run --user --unit="d4mj-oct06-$unit" --property=MemoryHigh=6G --property=MemoryMax=8G --working-directory="$PWD" \
      /bin/bash "artifacts/experiments/20260927_levers/$lane.sh"
  fi
done
if [ "$(systemctl --user show d4mj-oct05-status.service -p LoadState --value)" = loaded ]; then
  systemctl --user restart d4mj-oct05-status.service
else
  systemd-run --user --unit=d4mj-oct05-status --working-directory="$PWD" \
    "$PWD/.venv/bin/python" -B "$R/monitor.py"
fi
