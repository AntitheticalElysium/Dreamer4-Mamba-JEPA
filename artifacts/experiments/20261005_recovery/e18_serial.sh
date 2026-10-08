#!/usr/bin/env bash
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
# Actual M16 + desktop uses ~3,599 MiB of 5,850 MiB available device memory. Canvas cannot pass 2,556 MiB admission.
# Evaluators release memory during head fitting, then grow again within the same process. Hold the independent canvas
# lane until E17 finishes, avoiding a late memory collision after apparent free-memory admission. No treatment changes.
while systemctl --user is-active --quiet d4mj-oct05-e17.service; do
  sleep 30
done
exec /bin/bash artifacts/experiments/20260927_levers/lane74.sh
