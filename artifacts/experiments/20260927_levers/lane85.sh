#!/usr/bin/env bash
# Keep time rows fixed while removing only older visual history.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
job e17_fixed_clock 1300 $PY -B $L/e17_clock_control.py
