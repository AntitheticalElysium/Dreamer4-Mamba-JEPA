#!/usr/bin/env bash
# CPU only: extend the completed mechanism contrast using validated prefix reuse.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
job e17_recurrence_full_cpu 0 $PY -B $L/e17_recurrence_full.py
