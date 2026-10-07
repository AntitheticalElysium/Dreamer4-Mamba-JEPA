#!/usr/bin/env bash
# CPU only: separate SSM-history use from stacked temporal convolution use.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
job e17_recurrence_cpu 0 $PY -B $L/e17_recurrence.py
