#!/usr/bin/env bash
# CPU only: separate input/write/read convolution dependencies.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
job e17_conv_channels_cpu 0 $PY -B $L/e17_conv_channels.py
