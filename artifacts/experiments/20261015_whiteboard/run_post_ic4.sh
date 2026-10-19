#!/bin/sh
# Fourth stage: the preregistered posterior diagnostic at IC milestones (precision / capture vs T*, training windows).
set -u
cd "$(dirname "$0")/../20261012_oracle"
until grep -q POST3DONE ../../eda/levers_logs/post_ic3.log 2>/dev/null; do sleep 60; done
../../../.venv/bin/python ic_posterior.py milestones > ../../eda/levers_logs/post_ic4.log 2>&1
echo POST4DONE >> ../../eda/levers_logs/post_ic4.log
