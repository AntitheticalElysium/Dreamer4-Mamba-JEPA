#!/usr/bin/env bash
# Collect -> encode -> smoke -> train matched Direct-T and Direct-M, unattended.
# Every stage is resumable and skips itself if already done, so re-running is safe.
set -u
cd "$(dirname "$0")"
R=../..
PY=$R/.venv/bin/python
export PYTHONPATH=$R

say() { echo "[$(date +%H:%M:%S)] == $*"; }

say "waiting for collection"
while pgrep -f "[c]ollect_broad_forks" > /dev/null; do sleep 60; done
roots=$(ls broad_forks_v2/seed-*-r*.pt 2>/dev/null | sed 's/.*-r0*//;s/\.pt//' \
        | awk '{s+=$1} END {print s+0}')
say "collection done: $roots roots"

say "encoding"
$PY encode_broad_forks.py || { say "ENCODE FAILED"; exit 1; }
say "encoded $(ls broad_latents_v2/shard-*.pt 2>/dev/null | wc -l) shards"

say "target smoke"
$PY smoke_broad_forks_targets.py || { say "SMOKE FAILED"; exit 1; }

for mixer in attention mamba; do
  out=v2_direct_${mixer}
  if [ -f "$out/world.pt" ]; then say "$out already trained"; continue; fi
  say "training $out"
  $PY train_terminal_arms.py --arm counterfactual --roots v2 \
      --terminal-roots 4 --terminal-actions 17 --steps 20000 \
      --time-mixer "$mixer" --out "$out" || { say "TRAIN $mixer FAILED"; exit 1; }
done
say "pipeline complete"
