#!/usr/bin/env bash
# Localization inside Direct, both geometries and both tokenizer seeds. No training.
# Waits for the milestone rescore so the two do not share the GPU.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }

until grep -q "milestone rescore finished" triage.log; do sleep 20; done

for suf in s1 s0; do
  for n in 64 32; do
    [ -f "direct_path_${suf}_n${n}_020000.json" ] || \
      stage "B1_path_${suf}_n${n}" "$PY" probe_direct_path.py --n-latents $n --suffix $suf
  done
done
echo "direct path localization finished $(date -Is)" >> triage.log
