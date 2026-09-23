#!/usr/bin/env bash
# Frozen Raw readout ladder + confirmation. Nothing is trained but the probe heads; the world is
# the sealed H2 bridge checkpoint and is never updated, so this cannot disturb the M4 run.
#
#   ./run.sh ladder    the four-rung ladder      -> evidence/ladder.json   (RESULT.md)
#   ./run.sh confirm   trained head + exact refit -> evidence/confirm.json (CONFIRM.md)
#   ./run.sh all       both
#
# TRITON_F32_DEFAULT=ieee is NOT optional: sources.py records the numeric execution flags in
# every checkpoint and verify_lewm_sources is strict, so loading the bridge parent without it
# fails with "source/dependency drift in ['execution']".
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments is deliberately NOT set. After the 2026-09-23 system
# update the loaded NVIDIA kernel module (610.57.04) is older than nvidia-utils (615.71.09), and
# the expandable-segments VMM path calls nvmlInit_v2_, which aborts model .to(device) with
# "NVML_SUCCESS == ... INTERNAL ASSERT FAILED at PeerToPeerAccess.cpp". The default allocator
# avoids that path. A reboot is the real fix; this keeps the measurement runnable without one.
set -uo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee JAX_PLATFORMS=cpu PYTHONPATH=.
HERE=artifacts/experiments/20260921_readout_ladder
STAGE=${1:-ladder}
rc=0

if [ "$STAGE" = "ladder" ] || [ "$STAGE" = "all" ]; then
  # 8000/4000 exceed the sealed partition's 7085/3369 usable roots, so every allocated root is used.
  .venv/bin/python "$HERE/ladder.py" \
      --run artifacts/lewm_m4_canonical/raw \
      --checkpoint artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt \
      --train-roots 8000 --dev-roots 4000 --steps 6000 --draws 1000 \
      --out "$HERE/evidence" 2>&1 | grep -v "KernelPreference\|ScaleCalculationMode" | tee -a "$HERE/ladder.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && exit "$rc"
fi

if [ "$STAGE" = "confirm" ] || [ "$STAGE" = "all" ]; then
  # Fits on the 700 retired fit seeds; judges on the 405 the partition left unallocated, which no
  # head in this run was fitted or selected on. Both caps exceed the available roots on purpose.
  .venv/bin/python "$HERE/confirm.py" \
      --run artifacts/lewm_m4_canonical/raw \
      --checkpoint artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt \
      --fit-roots 8000 --judge-roots 8000 --steps 6000 --draws 1000 \
      --out "$HERE/evidence" 2>&1 | grep -v "KernelPreference\|ScaleCalculationMode" | tee -a "$HERE/confirm.log"
  rc=${PIPESTATUS[0]}
fi
exit "$rc"
