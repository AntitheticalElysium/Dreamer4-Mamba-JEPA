#!/usr/bin/env bash
# Frozen Raw readout ladder. Nothing is trained but the probe heads; the world is the sealed
# H2 bridge checkpoint and is never updated, so this cannot disturb the M4 run.
#
# TRITON_F32_DEFAULT=ieee is NOT optional: sources.py records the numeric execution flags in
# every checkpoint and verify_lewm_sources is strict, so loading the bridge parent without it
# fails with "source/dependency drift in ['execution']".
set -uo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee JAX_PLATFORMS=cpu PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HERE=artifacts/experiments/20260921_readout_ladder
# 8000/4000 exceed the sealed partition's 7085/3369 usable roots, so every allocated root is used.
.venv/bin/python "$HERE/ladder.py" \
    --run artifacts/lewm_m4_canonical/raw \
    --checkpoint artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt \
    --train-roots 8000 --dev-roots 4000 --steps 6000 --draws 1000 \
    --out "$HERE/evidence" 2>&1 | grep -v "KernelPreference\|ScaleCalculationMode" | tee -a "$HERE/ladder.log"
exit "${PIPESTATUS[0]}"
