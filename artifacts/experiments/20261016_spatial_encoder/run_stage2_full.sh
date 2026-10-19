#!/usr/bin/env bash
# Amendment 5: the full arm at Stage 2 (2 seeds, same protocol), after the current Stage 2 runner has finished, then the identical
# corrected evaluation (CPU, after chain_v2 has evaluated MoP + SIGReg and Raw-predictor) and both comparisons.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)/spatial_encoder_v1"
while systemctl --user is-active -q spatial-stage2; do sleep 120; done
for s in 1 2; do
  seed=$((20260730 + s))
  [ -f "$OUT/stage2_full_sig_s$s/step-010000.pt" ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY spatial_encoder.py full_sig --out "$OUT/stage2_full_sig_s$s" --stop-at 10000 --seed $seed
done
while systemctl --user is-active -q spatial-chain-v2; do sleep 120; done
export EVAL_DEVICE=cpu OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
$PY spatial_eval_v2.py full_s1=spatial:$OUT/stage2_full_sig_s1/step-010000.pt full_s2=spatial:$OUT/stage2_full_sig_s2/step-010000.pt
$PY tierb_eval_v2.py \
  full_s1=spatial:$OUT/stage2_full_sig_s1/step-010000.pt,$OUT/stage2_full_sig_s1/step-010000.pt,$OUT/probes2_full_s1.pt \
  full_s2=spatial:$OUT/stage2_full_sig_s2/step-010000.pt,$OUT/stage2_full_sig_s2/step-010000.pt,$OUT/probes2_full_s2.pt
$PY stage2_analysis_v2.py --prefix-sp full_s --prefix-raw rawpred_s --seeds 1,2 --tag _full
$PY stage2_analysis_v2.py --prefix-sp full_s --prefix-raw spatial_s --seeds 1,2 --tag _full_vs_mop
echo stage2_full_done
