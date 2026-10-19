#!/usr/bin/env bash
# FROZEN_STATE_CONTROL_PREREG.md (+ Amendment 1) stage-A evaluations, per seed as its FD checkpoint appears; then the paired analyses.
# Existing corrected evaluations only. GPU work is light; one evaluation process at a time.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)/spatial_encoder_v1"
export EVAL_DEVICE=cuda OMP_NUM_THREADS=4
for s in 1 2; do
  until [ -f "$OUT/frozen_fd_s$s/step-010000.pt" ]; do sleep 120; done
  sleep 30
  E="spatial:$OUT/stage2_full_sig_s$s/step-010000.pt"
  P="$OUT/probes2_full_s$s.pt"
  $PY location_dynamics.py "FD_s$s=$E,$OUT/frozen_fd_s$s/step-010000.pt,$P"
  $PY consequence_check.py "fd_s$s=$E,$OUT/frozen_fd_s$s/step-010000.pt,$P"
  $PY census_localization.py "fd_s$s=$E,$OUT/frozen_fd_s$s/step-010000.pt,$P"
  $PY erasure_diagnosis.py $(for k in 1000 2000 4000 10000; do echo "fd_s$s@$k=$E,$OUT/frozen_fd_s$s/step-$(printf %06d $k).pt"; done)
  EVAL_DEVICE=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $PY tierb_eval_v2.py "fd_s$s=$E,$OUT/frozen_fd_s$s/step-010000.pt,$P"
done
export EVAL_DEVICE=cpu CUDA_VISIBLE_DEVICES=
$PY location_dynamics.py --analyse FD_s1 FD_s2 J_full_s1 J_full_s2 rawpred_s1 rawpred_s2 \
  --pairs FD_s1:J_full_s1,FD_s2:J_full_s2,FD_s1:rawpred_s1,FD_s2:rawpred_s2 --tag stageA
$PY consequence_within_root.py fd_s1 fd_s2 full_s1 full_s2 rawpred_s1 rawpred_s2 mop_s1 mop_s2
$PY stage2_analysis_v2.py --prefix-sp fd_s --prefix-raw full_s --seeds 1,2 --tag _fd_vs_J
$PY stage2_analysis_v2.py --prefix-sp fd_s --prefix-raw rawpred_s --seeds 1,2 --tag _fd_vs_raw
echo frozen_eval_A_done
