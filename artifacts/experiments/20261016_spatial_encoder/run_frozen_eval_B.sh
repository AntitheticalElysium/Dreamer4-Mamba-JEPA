#!/usr/bin/env bash
# FROZEN_STATE_CONTROL_PREREG.md stage-B evaluations (frozen full-arm BACKBONE-LN predictors), per seed as checkpoints appear; then the
# registered comparisons (FB vs FD, FB vs Raw). Backbone-stage readers: spatial_eval_v2's a1 procedure on the spatial-bb: stage.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)/spatial_encoder_v1"
for s in 1 2; do
  until [ -f "$OUT/frozen_fb_s$s/step-010000.pt" ]; do sleep 120; done
  sleep 30
  E="spatial-bb:$OUT/stage2_full_sig_s$s/step-010000.pt"
  [ -f "$OUT/probes2_fullbb_s$s.pt" ] || EVAL_DEVICE=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $PY spatial_eval_v2.py "fullbb_s$s=$E"
  P="$OUT/probes2_fullbb_s$s.pt"
  export EVAL_DEVICE=cuda OMP_NUM_THREADS=4
  $PY location_dynamics.py "FB_s$s=$E,$OUT/frozen_fb_s$s/step-010000.pt,$P"
  $PY consequence_check.py "fb_s$s=$E,$OUT/frozen_fb_s$s/step-010000.pt,$P"
  $PY census_localization.py "fb_s$s=$E,$OUT/frozen_fb_s$s/step-010000.pt,$P"
  $PY erasure_diagnosis.py $(for k in 1000 2000 4000 10000; do echo "fb_s$s@$k=$E,$OUT/frozen_fb_s$s/step-$(printf %06d $k).pt"; done)
  EVAL_DEVICE=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $PY tierb_eval_v2.py "fb_s$s=$E,$OUT/frozen_fb_s$s/step-010000.pt,$P"
done
export EVAL_DEVICE=cpu CUDA_VISIBLE_DEVICES=
$PY location_dynamics.py --analyse FB_s1 FB_s2 FD_s1 FD_s2 rawpred_s1 rawpred_s2 J_full_s1 J_full_s2 \
  --pairs FB_s1:FD_s1,FB_s2:FD_s2,FB_s1:rawpred_s1,FB_s2:rawpred_s2 --tag stageB
$PY consequence_within_root.py rawpred_s1 rawpred_s2 full_s1 full_s2 mop_s1 mop_s2 fd_s1 fd_s2 fb_s1 fb_s2 D6k D12k D18k D36k D42k D45k
$PY stage2_analysis_v2.py --prefix-sp fb_s --prefix-raw fd_s --seeds 1,2 --tag _fb_vs_fd
$PY stage2_analysis_v2.py --prefix-sp fb_s --prefix-raw rawpred_s --seeds 1,2 --tag _fb_vs_raw
echo frozen_eval_B_done
