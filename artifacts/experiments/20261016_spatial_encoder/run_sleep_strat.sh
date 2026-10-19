#!/usr/bin/env bash
# Amendment 6: sleep-stratified, level-separated evaluation on CPU. Raw / MoP + SIGReg after chain_v2 has finished (never alongside
# another CPU evaluation), the full arm after run_stage2_full.sh has finished; then the readings.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
OUT="$(cd ../../eda && pwd)/spatial_encoder_v1"
export EVAL_DEVICE=cpu OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
while systemctl --user is-active -q spatial-chain-v2; do sleep 120; done
args=()
for s in 1 2; do
  args+=("raw_s$s=raw@10000,$OUT/stage2_rawpred_s$s/step-010000.pt,$OUT/probes2_raw10000.pt")
  args+=("mop_s$s=spatial:$OUT/stage2_mop_sig_s$s/step-010000.pt,$OUT/stage2_mop_sig_s$s/step-010000.pt,$OUT/probes2_spatial_s$s.pt")
done
$PY sleep_strat_eval.py "${args[@]}"
$PY sleep_strat_analysis.py --seeds 1,2
while systemctl --user is-active -q spatial-stage2-full; do sleep 120; done
$PY sleep_strat_eval.py \
  full_s1=spatial:$OUT/stage2_full_sig_s1/step-010000.pt,$OUT/stage2_full_sig_s1/step-010000.pt,$OUT/probes2_full_s1.pt \
  full_s2=spatial:$OUT/stage2_full_sig_s2/step-010000.pt,$OUT/stage2_full_sig_s2/step-010000.pt,$OUT/probes2_full_s2.pt
$PY sleep_strat_analysis.py --seeds 1,2
echo sleep_strat_done
