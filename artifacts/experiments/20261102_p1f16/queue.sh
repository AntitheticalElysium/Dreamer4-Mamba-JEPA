#!/bin/bash
# Same F1.5 readings; new outputs, F1.5 control, and F1.6 labels only.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
X=artifacts/experiments/20261102_p1f16
BASE=artifacts/experiments/20261101_p1f15/run/ckpt_10.pt
CK=$1
R=$X/readings
mkdir -p "$R" "$(dirname "$CK")/s1"
step() { echo "== $(date -u '+%Y-%m-%dT%H:%M:%SZ') $*"; }
step s1
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$PY" -u -m saccade.eval.s1 "$CK" "$(dirname "$CK")/s1/results.json"
step uptake_started_on_CPU
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 "$PY" -u -m artifacts.experiments.20261101_p1f15.uptake "$CK" "$R/uptake" > "$R/uptake.log" 2>&1 &
UPTAKE=$!
step interface_random
"$PY" -u -m artifacts.experiments.20261030_actor_interface.interface "$R/interface_random.json" --policy=random "p1f15=$BASE" "p1f16=$CK"
step interface_guided
"$PY" -u -m artifacts.experiments.20261030_actor_interface.interface "$R/interface_guided.json" --policy=guided "p1f15=$BASE" "p1f16=$CK"
step m05_evaluator
"$PY" -u -m artifacts.experiments.20261030_m05_guided_fovea.evaluate "$R/eval_m05" "base=$BASE" "p1f16_1500=$CK"
"$PY" -u -m artifacts.experiments.20261030_m05_guided_fovea.readings "$R/eval_m05/results.json" base p1f16_1500 > "$R/eval_m05/readings_p1f16_1500-base.txt"
CUDA_VISIBLE_DEVICES= "$PY" -u -m artifacts.experiments.20261030_m05_guided_fovea.dissect.token_geometry "$R/eval_m05" "$R/token_geometry.json" base p1f16_1500
step decoder_render
"$PY" -u -m artifacts.experiments.20261030_decoder_diagnosis.refit "$R/render_p1f16.json" "--ckpt=$CK" --trained-only
step codex_panel
"$PY" -u -m artifacts.experiments.20261102_p1f16.readings_codex panel "$CK"
step codex_cpu
CUDA_VISIBLE_DEVICES= "$PY" -u -m artifacts.experiments.20261102_p1f16.readings_codex cpu "$CK"
step codex_sampler
"$PY" -u -m artifacts.experiments.20261102_p1f16.readings_codex sampler "$CK"
step consistency_and_conditioning
"$PY" -u -m artifacts.experiments.20261101_p1f15.consistency "$CK" "$R/consistency"
step locus
F=artifacts/experiments/20261031_learned_fixation_sweep
"$PY" -u -m artifacts.experiments.20261031_learned_fixation_sweep.locus "$R/locus" "--ckpt=p1f16=$CK"
for w in w1_10 p1f_10; do ln -sfn "$(pwd)/$F/locus/$w.npz" "$R/locus/$w.npz"; done
ln -sfn "$(pwd)/artifacts/experiments/20261101_p1f15/readings/locus/p1f15.npz" "$R/locus/p1f15.npz"
"$PY" -u -m artifacts.experiments.20261031_learned_fixation_sweep.locus --summary "$R/locus"
step paired_codex_summary
CUDA_VISIBLE_DEVICES= "$PY" -u -m artifacts.experiments.20261102_p1f16.readings_summary
step waiting_for_uptake
wait "$UPTAKE"
step done
