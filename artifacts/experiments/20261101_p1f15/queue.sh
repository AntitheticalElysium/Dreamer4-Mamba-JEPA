#!/bin/bash
# P1-F1.5's fixed readings (PLAN.md "Readings", items 1-6; 6 added 11-02), after the run's clean exit, on the checkpoint chosen by
# P1-F's rule (argument). GPU jobs one at a time; reading 5 (uptake, CPU) starts once S1 has written its reader and
# runs beside them. P1-F is re-run beside this world in the interface and M0.5's evaluator (same rows and draws), as
# for beta-TSP; its Codex panels and sampler factorial are beta-TSP's (same code, contexts asserted). M0.5's evaluator
# reads belief / entity panels only for labels ending in _1500 (beta-TSP's convention): this world is p1f15_1500.
# Usage: bash artifacts/experiments/20261101_p1f15/queue.sh artifacts/experiments/20261101_p1f15/run/ckpt_10.pt
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
X=artifacts/experiments/20261101_p1f15
P1F=artifacts/experiments/20261030_p1_guided/run/ckpt_10.pt
CK=$1
R=$X/readings
mkdir -p "$R" "$(dirname "$CK")/s1"
step() { echo "== $(date '+%m-%d %H:%M:%S') $*"; }
step s1
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 $PY -u -m saccade.eval.s1 $CK $(dirname "$CK")/s1/results.json
step uptake started on CPU
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 $PY -u -m artifacts.experiments.20261101_p1f15.uptake $CK $R/uptake > $R/uptake.log 2>&1 &
UPTAKE=$!
step interface random
$PY -u -m artifacts.experiments.20261030_actor_interface.interface $R/interface_random.json --policy=random p1f=$P1F p1f15=$CK
step interface guided
$PY -u -m artifacts.experiments.20261030_actor_interface.interface $R/interface_guided.json --policy=guided p1f=$P1F p1f15=$CK
step m05 evaluator
$PY -u -m artifacts.experiments.20261030_m05_guided_fovea.evaluate $R/eval_m05 base=$P1F p1f15_1500=$CK
$PY -u -m artifacts.experiments.20261030_m05_guided_fovea.readings $R/eval_m05/results.json base p1f15_1500 > $R/eval_m05/readings_p1f15_1500-base.txt
CUDA_VISIBLE_DEVICES= $PY -u -m artifacts.experiments.20261030_m05_guided_fovea.dissect.token_geometry $R/eval_m05 $R/token_geometry.json base p1f15_1500
step decoder render
$PY -u -m artifacts.experiments.20261030_decoder_diagnosis.refit $R/render_p1f15.json --ckpt=$CK --trained-only
step codex panel
$PY -u -m artifacts.experiments.20261101_p1f15.readings_codex panel $CK
step codex cpu
CUDA_VISIBLE_DEVICES= $PY -u -m artifacts.experiments.20261101_p1f15.readings_codex cpu $CK
step codex sampler
$PY -u -m artifacts.experiments.20261101_p1f15.readings_codex sampler $CK
step consistency
$PY -u -m artifacts.experiments.20261101_p1f15.consistency $CK $R/consistency
step F re-read with surprise
F=artifacts/experiments/20261031_learned_fixation_sweep
$PY -u -m artifacts.experiments.20261031_learned_fixation_sweep.locus $R/locus --ckpt=p1f15=$CK
for w in w1_10 p1f_10; do ln -sfn "$(pwd)/$F/locus/$w.npz" $R/locus/$w.npz; done
$PY -u -m artifacts.experiments.20261031_learned_fixation_sweep.locus --summary $R/locus
step waiting for uptake
wait $UPTAKE
step done
