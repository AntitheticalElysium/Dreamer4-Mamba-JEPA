#!/bin/bash
# The declared readings (PLAN.md "Readings"), in order, one GPU job at a time, after the run's clean exit.
# P1-F's battery: S1 on this run (P1-F's run/s1 exists, same code); the actor interface under random and guided gaze
# and M0.5's evaluator with P1-F re-run beside this run (same rows and draws; P1-F's re-run reproduces its 10-31
# numbers or the code drifted); the decoder render of this run (P1-F's render_p1f.json, same code). M0.5's evaluator
# takes differences against the label `base`, so there base = P1-F checkpoint 10 and beta_1500 = this run (the suffix
# also selects its belief / entity readings). Then Codex's readings (readings_codex.py).
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
X=artifacts/experiments/20261101_beta_tsp
P1F=artifacts/experiments/20261030_p1_guided/run/ckpt_10.pt
CK=$X/run/ckpt_10.pt
R=$X/readings
mkdir -p "$R" "$X/run/s1"
step() { echo "== $(date '+%m-%d %H:%M:%S') $*"; }
step s1
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 $PY -u -m saccade.eval.s1 $CK $X/run/s1/results.json
step interface random
$PY -u -m artifacts.experiments.20261030_actor_interface.interface $R/interface_random.json --policy=random p1f=$P1F beta=$CK
step interface guided
$PY -u -m artifacts.experiments.20261030_actor_interface.interface $R/interface_guided.json --policy=guided p1f=$P1F beta=$CK
step m05 evaluator
$PY -u -m artifacts.experiments.20261030_m05_guided_fovea.evaluate $R/eval_m05 base=$P1F beta_1500=$CK
$PY -u -m artifacts.experiments.20261030_m05_guided_fovea.readings $R/eval_m05/results.json base beta_1500 > $R/eval_m05/readings_beta_1500-base.txt
CUDA_VISIBLE_DEVICES= $PY -u -m artifacts.experiments.20261030_m05_guided_fovea.dissect.token_geometry $R/eval_m05 $R/token_geometry.json base beta_1500
step decoder render
$PY -u -m artifacts.experiments.20261030_decoder_diagnosis.refit $R/render_beta.json --ckpt=$CK --trained-only
for w in p1f10 beta10; do
  step codex panel $w
  $PY -u -m artifacts.experiments.20261101_beta_tsp.readings_codex panel $w
  step codex cpu $w
  CUDA_VISIBLE_DEVICES= $PY -u -m artifacts.experiments.20261101_beta_tsp.readings_codex cpu $w
done
step codex sampler
$PY -u -m artifacts.experiments.20261101_beta_tsp.readings_codex sampler
step done
