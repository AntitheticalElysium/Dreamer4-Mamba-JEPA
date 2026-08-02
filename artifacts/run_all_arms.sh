#!/bin/bash
# One process per arm. Mamba's Triton autotuner needs contiguous free memory to
# benchmark its backward kernels, and a long-running process fragments the
# allocator across arms; a fresh process per arm removes that coupling entirely.
# The driver skips arms already in report.json, so this is safe to re-run.
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in 1 2 3 4; do
  echo "=== launching pass $i ==="
  .venv/bin/python artifacts/run_stage_a.py \
    --expert 96 --tokenizer-steps 3000 --dynamics-steps 5000 --agent-steps 2500 \
    --actor-steps 800 --eval-episodes 16 --eval-limit 800 \
    --out artifacts/stage_a || echo "pass $i exited nonzero"
done
echo "ALL PASSES FINISHED"
