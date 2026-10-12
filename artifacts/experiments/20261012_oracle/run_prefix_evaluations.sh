#!/usr/bin/env bash
# Separate evaluator unit: wait for training completion, then GPU evaluation,
# then CPU readout controls. Every stage resumes from source-bound artifacts.
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PREFIX_RUN=artifacts/eda/oracle_v1/generated_prefix
PREFIX_LOG=artifacts/eda/levers_logs/generated_prefix_eval.log
PREFIX_TRAINER=oracle_generated_prefix_20261012.service
{
    date '+START %Y-%m-%d %H:%M:%S %Z'
    # Both final snapshots are written before trainer success. Do not compete
    # with the trainer for GPU compute during this matched comparison.
    while [[ ! -f "$PREFIX_RUN/teacher_u3000.pt" || ! -f "$PREFIX_RUN/prefix_u3000.pt" ]]; do
        if ! systemctl --user is-active --quiet "$PREFIX_TRAINER"; then
            printf 'FAILED: trainer stopped before both final snapshots\n'
            exit 1
        fi
        sleep 10
    done
    while systemctl --user is-active --quiet "$PREFIX_TRAINER"; do
        sleep 10
    done
    for update in 0 500 1500 3000; do
        if [[ "$update" == 0 ]]; then
            # Initial weights match; checking the two inferred arms here also
            # proves source/checkpoint inference equality at the true protocol.
            .venv/bin/python artifacts/experiments/20261012_oracle/prefix_eval.py \
                --updates "$update" --prefix generated_prefix --arms teacher prefix
        else
            .venv/bin/python artifacts/experiments/20261012_oracle/prefix_eval.py \
                --updates "$update" --prefix generated_prefix
        fi
    done
    .venv/bin/python artifacts/experiments/20261012_oracle/prefix_readout_refit.py \
        --prefix generated_prefix --updates 0 1500 3000
    .venv/bin/python artifacts/experiments/20261012_oracle/prefix_readout_calibration.py \
        --prefix generated_prefix --updates 0 1500 3000
    .venv/bin/python artifacts/experiments/20261012_oracle/prefix_readout_spatial.py \
        --prefix generated_prefix --updates 0 1500 3000
    .venv/bin/python artifacts/experiments/20261012_oracle/prefix_readout_compare.py \
        --prefix generated_prefix --updates 1500 3000
    date '+SUCCESS %Y-%m-%d %H:%M:%S %Z'
} >> "$PREFIX_LOG" 2>&1
