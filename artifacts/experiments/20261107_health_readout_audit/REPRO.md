# Commands

All commands run from repository root, using the project venv. Set `JAX_PLATFORMS=cpu` for any source imports;
the scripts set it before imports. No package installation. GPU commands were started only after confirming
Claude's GPU probe had completed. The CPU replay was left alone.

```sh
MAMBA_DETERMINISTIC=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m artifacts.experiments.20261107_health_readout_audit.audit collect --smoke
MAMBA_DETERMINISTIC=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m artifacts.experiments.20261107_health_readout_audit.audit collect
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m artifacts.experiments.20261107_health_readout_audit.audit fit
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m artifacts.experiments.20261107_health_readout_audit.analyse
MAMBA_DETERMINISTIC=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m artifacts.experiments.20261107_health_readout_audit.audit evaluate
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m artifacts.experiments.20261107_health_readout_audit.analyse --native
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m artifacts.experiments.20261107_health_readout_audit.nonlinear
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m artifacts.experiments.20261107_health_readout_audit.global_probe
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m artifacts.experiments.20261107_health_readout_audit.global_probe 100
```

Primary outputs: `STAGES.json`, `FIT.json`, `NATIVE.json`, `NONLINEAR.json`, `GLOBAL_PROBE*.json`.
Raw captures and encoded prefixes are retained in ignored `.pt` files.

Ancillary deterministic calculations in `RIDGE_SENSITIVITY.json`, `PURE_NOISE_REFIT.json`, `LOSS_SHARES.json`,
`DAMAGE_GEOMETRY.json` and `INSTRUMENT.json` use only `capture.pt`:

- ridge sensitivity repeats `audit.fit` at penalty multipliers 1e-6, 1e-4, 1e-2; evaluate each on its omitted fold;
- pure-noise refit applies the same weighted residual-ridge solve to `fine0/mod` or `native0/mod`, targeting `truth`,
  rather than the random-signal sufficient statistics; natural row weights, not ramp weights;
- loss shares use `(flow_out-flow_y)^2` times natural row weights times `.1+.9*flow_tau`, divided by the all-cell
  squared error reconstructed from `stats.shared` and `old_weight`;
- instrumentation compares `[flow_x,1] @ old_weight` with the captured bfloat16 `flow_out`;
- damage geometry tabulates `T.body[rows,0]-T.true_body[rows,0]` within each adjacent cohort's actual drop rows.

Limits: no refit is an original-P1 retrain; context is native lightly corrupted history, while P1 uses a distribution
of histories. The two-finer-step teacher check holds that cache fixed, unlike the training bootstrap's full chunk.
Classifier probabilities are not physical token-displacement targets. Extra CPU controls are labelled post-hoc
in PLAN.md. The C=100 and ridge sensitivities are checks, not used to select the native intervention.
