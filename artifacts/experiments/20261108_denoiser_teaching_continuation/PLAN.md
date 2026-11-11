# Finish the interrupted denoiser-teaching analysis

2026-11-08. User asked to continue Claude's unfinished work and answer the three foundational questions. Preserve
its original results and code. No training changes or full world run. No other-agent jobs are active at inspection.

1. Review design record and original discussion to distinguish next-percept prediction from a separate Mamba-only
   next-embedding loss. Check why generated percepts condition outcome models.
2. Audit completed M1-M3 before interpreting them. Magnitude is not causal teaching content; state-gradient
   consistency is not parameter-update agreement; nonzero reset gradients are not a proof of a working timer.
3. Complete Claude's declared M4 correction in an isolated script: use |delta HUD token| and |delta percept_embed|
   instead of signed differences. Same 40k DEV rows (seed2120), same five episode folds, same logistic protocol.
   Reuse the existing source-bound F4 encoded DEV trajectories from the readout audit; recompute only percept_embed
   at the actual checkpoint. Reproduce the original signed-health control before accepting interpretations.
4. Report original valid zero-value probes without presenting them as new fits. If an absolute-difference HUD
   control remains below .95 AP, flag that reader limit rather than attribute its shortfall to missing information.
   Even qualified controls establish comparative accessibility, not irreversible destruction or causal relevance.

No automatic followup. The goal is to discuss a coherent training/state contract, not add heads by default.
