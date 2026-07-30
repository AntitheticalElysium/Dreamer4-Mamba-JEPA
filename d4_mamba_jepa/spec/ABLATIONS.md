# Ablations

What we RAN. Past tense, append-only, never edited. One row per run. `commit`
is the commit that records the run, so the codebase state is `git show <commit>`.
`flags` mark confounds that limit what the row can be read as.

Rule: a row is never corrected. If a later run supersedes it, add a row. If a
defect invalidates it, add a flag to the flag list and mark the affected rows.

Flags: `EMA*` EMA tau ramped over the run budget, not 20k, so the run is NOT a
prefix of the 20k schedule · `INIT*` pre-`ba3ae1e`, so T-vs-M shared init
differed · `1SEED` one training seed · `SUBSET` reported on a 6-target subset ·
`WD*` the M arm decayed `dt_bias`/`A_log`/`D`, which official Mamba-2 marks
`_no_weight_decay` (`mamba2.py:130,136,140`). The T arm owns no such tensor, so
weight decay was a second uncontrolled difference on the research axis for these
rows. Fixed after `81d3466`; the T arm's optimizer is unchanged, so only M-arm
numbers are affected.

| # | date | ablation | artifacts | commit | outcome | flags |
|---|---|---|---|---|---|---|
| 1 | 07-27 | Craftax baseline T-JEPA + M-JEPA, 20k/3k/500 | `outputs/craftax_expert_v1` | `74f2c71` | runs at 761 MB; BC at majority floor; T/M indistinguishable | INIT* 1SEED WD* |
| 2 | 07-27 | BC budget ladder 0.5k-30k | `craftax_bc_budget.json` | `74f2c71` | plateaus at 0.183/0.190 vs 0.149 floor; under-training refuted | |
| 3 | 07-27 | Representation oracle, expert probe | `craftax_oracle.json` | `74f2c71` | 0/16 preserved (T), 1/16 (M); pixels recover HUD at R2~1.0 | |
| 4 | 07-27 | `d_bottleneck` 16/32/64 | `craftax_oracle_capacity.json` | `8c0307d` | no effect on preservation; channel width is not the cause | |
| 5 | 07-28 | `n_latents` 16/64/256 | `craftax_oracle_n_latents.json` | `a60d30e` | no effect; preserved stays 1/16 at 16x tokens | |
| 6 | 07-28 | Random-encoder floor | `craftax_oracle_random.json` | `4306c87` | untrained encoder beats trained on all but health | |
| 7 | 07-28 | T-BASE reconstruction control, n16 + n64 | `craftax_oracle_tbase{16,64}.json` | `8517c30`, `e3b9e78` | reconstruction retains far more; capacity helps it (+0.19) not JEPA (+0.03) | |
| 8 | 07-28 | Latent spectrum / effective rank | `craftax_latent_rank.json` | `8517c30` | dimensional collapse refuted; rank rises 4.0 -> 12.0 -> 30.3 | |
| 9 | 07-28 | Encoder time-course, 8 checkpoints | `craftax_timecourse.json` | `c26e162` | flat to step 500, then monotone decay; health rises as the rest falls | EMA* |
| 10 | 07-28 | `jepa_weight=0` | — | `9df6b9c` | VOID: field was dead; run reproduced baseline bit-for-bit | |
| 11 | 07-28 | Self-prediction off, heads only | `craftax_timecourse_jepaweight0.json` | `2ffdc85` | erosion persists and worsens; self-prediction is partially protective | EMA* |
| 12 | 07-28 | Sampler `terminal_fraction` 0.5 vs 0.0 | `craftax_sampler_termfrac{050,000}.json` | `52fe01d` | health spike is entirely the sampler; erosion ~23% | EMA* SUBSET |
| 13 | 07-28 | 2x2 sampler x objective | `craftax_tf0{50,00}_jepaonly.json` | `202bcfb` | heads ~0 on inventory mean; per-target effects cancel (mean-abs 0.113) | EMA* SUBSET |
| 14 | 07-28 | Predictor context 64 -> 384 | `craftax_spatialpred_*.json` | `c562e46` | rejected: +0.003 in the isolating cell | EMA* |
| 15 | 07-28 | Sampler distribution audit | `craftax_sampler_audit.json` | `cbb6f22` | 16.67% terminal targets after MTP, 61.5% BCE mass, reward label -0.0238 | |
| 16 | 07-29 | Encoder anchor: LR full/slow/frozen x 3 seeds | `reviews/artifacts/encoder_anchor` | `1c77766` | `enc_lr=6e-6` removes ~all erosion (0.661 -> 0.651 vs -> 0.465) | |
| 17 | 07-29 | Encoder-LR full pipeline + executed, T and M | `outputs/craftax_slowenc_v1`, `craftax_lr_experiment_summary.json` | `0d94c24` | slow-T actor +2.042 [0.96, 2.62] over own BC; slow-M -0.262 | INIT* 1SEED WD* |
| 18 | 07-29 | SIGReg vs EMA, Craftax | `reviews/artifacts/lr_objective_grid` | `0d94c24` | SIGReg worse on every target; throwaway, not a Craftax rejection | INIT* 1SEED WD* |
| 19 | 07-29 | Fixed-init paired T/M at `enc_lr=6e-6` | `outputs/craftax_fixedinit_slow` | `ba3ae1e` | ABORTED at T world 3k/20k; no result | |
