# Ablation log

One row per run. `commit` is the commit that RECORDS the run, so the codebase
state is `git show <commit>`. `flags` mark known confounds that limit what the
row can be read as.

Flags: `EMA*` EMA tau ramped over the run budget, not 20k, so the run is NOT a
prefix of the 20k schedule · `INIT*` pre-D046, so T-vs-M shared init differed ·
`1SEED` one training seed · `SUBSET` reported on a 6-target subset, not all 12.

| # | date | ablation | artifacts | commit | outcome | flags |
|---|---|---|---|---|---|---|
| 1 | 07-27 | Craftax baseline T-JEPA + M-JEPA, 20k/3k/500 | `outputs/craftax_expert_v1` | `74f2c71` | runs at 761 MB; BC at majority floor; T/M indistinguishable | INIT* 1SEED |
| 2 | 07-27 | BC budget ladder 0.5k-30k | `craftax_bc_budget.json` | `74f2c71` | under-training real (+0.03) AND ceiling real (flat from 10k) | |
| 3 | 07-27 | Representation oracle, expert probe | `craftax_oracle.json` | `74f2c71` | 0/16 preserved (T), 1/16 (M); pixels recover HUD at R2~1.0 | |
| 4 | 07-27 | `d_bottleneck` 16/32/64 | `craftax_capacity`, `craftax_oracle_capacity.json` | `8c0307d` | no effect on preservation; channel width is not the cause | |
| 5 | 07-28 | `n_latents` 16/64/256 | `craftax_oracle_n_latents.json` | `a60d30e` | no effect; preserved stays 1/16 at 16x tokens | |
| 6 | 07-28 | Random-encoder floor | `craftax_oracle_random.json` | `4306c87` | untrained encoder BEATS trained on all but health; training REMOVES state | |
| 7 | 07-28 | T-BASE reconstruction control, n16 + n64 | `craftax_tbase`, `craftax_oracle_tbase{16,64}.json` | `8517c30`, `e3b9e78` | reconstruction retains far more; capacity helps it (+0.19) but not JEPA (+0.03) | |
| 8 | 07-28 | Latent spectrum / effective rank | `craftax_latent_rank.json` | `8517c30` | dimensional collapse REFUTED — training raises rank (4.0 -> 12.0 -> 30.3) | |
| 9 | 07-28 | Encoder time-course, 8 checkpoints | `craftax_timecourse.json` | `c26e162` | flat to step 500, then monotone decay; health rises as the rest falls | EMA* |
| 10 | 07-28 | `jepa_weight=0` (VOID) | — | `9df6b9c` | VOID: `cfg.jepa_weight` was a dead field; run reproduced baseline bit-for-bit | |
| 11 | 07-28 | Self-prediction off, heads only | `craftax_timecourse_jepaweight0.json` | `2ffdc85` | erosion persists and worsens; self-prediction is partially PROTECTIVE | EMA* |
| 12 | 07-28 | Sampler `terminal_fraction` 0.5 vs 0.0 | `craftax_sampler_termfrac{050,000}.json` | `52fe01d` | health spike is ENTIRELY the sampler; erosion only ~23% (12 targets) | EMA* SUBSET |
| 13 | 07-28 | 2x2 sampler x objective | `craftax_tf0{50,00}_jepaonly.json` | `202bcfb` | heads ~0 on inventory MEAN, but per-target effects cancel (mean-abs 0.113) | EMA* SUBSET |
| 14 | 07-28 | D045 predictor context 64 -> 384 | `craftax_spatialpred_*.json` | `c562e46` | REJECTED: +0.003 in the isolating cell | EMA* |
| 15 | 07-28 | Sampler distribution audit | `craftax_sampler_audit.json` | `cbb6f22` | 16.67% terminal targets after MTP, 61.5% BCE mass, reward label -0.0238 | |
| 16 | 07-29 | Encoder anchor: LR full/slow/frozen x 3 seeds | `reviews/artifacts/encoder_anchor` | `1c77766` | `enc_lr=6e-6` removes ~all erosion (0.661 -> 0.651 vs -> 0.465) | |
| 17 | 07-29 | Encoder-LR full pipeline + executed, T and M | `outputs/craftax_slowenc_v1`, `craftax_lr_experiment_summary.json` | `0d94c24` | slow-T actor **+2.042 [0.96, 2.62]** over own BC; slow-M **-0.262**, does not clear | INIT* 1SEED |
| 18 | 07-29 | SIGReg vs EMA, Craftax | `reviews/artifacts/lr_objective_grid` | `0d94c24` | SIGReg worse on every target; throwaway, NOT a Craftax rejection of D036 | INIT* 1SEED |
| 19 | 07-29 | Fixed-init paired T/M at `enc_lr=6e-6` | `outputs/craftax_fixedinit_slow` | `ba3ae1e` | **ABORTED** at T world 3k/20k; no result | |

## Executed control, current standing (run 17)

| arm | random | BC | actor | actor - BC (95% CI) |
|---|---:|---:|---:|---|
| T full | 1.640 | 2.606 | 2.481 | -0.124 [-0.862, +0.456] |
| T slow | 1.640 | 2.353 | **4.395** | **+2.042 [+0.960, +2.623]** |
| M full | 1.640 | 3.604 | 2.948 | -0.656 [-1.316, +0.120] |
| M slow | 1.640 | **3.554** | 3.292 | -0.262 [-1.177, +0.626] |

M has the stronger BC in both conditions and the weaker actor. All four carry
`INIT*`, so no T-vs-M row here is a single-axis measurement.

## Provenance note

Runs 1 and 17's `full_*` arms were evaluated with
`implementation_drift_allowed=true` against stored hash `c513c752`. Verified
benign: D045 is RNG-inert (237/237, 243/243 identical rebuild) and `jepa_weight`
multiplies by exactly 1.0.

## Open before the next comparison is interpretable

D059 (imagined context slides, so Mamba never accumulates state) · D046 applied
but no post-fix T/M run exists · D060 (no Craftax T-BASE policy baseline).
