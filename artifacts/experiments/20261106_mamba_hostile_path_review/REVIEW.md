# Review of `20261106_mamba_hostile_path` (Codex, 11-06): fine hostile evidence through Mamba

Amended 11-06 after Codex's reply (`../20261106_determinant_review/REVIEW.md` 2): claims narrowed in sections 1-3 and 5.

Reviewer: Claude, 11-06, at the user's request. Everything below re-reads Codex's SAVED outputs (`features.pt`,
`bank.pt`, `latents/`, `native_results.json`); nothing was regenerated. Scripts in this folder; Codex's folder untouched.

## Verified as reported

- Mechanics are sound: bitwise prefix-repeat and same-input shams, batch-row invariance, frozen-parameter check, pins,
  preserved failed smoke, random-label control tasks (Hewitt & Liang), the reader-scope bug found and corrected
  without regenerating. The fine-only contrast really is fine-only (p, v, fixation, history, action equal).
- The native numbers reproduce from `native_results.json` (death: empty 0.275%, fine 0.275%, coarse 3.95%, both
  4.18%, natural both 8.77%).
- Independent cross-check: natural-both death 8.77% on these constructed scenes matches F3's own dev reading for
  adjacent-zombie, health <= 2 histories at 16 steps (risky death 0.088, `../20261105_p1f3/RESULTS.md` 3). The
  instrument agrees with a separate population.

## 1. Paired, the tested distinction survives; read from one state, accessibility still deteriorates

`paired.py` (`paired.json`): every background has all seven inputs, so the within-background difference
memory(zombie fine) - memory(empty) removes the background variation an unpaired reader must fight.

| stage | fine-only: |paired diff| / |background spread| | coarse-only | fine-only zombie vs cow, held-out backgrounds (paired linear) |
|---|---|---|---|
| raw fine tokens | 0.70 | (0, unchanged) | 100% |
| percept embedding (fusion + 6,569 -> 256) | **0.039** | 0.54 | 100% |
| belief (after Mamba) | 0.024 | 0.60 | 100% |
| memory tokens | 0.029 | 0.74 | 100% |

The fine-only zombie direction is consistent across backgrounds (the held-out difference projected on the fit-mean
direction ranks zombie above cow in 100% of held-out backgrounds at every stage). The tested distinction is not erased (this
does not show that all fine detail survives). What changes is
SCALE: at the embedding the fine signal is 14x smaller than the coarse one, then Mamba shrinks it a further ~1.8x
relative to coarse (fine/coarse ratio 0.073 -> 0.040 -> 0.039). Codex's unpaired readers (600 points in 1,024
dimensions) cannot find a direction at 3% of background variance; they also read the STRONG coarse signal at only
76-79% (3-class). "Both MLPs fit their training labels 100%" shows capacity to overfit 600 points, not reader
adequacy. Caveat: a paired reader uses the counterfactual reference, which the deployed model never has; it measures
information and scale, not usability.

## 2. The largest amplitude attenuation is at the percept embedding, before Mamba (not a unique culprit)

Codex: "we still cannot identify fusion, projection, Mamba, or downstream learned use as the unique cause".
`features.pt` already held the percept embedding, and `fusion_split.py` (`fusion_split.json`, CPU, F3 on the bank's
renders) splits it exactly, since d embed = W_canvas d canvas when only g changes:

| path | change in | change out | gain |
|---|---|---|---|
| fine: SpatialFusion | raw fine 5.57 | fused 1.28 | 0.23 |
| fine: projection's canvas block | fused 1.28 | embedding 0.38 | 0.29 |
| coarse: projection's periphery block (plus its own fusion) | raw coarse 5.45 | embedding 5.27 | ~0.97 |

The coarse token reaches the projection twice (directly, 1,296 columns, and through the fusion); the fine token only
through the fusion. Fusion compresses fine and coarse changes alike (fused change 1.28 vs 1.27), so it is not
specifically discounting fine; the asymmetry is the coarse token's direct, unattenuated path. The largest AMPLITUDE
attenuation (14x) is at the percept embedding (fusion 4.4x, projection 3.4x), and Mamba adds ~1.8x. Norms in different
learned spaces are not lost information or native importance, and Codex's matched binary single-state readers keep
falling after the embedding (zombie vs cow: fine 98%, embedding 81%, belief 61%, memory 56%; paired 100% throughout):
the embedding is not identified as the unique cause, and no repair there is justified by this.

## 3. The danger response is drawn in the frame (my head-only hypothesis refuted); memory draws consequences, the cache draws entities

The outcome head reads `features(after-action belief, embed(generated percept))`, and under memory-only evidence the
generated zombie is almost never drawn (0.625% coarse vs 0.475% empty, 77.2% with the zombie in the cache). I
suspected the 3.95% death was the head reading the belief. `frame_reads.py` (`frame_reads.json`) read the health digit
of all 24,000 saved futures with the determinant-curves health reader (100% on 34,059 dev frames; 100% agreement with
the bank's simulator successors, dead 98.4% / 0 / 0 as recorded):

| condition | frame dead (health 0) | head P(death) | frame damaged | head P(reward < -0.05) |
|---|---|---|---|---|
| empty | 0.28% | 0.27% | 0.53% | 0.55% |
| fine only | 0.28% | 0.27% | 0.55% | 0.59% |
| coarse only | 3.98% | 3.95% | 5.30% | 5.49% |
| both (memory) | 4.20% | 4.18% | 5.60% | 5.74% |
| natural both (cache too) | 8.78% | 8.77% | 12.4% | 12.9% |

The head's death mass sits entirely on frames that show health 0 (share on alive frames < 1e-6 in every condition).
So the native effects are WORLD-DRAWN and Codex's reading stands. The structural point is new: memory-only zombie
evidence makes the world draw the CONSEQUENCE in the HUD (health -> 0, at about half the natural rate) while almost
never drawing the ENTITY. On this constructed bank the two routes behave differently: the cache version draws the
entity, the memory version draws the HUD consequence (a route behaviour on this bank, not an architecture-wide law).

## 4. What the experiment can and cannot say about the user's question

The question was whether fine information is lost, "and/or the model doesn't use it because the periphery acuity
makes it redundant". The bank is daytime (light >= .8), adjacent zombies: the regime where 3 x 3 coarse tokens already
carry the zombie (11-02 acuity EDA: coarse carries nearly all task facts; fine adds value for night creatures and fine
HUD detail). Under the redundancy hypothesis the trained model SHOULD weigh fine little here, so the fine-only null is
predicted by both "route broken" and "fine rationally ignored". Codex flags the out-of-distribution conflict; the
stronger point is that this cue-conflict design cannot separate the two hypotheses at all. A separating test needs
content that is not coarse-available in natural data (night creatures, where 2 x 2 already loses recall), so that
redundancy predicts USE and a broken route predicts none.

## 5. Relation to our determinant work

- The consequences we are diagnosing (wake, counts, health) have COARSE-available determinants (HUD values exact
  at 3 x 3); the fine route is not on their path, so this result does not explain them.
- But the same stage is on their memory route: every percept fact reaches the belief through the 6,569 -> 256
  projection. The determinant-routes experiment measures the same paired scale profile (token -> embedding -> belief
  -> memory, and the denoiser's cache) for the energy, health and count digits.
- The route behaviour of section 3 (on that bank) is in line with the determinant curves
  (`../20261106_determinant_curves/RESULTS.md`): map-local consequences are drawn (mined cell cleared 0.92-0.99), HUD
  consequences fail (counts, food after a cow, damage arithmetic), and the coal event is fully in the belief (trace B1
  AP 1.0) while the coal slot copies. The memory -> HUD mapping is where our failures concentrate; the third
  experiment's cache x memory factorial tests it on our determinants directly.
- Head versus world: settled the same way in both studies. The heads read what the frame draws (wakes, death,
  damage all WORLD-DRAWN there; death here).

Reproduce (repo root): `PYTHONPATH=$PWD:/tmp/saccade-mamba-reader-deps .venv/bin/python
artifacts/experiments/20261106_mamba_hostile_path_review/paired.py` (CPU); `PYTHONPATH=$PWD:$PWD/artifacts/eda
.venv/bin/python artifacts/experiments/20261106_mamba_hostile_path_review/fusion_split.py` (CPU);
`MAMBA_DETERMINISTIC=1 .venv/bin/python -m artifacts.experiments.20261106_mamba_hostile_path_review.frame_reads` (GPU).

Sources: Elazar et al. 2021 (TACL, amnesic probing), Hewitt & Liang 2019 (control tasks), cited by Codex; Heimersheim &
Nanda 2024 (arXiv 2404.15255) on reading interventions as sufficiency vs necessity.
