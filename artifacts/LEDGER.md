# Experiment ledger

What was run, why, on what data, and what came out. `artifacts/` holds 134 GB on
disk and ~670 tracked files: run outputs are gitignored and reproducible from the
script beside them, while scripts and small result JSONs are tracked.

**Run outputs are never moved.** The sealed M03 contract pins 538 absolute paths by
SHA256 (`historical_inputs`, `direct_anchors`, dataset, checkpoints). Relocating any
of them breaks run resume, `--reuse-from`, and the 4.4 GB dependency cache. Reorganize
by adding an entry here, not by moving bytes.

## Convention for new work

One directory per experiment under `artifacts/experiments/<YYYYMMDD>_<name>/`:

| File | Contents |
|---|---|
| `README.md` | Question, protocol, data identity, outcome, and what it does *not* establish |
| `<name>.py` | The runner. Imports machinery from `d4mj/`; never duplicates it |
| `evidence/` | Small extracted JSON results with source hashes. Large outputs stay gitignored |

Add a row to *Current* below when you start, and record the outcome when it lands.
Specifications live in [`d4mj/spec/`](../d4mj/spec/); results live here.

## Current

| Date | Experiment | Question | Status |
|---|---|---|---|
| 2026-09-05 | [m0_m3_validation](experiments/20260905_m0_m3_validation/) | Do the M0–M3 implementation contracts hold on the deployment device? | Passed; evidence only, no research result |
| 2026-09-06 | [lewm_paired](experiments/20260906_lewm_paired/) | Raw vs TC SIGReg, 10k joint updates, one seed | Complete. Budget finished; TC projection concern unresolved |
| 2026-09-09 | [m03_capability](experiments/20260909_m03_capability/) | Did the new architecture resolve the prior semantic failures? | Complete. `insufficient_coverage`; TC positive mechanism, negative recipe |
| 2026-09-10 | [feature_ladder](experiments/20260910_feature_ladder/) | Where in patch → CLS → z is the state information lost? | Complete. Pooling/export bottleneck; probe width ruled out; patch tokens recover +0.10 successor AUC |
| 2026-09-11 | [predictability_bridge](experiments/20260911_predictability_bridge/) | Can the frozen world predict a spatial stream from `h` or `[z,h]`? | Complete. Effect R² 0.71 from `h` alone, but only +0.02 semantic AUC over persistence; contract not viable as-is |
| 2026-09-15 | [patch_token_policy](experiments/20260915_patch_token_policy/) | Does token-preserving cross-attention over 4×4 patch tokens beat pooled export, under real expert BC? | Complete. Tokens beat pooling by +0.066 (raw) / +0.036 (tc) paired; TC beats Raw at every condition |
| 2026-09-15 | [frame_skip_retrain](experiments/20260915_frame_skip_retrain/) | Is TC's centering window physically too short? (paper trains at frame skip 4; we used stride 1) | Complete. TC rank 5.68 -> 16.73 at 10k and scale inflation gone; not cured (raw 34.09). Semantic panels blocked on the macro-fork |
| 2026-09-16 | [centering_window](experiments/20260916_centering_window/) | Was TC's recovery the centering window, or the 4-step horizon and stacked actions? | Complete. The window, entirely: rank 5.14 -> 19.16 (+14.02) vs +11.04 for the whole stride-4 change. M03: no capability gain. BC: projected z falls to the action prior (0.204 -> 0.142, floor 0.147) while patch tokens hold. Higher rank, unusable projection |
| 2026-09-17 | [generated_readout](experiments/20260917_generated_readout/) | Is usable information absent from generated latents, or is the observed-fit decoder failing on them? | Complete. Decoder failing: refit recovers +0.12-0.15 AUC in every arm, death/damage from below 0.5 to 0.68-0.79. But no advantage over a trained root+action predictor, and refitting does not rescue successor state: TC ~0.63 vs Direct 0.79, a gap already present on observed z |
| 2026-09-17 | [learned_export](experiments/20260917_learned_export/) | Is 192-D too few, or did joint training choose a poor mapping into it? | Complete. Width is not the bottleneck: matched 192-D patch export 0.763 vs current 0.618, and unsupervised patch PCA alone gets 0.717 on the short window. Level with Direct's comparable 0.757, not ahead. CLS is the lossy step |
| 2026-09-17 | [state_transition](experiments/20260917_state_transition/) | Can the world preserve and predict a richer 192-D state? | Complete. Yes: u->u reaches 0.705 generated / 0.746 hidden vs the matched z->z baseline's 0.648 / 0.658, and turns continuous R2 positive. Both richer input and richer target are needed; neither alone suffices |

## Historical campaigns

Predates this convention. Grouped by campaign; outputs are gitignored unless noted.
The per-experiment family mapping — which historical script answers which question,
and what M03 does or does not reproduce from it — is in
[`d4mj/m03/README.md`](../d4mj/m03/README.md), which is the authority for provenance.

| Campaign | Directories | Produced by | Role now |
|---|---|---|---|
| Support corpus | `craftax_support_v2/` (36 GB), `_logs/`, `craftax_support_v1.pt` | `collect_support_v2.py` | Active dataset. 3,179,062 transitions, expert ε-rollouts, BC-ineligible |
| Stage-A lattice | `stage_a*/`, `stage_a_olddesign/` | `run_stage_a.py` | `{flow,direct}×{attention,mamba}` baselines. Superseded by v2 Direct |
| EDA / Phase-1B | `eda/` (54 GB), `phase1b_*/` | `eda/*.py`, `run_phase1b_*.sh` | **M03 input.** Fork histories, successors, root frames, damage streams, `capacity6k` MAE encoder, `v2_direct_{attention,mamba}` anchors |
| Consequence & identifiability | `consequence_learnability*/`, `identifiability_gate_v2/`, `fatality_identifiability/`, `encoder_fatality_fidelity/`, `branched_coverage_gate/` | `benchmark_/evaluate_consequence_learnability.py`, `run_identifiability_gate_v2.py` | Established the consequence gap. Re-run under M03's common scoring |
| Counterfactual localization | `direct_transition_stages/`, `*_localization.json` | `localize_*.py` | Localized Direct's loss to the transition, not the encoder |
| Terminal supervision | `terminal_diversity_*/`, `stage_a_terminal_dynamics/` | `run_terminal_*.sh`, `train_terminal_diversity_scaling.py` | Closed 2026-08-29: terminal tails install an action-prior shortcut |
| Oracle & horizon | `oracle_horizon_h{2,16}/`, `oracle_phase3_*/`, `matched_context_h2/`, `hybrid_context_h2/` | `run_oracle_phase3.py` | Separated transition, outcome and critic error |
| Generated outcome & drift | `generated_latent_outcome_*/`, `generated_drift/`, `multiworld_drift/`, `multiworld_entropy/` | `run_generated_latent_outcome_shaping.py`, `run_r16_drift.sh` | Depth-16 KL/agreement rows cited in the LeWM evaluation spec |
| Ceilings | `paired_ceiling/`, `paired_ceiling_smoke/`, `rollout_ceiling/` | `collect_paired_trajectory_forks.py` | Decoder/rollout ceilings for interpreting failures |
| Flow arm | `predictor_flow_attribution/`, `equalized_flow_continuation/`, `frozen_continuation_ablation/` | `run_predictor_flow_attribution.py` | Flow-side controls; not part of the LeWM family |
| LeWM gates | `lewm_gates_20260906/` (20 GB) | `python -m d4mj paired-run`, `python -m d4mj.m03.gate` | **Current.** Paired joint run, M03 suite, shared dependency cache |

## Known repository issues

- `.git` is 7.4 GB, dominated by tracked `data/*.pt` bundles (74 MB + 69 MB × 3 + …).
  Shrinking it requires history rewriting and has not been done.
- Roughly 60 loose runner scripts sit at the `artifacts/` top level. They are tracked,
  referenced by name from the M03 provenance mapping, and by absolute path from sealed
  run contracts, so they have deliberately not been moved into subdirectories.
