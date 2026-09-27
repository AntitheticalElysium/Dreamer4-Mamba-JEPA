# Lab notebook — LeWM + Mamba on Craftax-Classic

The single index of what was run, where, with which parameters, and what it established. Each entry points to
the script and evidence file that hold the exact numbers. Newest campaign first. Corrections are kept, not
deleted; each names the claim it retires.

## Conventions

**Data**
- Corpus: `artifacts/craftax_expert_store_v1` + `artifacts/craftax_support_v2`, contract in
  `artifacts/lewm_m4_canonical/raw/dataset.json`. 10,400 episodes: 8,325 TRAIN / 1,041 DEV / 1,034 FINAL.
- Latent cache: `lewm_m4_canonical/raw/cache` (eval-mode z of the Raw joint encoder).
- Pools:
  - `artifacts/eda/interface_pool_v1`: u, z; 24,576 main + 8,071 terminal 6-frame windows, TRAIN.
  - `spatial_pool_v1`: Raw layer-normed tokens.
  - `spatial_pool_tc_v1`: the same windows, TC encoder.

**Judgement blocks**
- `artifacts/eda/observe_fresh_v4..v12` = seeds 53k..61k. Each root has frames, visible/hidden simulator state,
  and 32-key P(death) for all 17 actions.
- 55k-58k are opened (post hoc use only). 59k, 60k and 61k were each read once, sealed.
- Diagnostic futures: `artifacts/eda/diagnosis_futures_v1`, seeds 900,000+, 1,002 roots.
  - Per root: 4 context frames, all 17 actions x 4 keys one step ahead, and the policy's 16-step future x 5 keys.
  - Diagnosis split: 70/30 by seed, `randperm(seed 0)`.

**Models**
- Encoders: Raw and TC joint step-10,000 (`lewm_m4_canonical/{raw,tc}/joint`). The bridge's encoder equals
  Raw joint's.
- Canonical world: Mamba-2 (6 layers, width 256, d_state 64, headdim 64, 4 heads).
- Joint recipe: 4-frame windows, batch 128, AdamW 5e-5, wd 1e-3, 500 warmup, cosine to 5e-6, 10,000 updates,
  pred MSE + 0.09 SIGReg (1,024 projections, 17 knots).

| world | architecture and state | trained by |
|---|---|---|
| H2 | canonical Mamba, raw z | joint 10k + bridge 2k |
| Z, U, W | canonical Mamba; z, u (PCA-192 of the 4x4 patch grid), w (u / std) | `interface.py` |
| sZ, T | block-causal transformer (`spatial.World`); LN z, or 81 LN patch tokens | `spatial.py` |

**Metrics**
- Expected safe = 1 - P(death1 | chosen action).
- "x copy" = squared error / error of copying the previous frame; < 1 beats copying.
- "/ V" = squared error / total natural variance of that state.
- Captured = 1 - error / predictable change.
- Facts = closed-form ridge probes fitted on TRUE states, read on imagined ones.

## Canonical pipeline status (2026-09-27)
- Raw: joint done; H2 bridge trained; **H2 gate failed** (5/7 checks).
  - The checks that fail are 1 broken, 1 borderline and 3 unmeasurable at their sample size (`raw/gates/h2`).
- TC: joint done, never bridged. It stopped on a cache tolerance: 1.9e-5 vs an allowed 1e-5.
- **Bugs found in `d4mj`:**
  1. H2 terminal depth alias. Fixed on branch `h2-terminal-depth` (9e42b7a2), not merged.
  2. Context length (2026-09-27; see below). Not yet fixed.
- The actor has never been trained on LeWM.

---

## 2026-09-27 — Levers campaign (`20260927_levers/`) — IN PROGRESS

Questions from the user:
- (b) Can training the Mamba at its deployment length fix it, and is its memory used?
- Does discretization help?
- TC + T?
- Delta/copy variants?

| id | question | script | key parameters | status / result |
|---|---|---|---|---|
| E1 | training length L | `context_length.py` | frozen Raw latents; fresh canonical Mamba; L = 4/16/64, B = 128/26/6 (~384 transitions/update); joint optimizer, 10k updates, no SIGReg | done (see below) |
| E1b | continue L4 at L16/L64 (Stuffed Mamba's remedy); L16 at 3x budget | `context_cont.py` | E1 recipe; continuation 5,000 updates, fresh AdamW; L16x3 30,000 updates | done |
| E1c | what the memory helps predict | `memory_use.py` | 256 held-out 128-frame windows; benefit by transition type | done |
| E1d | L64 at 24 windows/update (length or diversity?) | `context_cont.py` arm L64b24 | 10,000 updates, 1,512 transitions/update (DRAMA: 2,048) | done: diversity, not length |
| E1e | matched compute for E1d | `context_cont.py` arms L4b512, L16b100 | same 1,500-1,536 transitions/update | done |
| E1f | canonical shape (4-frame world, then long continuation, as the H2 bridge) with E1d's diversity | `context_cont.py` arm L4to64b24 | continue L4 at L=64, 24 windows/update, 5,000 updates | done |
| E2 | discretization | `codebook.py`, `tworld.py --head categorical`, `teval.py --snap` | k-means K = 1024/4096 on LN tokens; categorical CE, teacher-forced; snap = nearest code after each imagined step | codebooks done; arms queued |
| E3 | TC + T | `tc_pool.py`, `tc_equiv.py`, `tworld.py --pool tc` | identical windows re-encoded by TC; representational tests; residual/direct T on TC vs Raw tokens | representation done; arms queued |
| E4 | Delta-JEPA LDAD in joint | `ldad_joint.py`, `ldad_eval.py` | canonical loop + CE(D(z_{t+1}-z_t), a_t), lambda 10 (and raw lambda 1), MLP 192-256-17; paired init verified | running (lane B) |
| E5 | copy / Delta variants on T | `tworld.py --head {direct,residual,gated,corr}` | spatial.World backbone; 6,000 updates, batch 40, AdamW 1e-4 wd 0.01, 1,000 warmup; L1 teacher + depth-2 suffix | queued (`queue_tworlds.sh`) |

**E1 so far (`context_length.json`), DEV/FINAL 256-frame windows:**

| world | 4-7 | 8-15 | 16-31 | 32-63 | 64-127 | 128-255 |
|---|---|---|---|---|---|---|
| L4, full history (x copy) | 1.51 | 3.24 | 3.78 | 4.92 | 7.45 | 9.27 |
| L4, 3-frame window (x copy) | 0.61 | 0.88 | 0.69 | 0.73 | 0.82 | 0.86 |
| L16, full history (x copy) | 0.76 | 1.06 | 0.88 | 0.96 | 1.27 | 1.69 |
| L16 memory benefit | +11% | +10% | +10% | +6% | -6% | -35% |
| L64, full history (x copy) | 1.20 | 1.57 | 1.30 | 1.36 | 1.68 | 1.82 |
| L64 memory benefit | +18% | +18% | +20% | +20% | +20% | +19% |

- Memory benefit = 1 - err(full history) / err(3-frame window).
- Imagination after a 48-frame context, 16 steps (/ V): L4 window 0.48, L16 recurrent 0.68, L64 recurrent 0.98,
  copy 0.55.
- Reading: training length decides whether the recurrence is usable past it.
  - L4: collapses beyond its window.
  - L16: memory helps up to about 4x its length, then fails.
  - L64: memory helps (+18-20%) at every position to 255.
- But at equal transitions per update, the longer the training window, the less accurate the world. L64 is
  1.2-1.8x copy, against L4-with-window 0.61-0.86x. It sees 6 episodes per update vs 128.
- Revisits (next view closer to a frame 4+ back than to the current one) are 0.9% of transitions.
- E1b and E1c test continuation and what the memory is used for.

**E1b so far (`context_cont.json`), continuation of the L4 world:**

| arm | 4-7 | 16-31 | 64-127 | 128-255 | memory benefit 64-127 | imagine 16 (/V) |
|---|---|---|---|---|---|---|
| L4to16, full history (x copy) | 0.66 | 0.77 | 1.43 | 2.08 | -47% | 0.74 |
| L4to64, full history (x copy) | 1.05 | 1.20 | 1.58 | 1.68 | +7% | 0.91 |

| L16x3 (fresh, 30k updates), full history | 0.58 | 0.65 | 0.96 | 1.32 | -11% | 0.54 |

- Continuing destroys the L4 world's short-window quality: its 3-frame-window error rises from 0.61-0.86 to
  0.71-1.79.
- L16 at 3x budget matches or beats L4-with-window inside 4-63 (0.58-0.86 vs 0.61-0.88 x copy). Its memory helps
  +7-8% to position 31 and +3% to 63, then fails beyond 64. L16's E1 gap was largely under-training.

**E1d, L64 with 24 windows per update (`context_cont.json`):**
- Full history, x copy: 0.65 / 0.91 / 0.69 / 0.73 / 0.84 / 0.87 at positions 4-7 / 8-15 / 16-31 / 32-63 /
  64-127 / 128-255. Never worse than copying, out to 4x its training length.
- Memory benefit +24-29% at every position; by type +22% moved, +37% blocked, +42% sleep onset,
  +32% exact revisits.
- 16-step imagination after 48 frames: 0.447 (/V). The best Mamba world so far (L4 windowed 0.48, copy 0.55);
  its own 3-frame window gives 0.726, so the memory is what helps.
- E1's L64 gap was diversity (6 windows per update), not length. It saw 4x the transitions per update of
  L4/L16, so E1e runs the matched-compute controls.

**E1e / E1f, matched compute (~1,500 transitions per update):**
- Error x copy: full history / own 3-frame window, at positions 4-7, 16-31, 64-127, 128-255.
- Imagination: 16 steps after a 48-frame context (/V); copy = 0.553.

| arm | 4-7 | 16-31 | 64-127 | 128-255 | memory 16-31 / 128-255 | imagine 16 (full / window3) |
|---|---|---|---|---|---|---|
| L4b512 | 1.74 / **0.52** | 4.68 / **0.59** | 9.14 / **0.69** | 11.35 / **0.72** | -699% / -1487% | 2.076 / 0.406 |
| L16b100 | 0.56 / 0.63 | 0.63 / 0.70 | 0.91 / 0.85 | 1.32 / 0.89 | +11% / -48% | 0.426 / 0.524 |
| L64b24 | 0.65 / 0.85 | 0.69 / 0.94 | 0.84 / 1.18 | 0.87 / 1.23 | +26% / +29% | 0.447 / 0.726 |
| L4to64b24 | 0.60 / 0.72 | 0.65 / 0.78 | 0.78 / 0.97 | 0.82 / 1.03 | +18% / +20% | **0.400** / 0.650 |

- At matched compute the windowed L4 world is the most accurate one step ahead at every position; E1d's win was
  partly 4x data.
- For imagination, (b) ties (a): L4to64b24 full 0.400 vs L4b512 window 0.406. (b) keeps a memory worth +18-20%
  over its own window.
- The best (b) is the canonical shape: a 4-frame joint world continued at L=64 with 24 windows per update. The
  H2 bridge continues too, but with 16 windows per update, heads and 2,000 updates, and its world ended at
  1.4-2.6x copy.
- Recall is 0.19% of transitions, so next-frame accuracy cannot show memory's value for long-horizon play.

**E1c (`memory_use.json`)**, 256 held-out 128-frame windows, positions >= 16. Memory benefit by transition type:

| world | moved | blocked | sleep onset | exact revisit | other | all |
|---|---|---|---|---|---|---|
| L4 | -511% | -1274% | -180% | -1114% | -828% | -641% |
| L16 | +2% | -7% | +15% | -7% | -1% | +1% |
| L16x3 | -3% | -15% | +13% | -9% | -7% | -5% |
| L64 | +19% | +24% | +42% | +11% | +21% | +21% |
| L4to64 | +9% | +9% | +13% | -3% | +8% | +9% |

- Exact revisits: the next map view pixel-identical to a frame 4+ back and to none of the last 3. They are
  53 of 28,672 transitions (0.19%). Recall of content that left the window is almost absent from this data.
- Where memory helps (L64), it helps every transition type about equally.

**Precedents for (a) vs (b), verified in source:**
- DRAMA (our Mamba base, `third_party/Drama/config_files/configure.yaml`):
  - trains its Mamba world on BatchLength 128 x BatchSize 16 (2,048 transitions per update);
  - imagines 16 steps from an 8-frame context (ImagineContextLength 8);
  - acts from a bounded 16-frame deque (RealityContextLength 16, `train.py:149`, `eval.py:56`).
  - So every deployed position (<= 24) lies inside its training length.
- Dreamer 4: batch lengths 32/128 alternating, context 96-192 frames.
- le-wm: trains 4 frames, deploys a 3-frame window.
- All three keep deployment inside the trained range. We trained 4 frames and deployed an unbounded recurrence.

**E4a action identifiability (`identifiability.py`, exact):**
- Futures roots, all 17 actions under one key. Bayes accuracy of ANY decoder of (o_t, o_{t+1}):
  0.339 with uniform actions, 0.583 with the corpus action prior.
- 44% of transitions have a uniquely identified action.
- Dominant collision: NOOP / DO / SLEEP / failed PLACE and MAKE; blocked moves when already facing that way.
- LDAD's training accuracy reached 0.57 by update 1,500 (lambda 10): it saturates at the ceiling early.

**E5 / E3b, per-tile worlds, seed 7 (`evals/*.json`, `compare.json`).**
- Setup: futures roots; paired bootstrap over 143 walk seeds (root-sampling uncertainty only; second
  training seeds queued).
- Each state's errors are relative to its own copy baseline (x copy) or variance (/V).

| world | one-step all | moved | blocked | sleep | idle | 16-step imagined (/V) | health R^2 at k16 |
|---|---|---|---|---|---|---|---|
| direct, Raw tokens | 0.902 | 0.971 | 0.978 | 0.750 | 1.066 | 0.969 | 0.11 |
| residual, Raw tokens | **0.596** | 0.912 | 1.335 | 0.049 | 0.805 | **0.849** | 0.56 |
| residual, TC tokens | **0.541** | 0.936 | **0.542** | 0.040 | 0.693 | 0.869 | 0.60 |

- Residual vs direct: better on everything except blocked moves (+0.36*); 16-step -0.120*.
- TC vs Raw (residual):
  - one-step -0.055*, blocked -0.79*;
  - moved +0.024* and interact +0.059* (TC worse);
  - 16-step +0.020, not resolved.
- Moves remain almost unlearned by the direct/residual/gated heads: 0.91-0.97x copy, against a local-linear
  ceiling of 0.72 captured.

**E5, copy variants, seed 7 (Raw tokens; `compare.json`):**

| head | moved | blocked | idle | all | gen 4 / 8 / 16 (/V) |
|---|---|---|---|---|---|
| residual | 0.912 | 1.335 | 0.805 | 0.596 | 0.425 / 0.605 / 0.849 |
| gated | 0.910 | 0.933 | **0.483** | 0.548 | 0.418 / 0.598 / 0.845 |
| corr (neighbour copy) | **0.203** | 6.121 | 0.848 | **0.380** | **0.307 / 0.505** / 0.825 |

- corr vs residual:
  - moved -0.709 [-0.721, -0.699]*, all -0.216*;
  - gen 4 -0.118*, gen 8 -0.100*, gen 16 -0.024 (not resolved);
  - blocked +4.79*.
- corr learns WHAT a move does (the scroll) but not WHETHER it happens. Imagined moved/blocked change 1.15x vs
  8.0x true; every blocked move in a rollout injects a full wrong scroll, so the gain fades by depth 16.
- The decision is one global bit set by the tile in front of the player, but corr decides per tile.
- E5b `corrg` = corr + a frame-level move logit from the action token, zero-init (verified identical to corr at
  init).
  - vs corr: blocked 6.12 -> 4.61 [-2.64, -0.94]*; all 0.380 -> 0.350*; gen 4 -0.018*, gen 16 +0.014*.
  - vs residual: gen 16 -0.010 (not resolved).
- E5c (`corrg_diag.json`), read out of the model, moves only:
  - the corrg frame logit separates moved/blocked at AUC 0.78 (training windows 0.77): means 2.19 vs -0.18;
  - neighbour copy mass on blocked moves is still 0.36;
  - the decision is under-learned on training data too.
- E5d (`corrg_probe.json`), probes on the frozen corrg backbone, moved vs blocked AUC:
  - input token of the target tile 1.00; backbone output AT the target tile 1.00;
  - player tile 0.86; action token 0.88; the gate itself 0.79.
  - The backbone knows passability at the target tile but does not route it to the global decision (an
    action-keyed lookup of the tile at player + direction).
- **E5e result, `corrt`** (the gate reads the backbone output AT the target tile). Routing was the problem:

| head | moved | blocked | all | gen 1 / 4 / 8 / 16 (/V) |
|---|---|---|---|---|
| residual | 0.912 | 1.335 | 0.596 | 0.168 / 0.425 / 0.605 / 0.849 |
| corr | 0.203 | 6.121 | 0.380 | 0.096 / 0.307 / 0.505 / 0.825 |
| corrg | 0.203 | 4.612 | 0.350 | 0.086 / 0.289 / 0.495 / 0.839 |
| **corrt** | **0.196** | **0.913** | **0.302** | **0.054 / 0.229 / 0.432 / 0.810** |

  - corrt vs corr: blocked -5.21*, all -0.078*, gen 1 / 4 / 8 all resolved; gen 16 -0.015 (not resolved).
  - corrt vs residual: every depth resolved (gen 16 -0.039*).
  - Imagined moved/blocked change 70.7x (true 8.0; corr 1.15x): it now scrolls when the move succeeds and holds
    the view when it is blocked.
  - Costs: sleep onset 0.19 vs 0.05*; 16-step error still 0.81 of variance (compounding remains).
  - The bias assumes the player sits at the centre of an egocentric view (Craftax-specific). corrg at 3x updates
    (lane 3d) tests whether plain attention learns the same routing.
- **Training-seed noise floor** (corr seed 7 vs seed 8, same recipe):
  - moved 0.203 vs 0.197; blocked 6.12 vs 5.82 (not resolved);
  - idle 0.848 vs 0.518*, all 0.380 vs 0.329*, gen 16 0.825 vs 0.799*.
  - The bootstrap covers root sampling only, so one-seed differences of this size (idle, all) are NOT
    established. Gated's idle 0.48 falls inside this. Effects far beyond it stand:
    - corr on moves (-0.71 vs seed spread 0.006);
    - TC on blocked moves.
- **corr on TC tokens** vs Raw: moved +0.091*, blocked -2.54*, all +0.030*, gen 16 -0.016 (not resolved).
  Same trade-off as under the residual head (blocked better, moves worse), in both heads.
- E5e (lane 3d), which of the two is it:
  - `corrt` adds a gate read directly at the target tile (player token 31 + direction, moves only), zero-init,
    verified identical to corr at init;
  - `corrg` at 3x updates (18,000).

**E4, Raw LDAD lambda 10 training curve** (`levers_ldad_v1/raw_lam10/metrics.jsonl`, paired with canonical Raw):
- At updates 9,501-10,000: prediction MSE 0.097 vs 0.018 (5.3x); SIGReg 2.37 vs 1.25.
- Gradient norm 29 vs 0.93: clipped at 1.0 throughout, so LDAD sets the update direction.
- LDAD accuracy plateaus at 0.607, at the identifiability ceiling (E4a: 0.583 on futures roots).
- Whether the MSE rise is only larger latent steps: the copy-normalized evaluation (`ldad_eval.py`) decides.

**Incident 3 (15:04):** PC shut down. `/tmp` was wiped, so scratchpad logs are lost; all artifacts survived.
- Lost: the corr arm, E1e, lane D.
- Relaunched as admission-controlled lanes (`lib.sh`):
  - a job starts only when free GPU memory >= its declared need + 256 MiB;
  - CUDA-OOM is retried; any other failure stops the lane;
  - jobs are idempotent.
- Logs: `artifacts/eda/levers_logs/` (`lanes.log` = the timeline).
- Lanes: 1 = per-tile arms + evals + second seeds (`lane1.sh`); 2 = LDAD screens + LDAD evals + E1e/E1f + E1c
  (`lane2.sh`); 3 = early eval of the gated/corr arms (`lane3.sh`).

**Incident 2 (14:48-14:58):** too many concurrent GPU jobs.
- The lambda screens, the screen evaluation and one context run died of CUDA OOM, and the lane script moved on
  silently.
- Fix: lane D (`lane_d.sh`) runs all non-lane-A GPU jobs serially after lane C4, retries CUDA-OOM failures (6 x 3 min)
  and stops on any other failure.

**E4 results, Delta-JEPA (LDAD).**
- Encoder content (`ldad_eval.json`, 10k; `ldad_screens.json`, 2k paired with canonical step-2000):

| 10k updates | canonical Raw | Raw + LDAD lambda 10 |
|---|---|---|
| z: zombie adjacent (AUC) | 0.606 | **0.961** |
| z: passability, 4 dirs (AUC) | 0.588 | **0.855** |
| z: health (R^2) | 0.122 | **0.728** |
| z: d' of one added zombie | 0.51 | **3.38** |
| z: cos(adjacent, far zombie change) | 1.000 | **0.940** |
| action from dz (accuracy) | 0.35 | 0.67 |

- 2k screens, z zombie adjacent / passability: canonical 0.616 / 0.600; lambda 0.1 0.627 / 0.624; lambda 1
  0.850 / 0.791.

**E4b common currency** (`ldad_facts.py`). Probes fitted on TRUE z per run, read on each world's IMAGINED z
(3-frame window). Imagined at depth 1 / 4 / 16:

| fact | canonical Raw | Raw + LDAD lambda 10 |
|---|---|---|
| zombie adjacent | 0.63 / 0.66 / 0.54 | **0.91 / 0.82** / 0.60 |
| pass up | 0.53 / 0.50 / 0.54 | **0.79 / 0.65** / 0.56 |
| health (R^2) | 0.08 / 0.07 / 0.06 | **0.42 / 0.37 / 0.13** |
| food (R^2) | **0.75 / 0.76 / 0.69** | 0.54 / 0.49 / 0.38 |
| energy (R^2) | **0.71 / 0.67 / 0.62** | 0.47 / 0.37 / 0.26 |
| one-step moved vs blocked from imagined z(a) | 0.71 | **0.86** |
| one-step health drop from imagined z(a) | 0.56 | **0.88** |

- LDAD makes the world imagine the decision facts (zombie, passability, health drop).
- Costs:
  - slow HUD facts (food, energy) degrade;
  - by depth 16 imagined zombie / passability fall to or below copying the root (0.60 vs 0.66; pass left
    0.51 vs 0.68);
  - prediction MSE 5.3x, SIGReg 1.9x.
- 2k screens cannot judge imagination (the world lags the encoder: raw lambda 1 true zombie 0.96, imagined 0.62).
- TC + LDAD lambda 10 at 2k restores passability in TC's z (true 0.84-0.91 vs plain TC 0.60-0.70).
- Full 10k runs of TC lambda 10 and Raw lambda 1 are queued (lane 4).

**Incident (2026-09-27 12:10):** the machine hung and rebooted with 5 jobs running (no OOM-killer record; no swap).
- `ldad_eval.py`'s probes loaded four full judgement blocks (~13 GB of rows) twice.
- Fix: one lean loader (`tc_encoder.blocks()`: visible state + last frame only).
- Since then every job runs in a systemd unit with MemoryHigh/MemoryMax, in at most 3 lanes:
  - A: `queue_tworlds.sh`, 9/12 GB.
  - B: `lane_b.sh`, 15/17 GB.
  - C: E1b + E1c, 6/8 GB.
- Measured process (anonymous) memory: 1.3-2.0 GB per lane; the rest is reclaimable corpus page cache.
- Lost and restarted: the residual T arm (2,500/6,000 updates) and the Raw LDAD run (487/10,000).

**E3a (`tc_equiv.json`):** the tokens carry the same facts but are NOT linearly equivalent.
- Linear R^2: raw -> tc 0.83, tc -> raw 0.89; on the player token 0.48 / 0.70. CKA 0.63.
- Facts: tile 0.990 / 0.989, zombie near 1.00 / 1.00, health 0.999 / 0.997.
- Linearly learnable change: blocked -2.09 vs -1.07, idle -0.21 vs +0.03.

**E2 codebooks (`codebook.json`):**
- Raw K = 4096: relative error 0.033; tile facts 0.96-0.98 (continuous 0.97-0.99); health R^2 0.94 (1.00).
- Code flips on tiles where nothing changed: 14.7% for Raw tokens, 27.6% for TC tokens (contextual ViT tokens).

---

## 2026-09-27 — Diagnosis part 2 (`20260926_diagnosis/TRANSITION.md`, commits 6a744366, a9ee14d1)

- **The one-step failure is on each world's own TRAIN data** (`insample.json`); held-out is within 0.05.
  - All transitions, x copy: H2 1.34, Z 0.61, sZ 1.17, U 0.57, W 0.97, T 0.89.
- **Per state** (`learnable.json`, `oracle.json`):
  - z: a move's change is not a function of z; ridge captures 0.08.
  - u: the worlds sit at u's linear ceiling (0.63); u lacks passability (AUC 0.69).
  - T: the information is present (local-linear 0.72, rigid-shift oracle 0.83), but the trained world captures
    0.08.
- **T/sZ cannot beat copying because the output has no copy path** (`parameterization.json`). Held-out L1:
  - T: direct 0.141, copy 0.122, residual 0.112.
  - sZ: direct 0.097, copy 0.077, residual 0.069.
- **Canonical H2: bridge broke it; the Mamba does not extrapolate its 4-frame training** (`bnmode.json`,
  `context.json`, `memory.json`).
  - x copy: joint 0.66, bridge 2.01. BatchNorm mode is ruled out.
  - The Mamba state grows 3.2-4.4x past the training length.
- **LeWM's own rule repairs the joint world** (`sliding.json`): le-wm rollout uses `emb[:, -3:]`.
  - Joint world, 16-step imagined error / V: 2.57 -> 0.53 (copy 0.93); one step 0.048 -> 0.025.
- **Recursion compounds because the latent is a near-integrator** (`compound.json`, `persistence.json`):
  - 81-92% of depth-16 error is inherited, passed on at 0.82-0.97 per direction.
  - True autocorrelation 0.92-0.99 (Lambert et al. 2022).
  - Fresh errors are correlated (cos 0.48-0.72); growth exponent 1.18-1.28.
  - Error sits 1.5-22x in low-variance directions, and HUD facts end up below copying.
- **TC encoder** (`tc_encoder.json`):
  - z within-window variance 1.6% -> 44.8%; mobs up to 5x more detectable.
  - Static terrain nearly dropped (d' 0.1); passability unchanged (0.59).
- **Correction (E3a above):** "TC and Raw patch tokens are identical" was too strong. Same facts, different
  geometry.
- **Correction:** head-only whitening on the frozen U world gives +0.01 SLEEP-death AUC, of W's 0.09
  (`headonly.json`). The LeJEPA-Lemma-1 account of W is not supported.

## 2026-09-26 — Diagnosis part 1 (`20260926_diagnosis/DIAGNOSIS.md`, commit 8e95f745)

- **The zombie decision is a visible rule**: move into a free tile. A rule on visible tiles scores 0.977-0.982;
  the hidden cooldown barely matters (`decision.json`).
- **True-successor substitution**: U 0.975-0.981, W 0.990-0.999 against 0.49-0.64 imagined. For U and W the
  whole loss is imagination (`choices.json`).
- **z is position-blind**: cosine 1.000 for an adjacent vs a far zombie. Training erased zombie presence
  (AUC 0.947 -> 0.528) (`twins.json`).
- **Heads underrate SLEEP**: only 3.2% of corpus death frames render asleep. H2 picks SLEEP on 74-79% of
  zombie roots (`sleep.json`, `followups.json`).
- **W > U** is about 79% not choosing SLEEP (`followups.json`).
- Other agent's audit (`20260926_mechanism_audit/FINDINGS.md`):
  - Lava-entry mislabel corrected.
  - CLS position blindness already exists at initialization.
  - Positional embeddings are ~8% of the patch-embedding norm.

## 2026-09-21..26 — Readout ladder (`20260921_readout_ladder/`, WHY.md, INTERFACE.md)

- Sealed 59k: the whitened patch-state Mamba world's own head beats DOWN (+0.083*) and actions_only (+0.056*),
  and beats DOWN on zombie roots (+0.059*). Positive in 7/7 blocks, resolved on zombie roots in 2/7.
- Factorial (post hoc 55k-58k): only mob-in-state + isotropy works (W 0.54-0.64 vs Z/ZW/U).
- Canonical SIGReg z is not eigen-isotropic: spread 6,220x, effective rank 44/192.
- H2 depth alias: P(dead) 0.73 at depth 2, 0.05 at the gate position.
- Superseded by the diagnosis above:
  - "tail learned last" was retracted (ridge oracle).
  - The isotropy/Lemma-1 mechanism for W is not supported (`headonly.json`).
