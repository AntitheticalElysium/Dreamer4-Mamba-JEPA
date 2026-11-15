# Temporal memory research pass (2026-11-16): plan

User 11-16: research how to integrate the per-cell recency (F6.1 dissection, checks 16-18), with the brain's memory
as the hook; dig into CogRNN; discuss whether the recency should be separate from Mamba; run what is needed.
No design is chosen by this folder. Literature notes: RESEARCH.md (written alongside).

## Measurement 1: how readable is each per-cell time code? (`readability.py`, CPU + GPU tokens only, no training)

Why: training showed that the same timing information is used by the denoiser as an exact recency (check 16-18: awake
energy drawn 0.03 -> 0.80) and not as two leaky traces (check 12-14: gap 2-3% of the input's spread). CogRNN (Kabir,
Mochizuki-Freeman & Tiganj, AAAI 2025) offers a third form: a Laplace bank of leaky integrators per input channel and
its fixed Post inverse (time cells), whose widths grow with elapsed time (Weber: coefficient of variation 1/sqrt(k+1)).
Before proposing a fine-tune with it, measure what its form gives on our own data for these clocks.

Claim measured: how much of each digit's next-frame change is readable from each code by a linear reader (the form a
zero-initialised linear input map gives the denoiser) and by a nonlinear reader (whether the information is there).
Competing explanation for any gap: missing information vs unreadable form; the nonlinear reader separates them.
Instrument check (known outcomes): the exact recency and the traces must order as training ordered them.

Inputs, label-free: F6.1's coarse periphery tokens (ckpt 10) at the digit cells 63-66; a cell's event = squared token
change > 1 (check 16's threshold). Codes at frame t-1 for the change into frame t (token_trace's alignment):
- `csc`: check 16's recency u = max(0, 63 - frames since the cell's last event), one-hot over 64 values (the complete
  serial compound of Ludvig, Sutton & Kehoe 2008).
- `traces`: check 12's input, log(1 + EMA) of the squared change with time constants 8 and 32.
- Laplace F and its inverse f~ of the event train, from the vendored CogRNN code (`cogrnn_kabir2025.py`, decay rates and
  Post matrix built by its own class; our scan checked against its scan): `paper_8_k8` (the paper's main setting:
  8 time cells, tau* 1-1000, k = 8) and the repository default `repo_80_k8` (80 time cells), plus the paper's k sweep
  values 4 and 16 at 80 cells (Fig. S7). For `paper_8_k8` both F and f~ (the paper's ablation contrast).

Readers per digit cell, fitted on DEV folds 0-3 (token_trace's 400 training episodes, every frame; target: the cell's
token changes at the next frame), scored on fold 4: (a) linear: logistic regression on standardised codes, L2 1e-4,
L-BFGS; (b) nonlinear: LightGBM with token_trace's settings. Score: AP at the clock events of the `after_reset` cohort
(token_trace's rows) for food / drink / awake energy drops and the asleep energy rise, with the base rate.
Second reading, idealised: the distance between a code after one isolated event at the pair lags (energy 30 vs 22,
food 25 vs 17, drink 20 vs 12, and 30 vs 29) as a fraction of the code's spread over the training frames (check 13's
"gap as a share of the input's spread").

Readings fixed before results:
- Instrument check: linear AP of `csc` on the awake energy drop at least twice that of `traces`. If it fails, the
  linear reader does not reflect what training found and the Laplace rows are reported but not interpreted.
- Each Laplace code: linear AP as a fraction of `csc`'s linear AP, per clock; nonlinear AP beside it.
- No bar decides a design: this is a fact for the discussion (b) is about.

## Measurement 2 (user 11-16, question 1): does the SSM's eigenvalue structure decide time readability? (`ssm_bases.py`)

User: state-space models should excel at remembering time; is the CogRNN linear-vs-nonlinear gap a sign of the same
problem around Mamba? Sources read: Mamba-3 (arXiv 2603.15569, its text: "S4 used complex-valued NPLR matrices, Mamba
reduced this to a diagonal of reals, and Mamba-2 further simplified it to a single scaled identity matrix"; real
non-negative eigenvalues "cannot represent rotational hidden state dynamics"), HiPPO (2008.07669: LegT = the LMU's
sliding-window Legendre memory, which reconstructs the input over the window linearly from the state), S4D
(2206.11893: S4D-Lin A_n = -1/2 + i pi n, "damped Fourier basis functions"; S4D-Real A_n = -(n+1), S4's real spectrum).

Claim measured: fixed SSM bases whose transitions rotate (complex or Legendre) make elapsed time linearly readable;
real-decay bases (what Mamba-1/2 can express per head) do not. Same events, rows, readers and references as
measurement 1; the exact recency and the Laplace F are the references. Codes, each from its paper:
- `legt_16`, `legt_64`: HiPPO-LegT in its LMU form (HiPPO eq. translated-legendre-dynamics: A_nk = (2n+1)/theta x
  ((-1)^(n-k) if n >= k else 1), B_n = (2n+1)(-1)^n / theta), ZOH at dt = 1 (HiPPO: only theta / dt matters), window
  theta = 64 frames (the recency's horizon), N = 16 and 64 (64 = the recency's 64 values).
- `s4d_lin`: S4D-Lin, N/2 = 32 complex modes, B_n = 1, the S4D listing's bilinear discretisation, dt = 1/32 (damping
  time 2/dt = 64 frames; highest frequency below Nyquist; inside S4D's dt range 0.001-0.1); real and imaginary parts.
- `s4d_real`: S4D-Real, N = 64, same dt and discretisation (the real counterpart the S4D paper compares).
Reading fixed before results: linear AP per clock as a fraction of the recency's; no design is decided by it.

## Measurement 3 (user 11-16, question 1): where does a learned Mamba memory lose readable time? (`architecture.py`)

User: is the belief's global compression before Mamba, or the memory tokens after it, why Mamba keeps so little
readable time beyond what a loss asks for, or is it incentive? A localisation by substitution, not a selection: no
arm is adopted on its result.

Known before this test: asked directly for the events (11-14 A1, simulator labels, fine-tune of F6's belief), the
global belief reached awake energy AP 0.65 and the asleep rise 0.96 by 1,500 updates, food and drink about 0.20
(different rows from ours); a small global Mamba-2 fixture on 80-frame crops under a generic all-patch change target
stayed at base rate (Codex 11-15). Po et al. (ICCV 2025, read in source): a single global scan over the flattened
frame lost long memory, per-block scans (each spatial block its own state) kept it, block size 1 lost spatial
coherence; and a diffusion world model ignored distant context until its training gave it a reason.

Setup: F6.1's frozen coarse tokens (81 x 16) and the actions, whole training-split episodes packed into 4,096 frames
per update (all 7,672 episodes), the belief's optimiser (AdamW, lr 1e-4, weight decay 0.01, clip 1, warm-up 1,024,000
frames, cosine to 0.1), 3,000 updates (12.3M frames, A1's frame count); every model trained from scratch on the same
batches. Objective, label-free: every cell's next-frame change (measurement 1's event), log-loss. Each model's own
head (a cell's memory feature + its current token + its identity -> logit, same form in every arm) is read at
measurement 1's clock rows on DEV fold 4, every episode from its zero state, at 1,000 / 2,000 / 3,000 updates.

| Arm | Memory | Hypothesis it isolates |
| :--- | :--- | :--- |
| G2 | the belief's form: flattened percept -> one linear map -> 6-layer Mamba-2 (width 256, the belief's settings) -> per-cell linear readout | reference |
| G2K | G2, read through K = 4 memory tokens by each cell's query (the denoiser's interface) | the memory-token interface |
| P2 | one 6-layer Mamba-2 (width 64) shared by all cells, each cell its own sequence and state | the global compression |
| G3 / P3 | G2 / P2 with the official Mamba-3 mixer (installed mamba_ssm 2.3.2, SISO, its defaults: complex, data-dependent rotations) | real decays vs rotations |
| G2_digits / P2_digits | G2 / P2 with the objective restricted to the four HUD digit cells (a diagnostic; never a recipe) | the incentive (dilution among 81 cells) |

Simplifications declared: one token per frame (percept + action), not the belief's interleaved pair; no glimpse,
lighting, reward or termination input; the readout head replaces the denoiser.

Readings fixed before results (two init seeds; a difference counts only if it holds in both seeds at 3,000 updates):
- Compression: P2 above G2 on the awake energy drop by more than 0.2 AP.
- Interface: G2K below G2 by more than 0.2.
- Rotations: P3 above P2, or G3 above G2, by more than 0.2.
- Incentive: an arm's digits-only variant above its all-cell form by more than 0.2.
- References from measurements 1-2: the exact recency (linear 0.94 on energy) and the Laplace record (nonlinear 0.93).
- The generic log-loss over all cells is reported beside each arm, to see whether an arm's clocks come at the generic
  task's expense.
Implementation notes before the run (smoke test): per-cell memories are computed in thirds of the cells with gradients
accumulated (cells are independent in them: equal to one pass up to bf16 rounding, max relative difference 1.6e-3),
the digit-only objective on its four cells only; the token cache's memmap shape fixed (NumPy 2 wrote np.int64 into
the header). Parameters: global arms 2.0-2.4M, per-cell arms 0.15M (one 64-wide stack shared by 81 cells, each with
its own state). About 1.45 s per update for all seven; seeds 1 and 2 run one after the other.
Amendment (user 11-16, before any reading): one seed. Compute goes to iteration; the second seed is dropped. The
readings' "both seeds" clause becomes: seed 1 at 3,000 updates; a difference within 0.1 of its 0.2 bar is reported as
unresolved, not as a finding (check 18 measured run-to-run spread up to 0.12 on 2,000-update pair reads).
