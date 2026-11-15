# Temporal memory in the brain, CogRNN, and Mamba (research pass 2026-11-16)

Question (user 11-16): checks 16-18 showed that a per-cell recency, read by the denoiser, lets G1 alone draw the
pixel clocks on time. How should it be integrated, is there a brain hook that makes it more general, is CogRNN a
serious contender, and should the recency be separate from Mamba ("two sources of truth")? Each source below was
read in the original (paper text, code, or abstract where marked); nothing here decides a design.

## 1. The brain: per-item records of "what happened when", read out by learned weights

**A log-compressed, per-item record of the past (Laplace layer).**
- Shankar & Howard 2012 (Neural Computation 24:134-193); Howard et al. 2014 (J Neurosci 34:4692): a bank of leaky
  integrators with a spectrum of rates s holds the real Laplace transform of each input's history; a fixed linear
  operator (Post's approximation, order k) inverts it into units that fire at a preferred lag tau* = k/s, with widths
  proportional to tau* (Weber). The same equations code time (alpha = 1), position (alpha = velocity) or sequence
  position (alpha = speed): "The function alpha(tau) is the same for all of the cells in F(s) but can change over time"
  (2014, read in PMC full text; equations are images there). Inputs are per stimulus: "many cells receiving input
  corresponding to many nonspatial stimuli".
- Bright et al. 2020 (PNAS 117:20274): macaque entorhinal neurons respond to image onset and relax at a spectrum of
  rates, so elapsed time is decodable for seconds; the same neurons carry which image was shown. The authors read this
  as the Laplace layer ("temporal context cells"), distinct from hippocampal time cells.
- Tsao et al. 2018 (Nature 561:57): lateral entorhinal populations carry time from seconds to hours, more strongly than
  MEC or CA3-CA1, and the signal depends on the experience that drives it ("time from experience"): the clock runs on
  events, not a fixed tick.

**A readable timeline (inverse layer, time cells).**
- Cao et al. 2022 (eLife 11:e75353): hippocampal time-cell field widths grow linearly with delay and their peaks are
  evenly spaced on a log axis (a neural Weber-Fechner law).
- Tiganj et al. 2018 (J Cogn Neurosci 30:935): in monkey lateral PFC each sample stimulus triggers its own sequence of
  time cells; the code carries what and when, with precision falling with age.
- Taxidis et al. 2020 (Neuron 108:984): CA1 has stimulus-specific sequences (stable odour cells, then time cells that
  grew with task training and remapped when the trial structure changed): time cells are shaped by experience.

**Per-item recency as a behavioural and neural quantity.**
- Howard et al. 2015 (Psych Rev 122:24): the same representation accounts for judgments of recency across scales.
- Xiang & Brown 1998 (Neuropharmacology 37:657, abstract): in TE, perirhinal and entorhinal cortex, 19% of differential
  neurons signal recency (how recently this item was seen), separate from familiarity and novelty neurons.

**Event-triggered temporal bases with a learned readout predict WHEN consequences arrive (the general motif).**
- Kennedy et al. 2014 (Nat Neurosci 17:416, abstract): in the electric fish's cerebellum-like lobe, granule cells
  provide a temporal basis after each motor command, and plastic readout cells learn negative images of its predictable
  sensory consequences.
- Shuler & Bear 2006 (Science 311:1606): rat V1 neurons learn to time when a reward follows a visual stimulus; timing
  appears in primary sensory cortex, at the stimulated location. Chubykin et al. 2013 (Neuron): learning it needs
  cholinergic input, expressing it does not.
- Janssen & Shadlen 2005 (Nat Neurosci 8:234): LIP neurons, at the receptive field of the planned saccade, carry
  elapsed time and the hazard rate of the go signal (blurred by the subject's timing uncertainty); a subpopulation only
  (Premereur et al. 2011).
- Ludvig, Sutton & Kehoe 2008 (Neural Computation 20:3034, abstract): TD models of dopamine need a temporal stimulus
  representation; the classic one is the complete serial compound (one feature per time step since the stimulus);
  microstimuli (Gaussian features over a decaying memory trace, weaker and wider with time) fit the data better.
  Our exact recency embedding (check 16) IS the complete serial compound.

**Multi-timescale memory is local, everywhere.**
- Ulanovsky et al. 2004 (J Neurosci): stimulus-specific adaptation in A1 at time scales from hundreds of ms to tens of
  seconds; Lundstrom et al. 2008 (Nat Neurosci 11:1335): single pyramidal neurons adapt on many time scales at once
  (fractional differentiation of the input history).
- Murray et al. 2014 (Nat Neurosci 17:1661): intrinsic timescales rise from sensory to prefrontal areas (about 50 to
  350 ms in each dataset), ordered like the anatomical hierarchy.
- Kietzmann et al. 2019 (PNAS): local recurrent circuits at every retinotopic location are needed to match human
  ventral-stream dynamics (cited in this session's earlier answer).

**Reference frames.**
- Golomb, Chun & Mazer 2008 (J Neurosci 28:10654, abstract): attentional traces stay at the retinotopic location
  after a saccade for 100-200 ms even when only the world location matters; remapping to world coordinates is slow and
  happens when needed.
- Tsao, Moser & Moser 2013 (Curr Biol 23:399, abstract): LEC object-trace cells fire where an object used to be: a
  world-keyed record of past content.

**Reading for us.** The general principle is not "a recency feature" but: each item (here: each cell) keeps its own
record of when things happened to it, in a form a learned readout can turn into "when will the next consequence
come". The record lives upstream of, and feeds, the structures that form beliefs (EC -> hippocampus); it is computed
in the frame where the item is stationary (retinotopic traces; world-keyed object traces). Our per-cell recency is the
screen-keyed member of that family; the world-keyed member is the I1 store already in STEP_SPEC ("location-keyed
traces: background content, time stamp, reliability").

## 2. CogRNN (Kabir, Mochizuki-Freeman & Tiganj, AAAI 2025, arXiv 2412.15292)

Read: the paper's LaTeX source, its supplementary figures, and the code (github.com/cogneuroai/RL-with-scale-invariant-
memory, commit 878f985, vendored as `cogrnn_kabir2025.py`).
- What it is: for every input feature f (encoder output), N = n_taus + 2k leaky integrators F_t = exp(-s dt) F_{t-1}
  + f_t dt, s = k / tau*, tau* log-spaced; then f~ = F @ Post, a fixed matrix (k-th finite-difference derivative in s
  times s^(k+1)/k!). No trainable weights in the memory. Impulse response of a time cell: peak at tau*, coefficient of
  variation 1/sqrt(k+1). The code also takes alpha (clock rate) and delta inputs: decay exp(-s (alpha dt + delta));
  with `alpha_mod` the encoder output drives the clock and f = 0 (Howard 2014's velocity case).
- Use in the agent: the 3D T-maze agent flattens f~ (features x tau*) or attends over tau* (two attention layers,
  8 heads, d_model 128) before the actor-critic heads.
- Evidence: A2C on interval timing (1D and 3D), interval discrimination, delayed match to sample, interval
  reproduction; 5 runs each; CogRNN learns at the same speed when task intervals are rescaled, LSTM and RNN do not;
  with convolution + pooling over tau* a REINFORCE agent trained at one time scale works at 2x and 4x. Time-cell widths
  grow linearly with peak time only in CogRNN agents.
- Ablation (Fig. S1, read from the figure): with F alone (no inverse) the 3D agent reaches a reward of about 6.3 at
  170k trials against about 8.2 for f~ (Fig. S7, k = 8); frozen RNN and LSTM plateau near 4.8. The inverse helps;
  F alone still learns part of it.
- k sweep (Fig. S7): k = 4 and 8 end near 8.2, k = 16 lower (about 7.4) and noisier.
- **Discrepancy found:** the paper states 8 log-spaced units, tau* 1-1000, k = 8; the repository's default is 80 units.
  With 8 units (spacing factor 2.68) the code's own Post inverse is numerically broken: impulse responses of the
  cells meant to peak at 1-7 frames peak at 255-1,837 frames, values reach 1.9e9. With 80 units (factor 1.09) the
  cells peak near tau* (k = 8: tau* 25.4 peaks at 28, half maximum 18-42 frames; k = 16: peaks drift later, 35 for
  tau* 25.4, a known effect the code's `DEBUG_dt_scale` corrects). Which setting produced the paper's figures cannot be
  told from the text.
- What it does not address: exact deterministic intervals. Its virtue is equal learning across time scales, bought
  with Weber blur: around 25-30 frames a time cell is about 24 frames wide at half maximum (k = 8).

## 3. Mamba is the same family, with a learned clock and no inverse

- Gu & Dao 2023 (arXiv 2312.00752, source text): "Delta controls the balance between how much to focus or ignore the
  current input x_t ... a large Delta resets the state h and focuses on the current input x, while a small Delta
  persists the state and ignores the current input"; A acts only through exp(Delta A). A diagonal real SSM is a bank
  of leaky integrators (F), and an input-dependent Delta is a learned clock rate: structurally Howard's alpha(t),
  and close to "time from experience" (Tsao 2018).
- Our belief (m0.yaml): Mamba-2, 6 layers, width 256, headdim 64 -> 4 heads per layer, one scalar decay per head:
  24 time constants in the whole belief, initialised log-spread (A in [1, 16], dt log-uniform in [0.001, 0.1]:
  0.6-1,000 tokens, i.e. about 0.3-500 environment steps, two tokens per step). Every cell enters through one
  flattened linear map; there is no per-item input and no inverse.
- Codex 11-15 (`../20261115_mamba_timescales/RESULTS.md`): in F6, 81 of 576 head-window combinations keep more than
  half of a value across 32 frames; two fixed EMAs per cell (162 channels) fit exactly into the existing carry
  (representable, at the cost of the heads' other uses); a nonlinear reader of five-time-constant visual traces
  reaches awake energy AP 0.83 (no step input); training a recurrence to reconstruct such traces did not teach the
  native learner (detached and attached arms equal).

## 4. What follows (for the discussion; measurement in RESULTS.md)

- Brain-grounded form: per-item event-triggered records, log-spread rates, a readable timeline, a learned readout.
  The exact recency is that family's idealised limit (complete serial compound): exact, no generalisation across
  lags, last event only, capped at 63.
- "Two sources of truth": the per-cell record is a deterministic function of the percept history, like the denoiser's
  8-frame window, so it cannot contradict the history; what it can do is bypass the belief. In the brain the record
  feeds the belief (EC -> hippocampus). The integration that matches is: one record, read by both Mamba and the
  denoiser, rather than a denoiser-private input or a replacement for Mamba.
- Frame: the screen-keyed record is right for content stationary on the screen; world-anchored timing needs the
  world-keyed store (I1).

## 5. Second round (user 11-16: should SSMs not excel here; integrate with Mamba or a separate system?)

**What an SSM's state can make readable depends on its transitions.**
- Mamba-3 (Lahoti et al., ICLR 2026, arXiv 2603.15569, read in source): "S4 used complex-valued NPLR matrices, Mamba
  reduced this to a diagonal of reals, and Mamba-2 further simplified it to a single scaled identity matrix"; the
  restriction to real non-negative eigenvalues "cannot represent rotational hidden state dynamics"; Mamba-3 restores
  complex states as data-dependent rotary embeddings on B and C, and solves parity and modular arithmetic where Mamba-2
  fails. Grazzi et al. (ICLR 2025, 2411.12537): positive real eigenvalues cannot solve parity; complex ones are needed to
  count modulo 3.
- HiPPO (Gu et al. 2020, 2008.07669, source): LegT keeps a Legendre projection of the last theta steps, from which the
  window is reconstructed linearly; its LMU form is "exactly the LMU update equation". Voelker & Eliasmith 2018 (Neural
  Computation, abstract): the same delay network gives an account of time cells.
- S4D (Gu et al. 2022, 2206.11893, source): S4D-Lin A_n = -1/2 + i pi n gives "damped Fourier basis functions";
  S4D-Real A_n = -(n+1) is S4's real spectrum.
- Our measurement 2: at equal size and step, the complex basis reads the awake energy clock linearly at 0.90, the real
  one at 0.15, with the same information in both (nonlinear 0.94).

**Where the memory sits in SSM world models.**
- Po et al. (ICCV 2025, 2505.20171, source): Mamba layers inside the generator, interleaved with frame-local attention
  (window 10 frames); "a separate scan is performed for each token block", each block its own state, block sizes
  varying across layers. Ablation (maze reasoning, SSIM): full 0.855; without block-wise scan 0.845; block size 1 only
  0.766; without the long-context training 0.809. Without that training "our model falls into a local minimum and
  generates frames without looking at distant context". Retrieval: Mamba2 + frame local attention 0.735 SSIM against
  0.898 for theirs ("limited state-space expressivity").
- SlotSSM (Jiang et al., NeurIPS 2024, abstract): modular slot states with independent transitions, interacting only
  through self-attention. ConvSSM (Smith et al., NeurIPS 2023, abstract): tensor-valued (per-location) SSM states with
  convolutional transitions; long-horizon Moving-MNIST, DMLab, Minecraft, Habitat.
- Our own design already plans per-record slots of the working state (RECORDS v3, "SlotSSM-style").
- Route starvation (research pass 11-11): SWAX (Cabannes 2025) and Po et al. both found a local route (window, cache)
  leaves the recurrent memory unsupervised unless training removes the shortcut part of the time.
