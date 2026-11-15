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
