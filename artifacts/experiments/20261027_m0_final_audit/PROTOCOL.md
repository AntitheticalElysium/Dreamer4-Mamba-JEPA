# Final pre-build audit of M0 at fc120c63

Scope: review the declared architecture and implementation contracts. No M0 training,
canonical design changes, or new architecture comparison is authorized by this audit.

Predeclared checks, before executing this audit's scripts:

1. Trace diffusion-forcing gradients on a two-time-step algebraic example. Compare
   label-only detachment against detachment of all noisy inputs. This is an interface
   counterexample, not a learnability experiment.
2. Check equivalence of raw-network EDM regression and weighted clean-x regression;
   quantify the weighting range under the declared noise bounds.
3. Use the existing reward calibration bin function to find the interval containing
   zero and the reward emitted by a bin-centre decoder. No reward sampler exists yet;
   this checks whether the planned one-bin-mass test would be sufficient.
4. Compute exact coverage of the declared zero-padded, centred 3x3 patch score.
5. Read the committed acuity JSON; check the original two-clause selection rule,
   reported per-kind recalls, and availability of per-kind precision references.
6. Profile the actual pinned SIGReg reduction on CUDA, N=4096 for each of the planned
   16- and 64-dimensional token families, including backward. No model-fit claim will
   be made from an isolated regularizer measurement.
7. Run the existing saccade evaluation regression tests. Record source hashes and
   distinguish existing implementation failures from unspecified future behaviour.

Additional contract inspection: complete sampler cache lifetime (not merely a fixed
noise-level forward); overlapping context prefixes versus carried recurrent state;
P2 head training; matched belief-probe controls; actor/BC comparison specification.

No audit result changes the declared training budgets or the chosen 3x3 acuity.

Follow-up declared after the isolated SIGReg profile (1913.8 MiB allocated peak):
compare the canonical statistic with averaging eight 128-projection reductions,
each activation-checkpointed, using the SAME pre-generated projection directions.
Check value and input-gradient parity to relative tolerance 1e-5, absolute 1e-6;
profile memory and elapsed time. This changes reduction scheduling only, not the
sample count, distribution, projections, weight, or scientific treatment.

Follow-up on the installed Mamba kernel: its state-passing implementation starts
the incoming carry's sequence index at zero. Compare a nonzero incoming state
with no seq_idx, constant zero seq_idx, and constant nonzero seq_idx, plus a
zero-state reference. This checks the sequence-id convention a future port must
respect; no canonical Mamba code is changed.

Concurrent design revision: e1b4cb40 appeared during the audit, replacing SIGReg
with token BatchNorm and fovea reconstruction and clarifying frame-input stop-gradients.
The original review remains pinned to fc120c63. Follow-up on the new normalization:
change only future tokens and test whether earlier train-mode normalized tokens
change; inspect future-token gradients; compare fixed-stat eval mode; check the
post-tanh bound. These are mathematical contract checks, not learning experiments.
