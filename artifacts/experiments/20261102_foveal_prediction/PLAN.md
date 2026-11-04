# F1: predictive value of fine evidence and exported memory

2026-11-02. User authorized the information-gap tests. This is the predictive-value
stage in ../20261102_foveal_information/DISCUSSION.md, not a routing repair. The frozen
F1.5 checkpoint and native architecture remain unchanged. Discuss a route continuation
only after this stage; no automatic new architecture/world training follows.

## Question and matched instruments

Does current fine evidence improve prediction of the actual next percept beyond full
coarse/history/prior/action information? Does its native mixed or exported representation
provide the same practical advantage? Current entity recognition is not the target here.

Use the previous conditional campaign's train400/dev250 real-history datasets, pinned to
F1.5ck10. Fit episodes0:360, validate360:400, excluding world-monitor episodes; development
episodes250:400 provide an additional episode-disjoint confirmation cohort. The first250
dev episodes are an inspected diagnosis bank, not untouched confirmation. Final split
untouched. Source queries/regions are defined by the current crop, never future gains.

Baseline inputs: all81 current coarse tokens, current lighting, native predicted belief
BEFORE this crop, previous8 coarse percepts and their prior beliefs/actions/outcomes,
current arrived reward/termination, next physical action a[t], and current fixation/mask.
No corrected/post-action belief in the baseline. Real past fine history is matched through
the prior; only current fine access changes. No next-frame percept/outcome is an input.

Five supplements, padded to the same1024 dimensions:
* base: zeros;
* fine: WHOLE current9x64 crop, not an oracle-chosen token;
* input: native fine contribution to the256-D mixed percept embedding;
* predicted: native256-D post-action emission (after observing current frame and a[t]);
* memory: native4x256 memory projection of that emission.

Two independent MLP256-256 predictors with common architecture across arms predict
(1) next81x16 coarse tokens, as a delta from CURRENT coarse tokens; (2) next9x64 fine
tokens at the CURRENT fixation. The latter is a fixed screen query known before the
future, not tracking an object using simulator state. It is a future forecast from past
evidence, NOT the cascade's glimpse forecast conditioned on next-frame coarse tokens.
Do not conflate them. No future coarse tokens enter either predictor.

Matched seeds611701/611702,2,400 AdamW updates,batch512,lr1e-3,wd1e-4,clip1 per predictor;
same initialization and sampled frame IDs across arms of a seed. Each predictor has
its own mean squared error, parameters and clipping; there is no coarse/fine multitask
tradeoff inside a shared body. Their disjoint losses are computed together for efficiency.
All encoder/belief/targets frozen. This clarification was written before fitting.
This is an accessible-prediction instrument, not a changed P1 objective or sampler.
Read final weights; validation descriptive, no budget extensions after scientific nulls.

## Targets, support and primary readings

Re-encode the real NEXT frame at current fixation for the fine target. Coarse target
comes from the exact native coarse collection. Include terminal successors; no post-end
frames. Retain current crop, one-ring dilated crop, and whole-frame coarse errors.
All arms score identical regions, independent of their predictions. Store per-cell MSE,
not only bulk averages. Source-local areas are screen regions, not oracle future positions.

Primary population = night-time current crops containing an entity and no camera scroll
on the next transition. Broader all-frame, day/night, entity/empty, scroll/non-scroll
cohorts remain reported. Entity labels define reporting strata only. No semantic labels
are optimization targets. State changes/mob movement inside the crop remain stochastic.

For each instrument seed, paired episode-bootstrap95% intervals of candidate-versus-base
error reduction on (a) current crop's next coarse tokens, (b) dilated crop, (c) next fine
tokens. Positive raw-fine benefit establishes predictive value for that target/moment.
Compare intermediate supplements on the SAME frames, including paired raw-fine versus
memory differences. Do not use the eight strongest previous probe-selected scenes as a
population estimate. Also report copy-current coarse and copy-current fine references.

The primary cohort deliberately asks about fine-relevant cases; all-frame rates provide
the population context. No claim about complete conditional distributions from an MSE
test: a null may reflect instrument limits or information affecting higher moments.
Intermediate-reader nulls do not prove irreversible erasure. Even positive raw-fine
versus negative-memory results localize accessibility, not a unique cause in Mamba.

## Mechanical qualification

Full crop supplied to the fine arm; shared fixation/mask; dimensions/capacity equal;
fine/body target branch independent of coarse target; frame t+1 never enters features;
post-action memory corresponds to current a[t], not the following transition; current
targets reconstructed from the identical checkpoint and observed frames. Current and
next crop differ only in time, not fixation. Epoch reset history masked. Check the
head can exactly express copy-current coarse and starts there; no zero-output test of
input connectivity. Archive all fits/predictions/losses and source/data hashes.

## Additional local reading declared during fitting, before development predictions

Alongside the declared crop and dilated-crop readings, score the cells currently
occupied by an entity inside the observed crop and their one-ring neighborhood,
for BOTH future coarse and fine targets. This prevents improvement on static background
inside an entity-containing crop being mistaken for improvement on its entity region.
These are secondary source-defined masks, not a change to the primary gate, training,
population or future-dependent target selection. The current training logs had been
seen; no development predictions/primary effects had yet been read when added.
