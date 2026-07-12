# M3-HJWM architecture specification (verification revision)

Status, 2026-07-13: this document is a falsifiable target, not a claim that the
model is ready for policy training. The representation objective still fails at
least one binding Phase B gate. Mamba-3, predictor mixtures, reliability
weighting, and online policy training are all off the critical path or no-go.

## 1. Scope

The target is a small, online model-based RL agent for Crafter that can scale
later without making Crafter-specific assumptions in the model. Crafter enters
only through:
- RGB observations,
- a discrete action count,
- reward,
- episode continuation.

The model does not reconstruct pixels.

## 2. State and timing

Transition convention:

\[
(s_t, a_t) \rightarrow (s_{t+1}, r_{t+1}, c_{t+1}).
\]

Every loss and buffer follows this convention. Reward and continuation heads
consume the post-transition state. This directly prevents the prior project's
off-by-one imagination bug.

## 3. Representation

Context encoder:

\[
X_t = E_\theta(o_t) \in \mathbb{R}^{64 \times d}.
\]

EMA target:

\[
Y_t = \mathrm{sg}(E_\xi(o_t)),\quad
\xi \leftarrow \mu \xi + (1-\mu)\theta.
\]

Two register tokens are added, yielding 66 streams. The default world-model
path is unmasked. Post-convolution token replacement remains a separately
runnable ablation; it is not I-JEPA masking and it creates a train/deploy input
mismatch.

The old flattened anti-collapse diagnostic is invalid: fixed position codes can
have high rank while ignoring the observation. Any candidate regularizer and
gate must compute statistics across observations independently at each stream,
and must also report pooled rank, fixed-stream variance, observation/position
variance decomposition, same-stream unrelated distance, semantic retention,
and inventory retention. VICReg/SIGReg ports are candidates, not validated
defaults; the corrected controls have not yet found a setting that preserves
both information and predictive fidelity.

## 4. Spatial model

A small spatial attention stack acts within each frame. This preserves 2D
interactions and keeps the temporal SSM from having to rediscover spatial
geometry.

## 5. Temporal model

GRU is the correctness/default backend while the representation is unresolved.
Mamba-2 is the research backend implementing the same contract:

- `forward_sequence([B,T,S,D]) -> [B,T,S,D]`
- `step([B,S,D], cache) -> [B,S,D], cache`
- explicit reset semantics.

Actions are injected before the temporal update. Mamba-2's official recurrent
cache and step/scan equivalence are verified at the pinned commit. Mamba-3 is a
no-go on the RTX 3060: its recurrent path is non-finite/inconsistent and the
official source notes it was only tested on H100. No silent backend fallback is
allowed.

## 6. Future predictor

Default: deterministic cross-token attention predictor.

\[
\hat Y^{1:K}_{t+1}=P^{1:K}(C_t,a_t).
\]

Training chooses:

\[
k^*=\arg\min_k d(\hat Y^k_{t+1},Y_{t+1}),
\]

and trains the selected predictor plus a router that predicts `k*` from context.
A weak global usage penalty prevents dead modes.

The hard-assigned mixture remains an exact, separately runnable control. Its
mechanics work on a synthetic two-mode process, but typical Crafter successor
dispersion is small and rare reward-relevant branches have not been sampled
adequately. It is not an uncertainty ensemble. Mode spread is mostly aleatoric
until calibrated otherwise.

A shortcut-flow predictor can later implement the same predictor interface if
mode mixtures prove insufficient.

## 7. World-model objective

\[
L =
\lambda_J L_{\text{JEPA}}
+\lambda_M L_{\text{mode}}
+\lambda_B L_{\text{balance}}
+\lambda_R L_{\text{reward}}
+\lambda_C L_{\text{continue}}
+\lambda_V L_{\text{stream-var}}
+\lambda_{\mathrm{cov}} L_{\text{stream-cov}}
+\lambda_{\mathrm{roll}} L_{\text{rollout}}
+\lambda_P L_{\text{manifold-projector}}
+\lambda_E L_{\text{energy}}.
\]

The two-step rollout bridge is adapted from V-JEPA 2-AC Eq. 3-4:

\[
L_{\mathrm{rollout}} =
d\!\left(P(a_{t:t+1}; C_t),\,Y_{t+2}\right).
\]

It feeds the first generated successor through the exact temporal deployment
composition and differentiates through predictor → temporal → predictor. The
visual prefix is detached for this auxiliary, matching V-JEPA 2-AC's frozen
encoder post-training assumption. Deviations are explicit: cosine replaces L1,
and the compact temporal core is separate from the spatial predictor. Its weight
defaults to zero until the corrected representation gate and multi-seed fixed-
representation comparison pass.

No loss is allowed to decode an action from a hidden state that already consumed
that action. An optional inverse objective must use `(Y_t,Y_{t+1}) -> a_t`.

## 8. Imagination

Start from a world state inferred from a real replay prefix. For each imagined
step:

1. `a_t ~ pi(. | s_t)`
2. sample/select one successor mode from `P(C_t,a_t)`
3. feed the full selected successor consistently into the temporal step
4. obtain `s_{t+1}`
5. predict `r_{t+1}, c_{t+1}` from `s_{t+1}`
6. repeat

One mode is selected for the complete successor state. The old design's
per-step/per-element random RPF head switching is forbidden.

## 9. Actor and critic

Lambda returns:

\[
G_t^\lambda = r_{t+1} + \gamma c_{t+1}
[(1-\lambda)V(s_{t+1}) + \lambda G_{t+1}^\lambda].
\]

The initial implementation uses a categorical score-function actor because the
actions and hard mode choices are discrete. A pathwise continuous-action version
can be added later.

## 10. Hallucination detection

Internal Mamba parameters are not used as gates.

Four shadow signals are logged:

1. successor-mode dispersion,
2. context/action/future compatibility energy,
3. target-manifold projection residual,
4. value-ensemble disagreement.

A reliability predictor is supervised with actual held-out multi-step target
error. It must be evaluated for calibration, AUROC, and error correlation before
it may influence actor or critic losses.

After calibration:

\[
w_t = \exp(-\hat e_t / T)
\]

may softly weight actor/critic losses. Hard truncation is a later ablation.

## 11. Required tests

- exact action/reward indexing,
- scan/step equivalence,
- reset isolation,
- no future leakage,
- no action-label leakage,
- no stale cache across world-model updates,
- mode precision and mode usage,
- flat rank plus observation-sensitive stream/pooled rank and variance,
- one/multi-step held-out latent error,
- reward and continuation calibration,
- reliability calibration,
- imagined versus real return from identical prefixes.

## 12. Explicit non-claims

- Mamba-3 is unsupported on this hardware; Mamba-2 is not assumed faster.
- Backend fidelity comparisons require one frozen shared target representation;
  independently trained latent spaces are not numerically comparable.
- Mode dispersion is not automatically epistemic uncertainty.
- JEPA representation non-collapse does not prove control sufficiency.
- "On Training in Imagination" supplies sufficient-condition bounds, not a
  certificate that a neural model is safe.
- The architecture is mathematically specified, not mathematically guaranteed to
  outperform Dreamer 4.
