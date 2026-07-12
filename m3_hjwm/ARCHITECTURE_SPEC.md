# M3-HJWM architecture specification

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

Four register tokens are added, yielding 68 streams. Context tokens are
multi-block masked. Target tokens remain dense.

The model logs target variance and effective rank. Anti-collapse regularizers
such as SIGReg/Barlow Twins are not enabled until diagnostics justify them.

## 4. Spatial model

A small spatial attention stack acts within each frame. This preserves 2D
interactions and keeps the temporal SSM from having to rediscover spatial
geometry.

## 5. Temporal model

The target backend is Mamba-3. Mamba-2 and GRU implement the same contract:

- `forward_sequence([B,T,S,D]) -> [B,T,S,D]`
- `step([B,S,D], cache) -> [B,S,D], cache`
- explicit reset semantics.

Actions are injected before the temporal update. A production Mamba-3 recurrent
adapter must be pinned to the exact official package/kernel version and pass
step/scan equivalence tests.

## 6. Future predictor

Default: hard-assigned mixture of K predictors.

\[
\hat Y^{1:K}_{t+1}=P^{1:K}(C_t,a_t).
\]

Training chooses:

\[
k^*=\arg\min_k d(\hat Y^k_{t+1},Y_{t+1}),
\]

and trains the selected predictor plus a router that predicts `k*` from context.
A weak global usage penalty prevents dead modes.

This avoids the regression-to-conditional-mean failure while generating all
modes in one forward pass. It is not an uncertainty ensemble. Mode spread is
mostly aleatoric until calibrated otherwise.

Deterministic prediction remains an exact control.

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
+\lambda_P L_{\text{manifold-projector}}
+\lambda_E L_{\text{energy}}.
\]

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
- target effective rank,
- one/multi-step held-out latent error,
- reward and continuation calibration,
- reliability calibration,
- imagined versus real return from identical prefixes.

## 12. Explicit non-claims

- Mamba-3 is not assumed faster at Crafter sequence lengths; benchmark it.
- Mode dispersion is not automatically epistemic uncertainty.
- JEPA representation non-collapse does not prove control sufficiency.
- "On Training in Imagination" supplies sufficient-condition bounds, not a
  certificate that a neural model is safe.
- The architecture is mathematically specified, not mathematically guaranteed to
  outperform Dreamer 4.
