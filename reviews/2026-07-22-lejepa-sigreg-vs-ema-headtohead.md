# Anti-collapse head-to-head: EMA vs LeJEPA/SIGReg (non-generative JEPA arm)

Date: 2026-07-22

## What was compared

The single moved axis is the **anti-collapse mechanism** of the non-generative
JEPA world arm. Everything else is held identical: the deterministic
action-conditioned next-embedding predictor (the rollout), the online encoder,
the reward/continuation heads (incl. terminal weighting D035), the multi-step
`jumps` (D034), actor/value, BC prior, pixel adapter, temporal operator,
schedules, and the sealed evaluation protocol (seeds `987000:987100` and
`988000:988100`, plus the BC-vs-anti-BC imagination probe).

- **EMA arm** (D031, committed `feefb87`): SPR/BYOL stop-gradient EMA target
  encoder + asymmetric projection/prediction heads.
- **SIGReg arm** (D036): LeJEPA, `rbalestr-lab/lejepa` commit `c293d291`. Drops
  the EMA teacher and the BYOL predictor/projection heuristics; the prediction
  loss is to the (non-EMA) stop-gradient online target; anti-collapse is
  Sketched Isotropic Gaussian Regularization — the sliced random-projection
  multivariate test (`SlicingUnivariateTest`, 1024 slices) with the **Epps-Pulley**
  univariate normality statistic, imported unchanged and digest-verified. A
  projector is kept (as in the paper); SIGReg + invariance act on its output.

## Result (numbers)

| | EMA arm | SIGReg arm (best of 7 variants) |
|---|---:|---:|
| Representation (BC executed return) | ~385 | **358–403** |
| Terminal signal decodable in latent (agent-token AUC) | 0.82 | **0.851** |
| Predictor action-sensitivity (action0-vs-1 cos, lower=better) | **0.987** | 0.992–0.9995 |
| Imagined good−bad divergence (probe) | **+4.5** (continue 0.92 vs 0.12, clean) | +1.1 to +1.65 (noisy, continue-min inverted) |
| **Imagination vs BC, sealed 987** | **+30.84, CI [2.9, 59.4]** ✓ | **−26.22, CI [−64.1, 11.9]** ✗ |
| **Imagination vs BC, sealed 988** | **+53.12, CI [21.9, 85.9]** ✓ | **−9.43, CI [−50.9, 32.1]** ✗ |

**EMA imagination beats BC on both sealed tiers; SIGReg imagination does not
(it ties, CI spans zero).**

## The attributable finding

Swapping the anti-collapse from EMA to SIGReg **keeps, and slightly improves,
the representation** (BC parity; the pole-fall signal is *more* linearly
decodable from the SIGReg latent, AUC 0.851 > 0.82) but **removes the usable
imagination**: the world model becomes action-insensitive, so imagined return
does not track policy quality and PMPO cannot improve BC.

The mechanism is attributable to the single moved axis. The EMA teacher's slow,
lagging target forces the predictor to model where the representation is *going*
(temporal, action-dependent), yielding an action-sensitive world model
(action-cos 0.987) and a clean imagined divergence (good 0.92 vs bad 0.12
survival). SIGReg's non-EMA target has a trivial action-independent solution on
CartPole's slow dynamics — the next embedding is predictable *without* the
action — so the predictor ignores the action (action-cos ≥ 0.99) and the
imagined rollout does not diverge for good vs bad control. LeJEPA is designed
for **view-invariance** SSL (predicting across augmentations of the same image,
where there is no trivial temporal shortcut); the temporal world-model
prediction task exposes a shortcut that SIGReg's distributional regularization
does not block, while the EMA teacher does.

**Conclusion: the anti-collapse mechanism is load-bearing for imagination-based
control. A better *representation* (SIGReg) does not imply a better *imagination*.**

### Consistent with the LeJEPA paper (2511.08544v3), not a contradiction

The paper's objective is to make embeddings an isotropic Gaussian that minimizes
downstream *probing* risk (linear/nonlinear probe accuracy, §3–4) — the metric
it optimizes and validates. It never claims, or optimizes for, action-conditioned
multi-step **rollout** fidelity (imagination). Our result matches the paper on
what it promises: SIGReg delivers the *better representation* (BC parity, higher
terminal-AUC). Faithfulness is confirmed against the paper as well as the code:
Definition 1 (`Enc(x_{t+1})` predictable from `Enc(x_t)`, non-degenerate) holds,
and §2.2 explicitly sanctions the action-conditioned predictor `Pred` "when
there exists an asymmetry between views, e.g., by conditioning on observed
actions" — so keeping it is prescribed, and the dropped "predictor heuristic" is
the BYOL anti-collapse head, which we removed. The EMA teacher's lagging target
enforces temporal/action-sensitivity as a *side effect* that helps imagination;
SIGReg's distributional regularization does the embedding job (better) but not
that dynamics job.

## Faithfulness and exhaustiveness

Seven faithful variants were run to convergence, all giving the same qualitative
result (excellent representation, action-insensitive predictor, imagination that
does not beat BC): SIGReg on the raw latent vs a projected space; prediction
loss as raw-MSE / projected-MSE / raw-cosine / projected-cosine; SIGReg weight
λ ∈ {0.02, 0.05, 0.1}; per-token vs per-frame embeddings; and post-hoc
continuation-head recalibration. The predictor action-cos never fell below the
EMA arm's 0.987 in any variant. The SIGReg import is the pinned `c293d291`
source unchanged (drift-verified); the teacher and predictor heuristics are
dropped as the paper prescribes; the normality test is Epps-Pulley.

Making SIGReg's imagination *beat* BC would require relaxing a held-fixed
setting (a longer imagination horizon / more `jumps`, or an inverse-dynamics
auxiliary to force action-aware latents, or a non-EMA-but-lagging target) — each
of which departs from "anti-collapse is the only thing that moves" / "no
hybrid", and so is a decision for the maintainer, not an implementation change.
