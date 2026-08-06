# Reward-head source correction and calibration boundary

Date: 2026-07-18

Status: **primary-source correction; no outcome claim**

## Correction

The evidence ledger previously called the local reward/continuation modules
“DreamerV3 heads.” That is too broad for the reward distribution.

The local reward head is structurally Dreamer/DRAMA-shaped—an MLP predicts a
255-bin categorical reward distribution from a post-transition feature—but
its exact target and decoding equations match DRAMA and the inspected
unofficial Dreamer-4 JAX implementation, not DreamerV3/CDP.

## Exact implementations

### Local / DRAMA / unofficial Dreamer-4 JAX family

Local code (`m3_hjwm_compact/model.py:169-190,940-958`) uses uniform support
\(z_i\) in symlog space:

\[
z_i \sim \operatorname{linspace}(-20,20), \quad
y = \operatorname{symlog}(r),
\]

interpolates \(y\) between adjacent \(z_i\), trains categorical cross-entropy,
and decodes:

\[
\hat r_{\mathrm{local}}
= \operatorname{symexp}\left(\sum_i p_i z_i\right).
\]

This matches:

- DRAMA commit `a50bd54c34e77d1d13e988a031733a47817098e2`,
  `sub_models/functions_losses.py:15-50`;
- unofficial Dreamer-4 JAX commit
  `8144b940d801971f12ec5633553b95001e555949`,
  `dreamer/models.py:764-806` and
  `scripts/train_bc_rew_heads.py:437-452`.

The JAX repository is explicitly an unofficial reproduction; it cannot define
the official Dreamer-4 implementation.

### DreamerV3/CDP family

Dreamer-CDP commit
`a851fa3e3d70b624b094ee1810ad4bb602346092` uses the DreamerV3
`symexp_twohot` head in:

- `embodied/jax/heads.py:132-144`;
- `embodied/jax/outs.py:273-319`;
- `dreamerv3/configs.yaml:100`.

It constructs support in original reward space:

\[
r_i = \operatorname{symexp}(z_i), \quad
z_i \sim \operatorname{linspace}(-20,20),
\]

interpolates the original target \(r\) between adjacent nonlinear \(r_i\),
and decodes:

\[
\hat r_{\mathrm{D3}} = \sum_i p_i r_i.
\]

The source also uses a symmetric summation order so a symmetric categorical
distribution decodes to exactly zero.

These operators agree on a point target when its ideal two-hot distribution
is used, because both interpolations reconstruct that target. They are not
equivalent for a general predictive distribution:

\[
\operatorname{symexp}\left(\mathbb E[z]\right)
\ne \mathbb E[\operatorname{symexp}(z)].
\]

Therefore post-hoc application of the DreamerV3 decoder to a locally trained
checkpoint would not be a source-faithful DreamerV3 control; the target
operator would also need to change and the head would need retraining.

### Dreamer 4 paper

Paper `2509.24527v1.pdf`, SHA-256
`8655cce4bf12ce6210f6694f83c1a723c7acd7579214ca3ebc57c4394d0b1aeb`,
Section 3.3 and Eq. 9 says:

- reward prediction uses MTP length 8;
- heads have one output layer per MTP distance;
- reward uses a “symexp twohot” output;
- task embeddings condition agent tokens;
- video prediction continues while policy/reward heads are learned;
- the Minecraft task phase mixes 50% uniform and 50% relevant sequences,
  while the dynamics loss is restricted to uniform sequences.

The paper does not resolve the two implementation variants above in enough
detail to make the local operator “official Dreamer 4.” The local system also
lacks task-conditioned agent tokens and true MTP semantics.

The separate PyTorch repository at commit
`b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6` states in its README that it
is unofficial and incomplete; it does not supply an authoritative reward
head.

## Relevance to the current failure

The source divergence is material but not yet causal:

- Crafter replay rewards occupy only `[-.7, 1.1]`; 255 bins over symlog
  `[-20,20]` devote almost all output classes to unused magnitudes.
- Stage-2C C-LR has good branch ordering but a biased small nonzero reward
  baseline.
- Stage-2D proves that ordinary or generated-state refitting of the same local
  decoder on the C-L representation does not recover branch ranking.

These facts do not show that changing two-hot equations will fix the model.
They justify testing a lower-risk calibration of the already useful C-LR
distribution before changing its loss/support and retraining.

## Licensed calibration control

The next control keeps the trained local distribution and all world
parameters frozen. It may apply only:

- a global categorical temperature; and/or
- one additive bias on the exact zero-reward bin.

Both scalars must be fitted by unweighted two-hot NLL on a calibration set
separate from the evaluation set. This is a local post-hoc calibration
control, not a Dreamer reproduction.

If that control fails, a future source comparison must retrain matched reward
heads under:

1. the current DRAMA/unofficial-D4 symlog-space operator; and
2. the DreamerV3/CDP original-space support/interpolation/expectation.

It must not change the temporal trunk, sampling distribution, head capacity,
or evaluation set at the same time.
