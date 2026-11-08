# Global percept projection: origin, scope and source audit

2026-11-03. Read against live code, the original design commit and primary papers.
This records facts and hypotheses; it does not approve a replacement architecture.

## Verified implementation and history

- `saccade/model.py:Belief.percept_embed`: 81x16 coarse tokens, nine64D fine
  tokens placed on an81x64 canvas,81 mask entries and8 lighting entries feed
  one6569->256 linear map. In F1.6, a local88->128->64 nonlinear joint transform
  replaces the nine raw fine tokens first. Only576 fine coordinates are active
  at one fixation;6569 is the padded input size, not independent content count.
- `saccade/design/M0_CODE.md`, present already in commit`fc120c63` (2026-10-27),
  declares a flattened linear percept token. The same commit's`m0.yaml` cites
  DRAMA for`percept_projection: flatten_linear`. First implementation`17c2a39c`
  already has the positioned canvas and linear mixing. It was not introduced
  by factorisation or the F1.6 fusion experiment.
- `STEP_SPEC.md` section7 explicitly calls this position-sensitive and lossy.
  The minimal design feeds one percept and one action token per frame; width256
  and the six-layer recurrence inherit the earlier LeWM-Raw Mamba recipe.
  Position-specific canvas weights are a Saccade adaptation, not DRAMA's exact
  observation interface. This explains the compact adapter's role; it is not
  empirical evidence that the adapter is sufficient for fine visual memory.
- `Belief.forward` processes the vector through residual Mamba layers. The
  persistent convolution/SSM carry is MUCH larger than the256D emitted vector.
  A weak emission readout cannot establish that the internal carry is empty.
- `Belief.memory` linearly expands the emission256->1024 and reshapes into four
  tokens. On the selected F1.6 checkpoint that matrix has full column rank256,
  condition14.0107, singular values0.39034..5.46892. In exact arithmetic this
  expansion does not discard emitted-vector dimensions. This says nothing about
  losses before emission or rounding/readability/consumer use after expansion.

## Papers and upstream code

**DRAMA, ICLR2025, section2.2.** The model learns a categorical scene latent with
image reconstruction, concatenates its sample with the action, and projects
that vector into Mamba. The vendored implementation's`MixerModel.stem` is
Linear->RMSNorm->SiLU. This is evidence for a compact frame interface, not for
compressing an independently trained coarse grid plus movable fine crop through
our screen-specific canvas. Its default Mamba width is512, not our256.
Paper: https://arxiv.org/html/2410.08893 . Upstream code:
`third_party/sources/realwenlongwang__Drama/mamba_ssm/models/mixer_seq_simple.py`
and`sub_models/world_models.py` (Encoder, DistHead, sequence-model calls).

**Po et al., ICCV2025, sections4.1–4.2.** The video model retains spatial tokens
and uses separate SSM scans per spatial block, plus local frame attention. The
paper explicitly treats fixed-state expressivity and temporal/spatial tradeoffs;
block-wise scans improve its memory tasks, while changing long-context training
also matters. This supports questioning our spatial bottleneck and training
pressure together; it is not a matched test of our one-vector percept interface.
Paper: https://arxiv.org/html/2505.20171 .

**JEDI2026, section3.1.** Its diffusion model is conditioned on four16x8x8 spatial
latents and actions. JEDI supports reconstruction-free latent diffusion, but not
our separate256D recurrent belief adapter. A diffusion precedent is not evidence
for every additional Saccade interface.
Paper: https://arxiv.org/html/2605.13013v1 . Local pinned TeX:
`third_party/paper_sources/2605.13013v1/main.tex`.

## Interpretation

The compact interface is an intelligible minimal-build choice, but its fit to
our movable fine evidence remains a design bet. A single linear map has no
content-dependent attention to choose what to retain at that stage; prior local
fusion and downstream nonlinear recurrence must do that work. Its screen-varying
fine slice also changes the coordinates the same crop content occupies.

Three broad failure sites remain: acquisition at mixing; storage/emission in
Mamba; use by the denoiser. They can coexist. Readout weakness, redundancy with
other inputs and objective pressure must be separated at each site. The frozen
projection ablation proves particular survival when recoding succeeds; it does
not train a better native route or establish that Mamba can decode it itself.
