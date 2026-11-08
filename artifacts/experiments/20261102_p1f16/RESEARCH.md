# Spatial evidence fusion: source review and implementation boundary

Read before P1-F1.6 build/launch,2026-11-02.

## Attentional Feature Fusion (Dai et al., WACV2021)

Paper: https://arxiv.org/html/2009.14082 (method3.2,4.1–4.3, ablation5.1).
Author implementation: https://github.com/YimianDai/open-aff .
Read the contributed PyTorch implementation at
https://raw.githubusercontent.com/YimianDai/open-aff/master/aff_pytorch/aff_net/fusion.py .
It confirms pointwise local transforms and the global pooling branch; its AFF
output scales both weighted branches by2, unlike the unscaled paper Eq4.
We do not silently adopt that implementation difference or its batch norms.

The local branch preserves spatial coordinates and uses two pointwise channel
transforms with a nonlinearity. It complements a global branch, and predicts
content-dependent feature weights from aligned maps. AFF uses a weighted
combination; iterative AFF refines the initial fusion. The authors explicitly
study the initial addition as a possible fusion bottleneck. Their global+local
variant outperforms local-only in their reported tasks. CIFAR experiments used
20 superclasses, a disclosed limitation; no world-memory validation is provided.

Transfer: shared nonlinear processing of aligned evidence before pooling. Our
per-cell joint MLP is an adaptation, NOT AFF: no global attention branch,
batch normalization, iterative gate or convex mixture. We do not transplant
those additional changes into this one-arm comparison. Existing lighting is
small global context; it is not an AFF global branch.

## CF-ViT (Chen et al.,2022)

https://arxiv.org/html/2203.03821 , fine inference/feature reuse, Eq11–14.

An MLP transforms coarse output tokens. The transformed tokens are spatially
replicated and selected to match split fine patches, then added residually to
fine input tokens. This supports explicit correspondence before downstream
integration. Their coarse-to-fine residual and our joint fine/coarse representation
are different parameterizations. Their classification/teacher loss
and adaptive patch splitting are not adopted or evidence of our acquisition.

## FiLM (Perez et al.,2018)

https://arxiv.org/html/1709.07871v2 , Eq1–2, model2.2 and residual ablations4.3.

Conditioning can alter features through content-dependent transformations while
retaining residual paths. Original FiLM predicts feature-wise scales/shifts
from language, typically globally across a feature map. Our local joint MLP
does not implement FiLM's exact parameterization; it learns joint features from
both coarse and fine evidence, with lighting, and shares it across locations.

## What is deliberately our own choice

The approved revision is88->128->64 GELU, operating on the nine observed locations,
with ordinary nonzero initialization and the existing out_map token normalization.
It replaces the raw fine-to-belief path, preserves the coarse pathway, and avoids
requiring fine evidence to squeeze into a16D correction. The earlier optional
zero-initialized16D residual prototype was rejected before training. Width128
is modest headroom, not a literature optimum. There is no paper
claim that these pieces automatically fix unsupervised recurrent fine retention.

Brain research discussed in the prior review supports correspondence-aware
integration, not this exact module. Coarse/fine views share pixels and noise:
the network does not multiply them as independent likelihoods. A from-scratch
matched world comparison is the empirical test.
