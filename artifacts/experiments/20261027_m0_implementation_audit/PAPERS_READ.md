# Primary-source pass for the implementation review

Read for contract-level claims, not as a request to change the architecture.

1. **EDM (Karras et al., NeurIPS 2022)** — main preconditioning/training section,
   Algorithm 2, Appendix B.3's optimal denoiser and B.6's loss equivalence. Browser
   extraction could not handle the PDF size; downloaded the public arXiv PDF to
   /tmp and read its text. Raw F regression and sampling use the same D; M0's extra
   nonlinear output map is an additional operation. [Paper](https://arxiv.org/abs/2206.00364),
   [reference loss](https://github.com/NVlabs/edm/blob/main/training/loss.py),
   [reference sampler](https://github.com/NVlabs/edm/blob/main/generate.py).
2. **JEDI (2026)** — method 3.1-3.2, Eq. 8-9, algorithm/hyperparameter appendix.
   Detached noisy future and target; differentiable context; encoder LR .3;
   categorical outcome loss; clamp factor 3. It does not establish M0's combination
   of LayerNorm and tanh as an objective-independent sampler projection.
   [Paper](https://arxiv.org/html/2605.13013v1). Local pinned PDF hash matches
   PAPERS.lock; its title string in the lock is wrong, the PDF itself is correct.
3. **Dreamer 4 (2025)** — sections 3.1-3.4, especially frozen tokenizer,
   diffusion forcing, factorized temporal/spatial blocks and PMPO. Supports
   the backbone family and recurrent sampling pattern, not M0's joint encoder
   or specific sparse-memory conditioning. [Paper](https://arxiv.org/html/2509.24527).
4. **V-JEPA 2 / 2-AC (2025)** — section 3.1 and the actual frozen-encoder
   action-conditioned training path. Prediction normalization is applied before
   teacher-forced and rollout losses in train.py:417-440; wrapper normalization
   is therefore consistent with training. [Paper](https://arxiv.org/html/2506.09985v1),
   [training](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_droid/train.py).
5. **PixelCNN++ (2017)** — discretized logistic likelihood construction and
   official implementation, including bounded log scale, absorbing edge bins,
   and stable log probability. M0 uses a simpler single-component pixel model;
   its normalization and finite gradients were checked independently. The paper
   is not evidence that logistic mixtures are generally unsuitable for rewards.
   [Paper](https://arxiv.org/abs/1701.05517),
   [implementation](https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py).
6. **HarmonyDream (ICML 2024)** — section 3 and Appendix B's rectified
   harmonizer derivation. The formula matches; equilibrium balances loss scales,
   not per-route gradients. [Paper](https://arxiv.org/html/2310.00344v3).
7. **Classifier Two-Sample Tests (Lopez-Paz & Oquab)** — sections 2-3 and
   generative-model evaluation/feature-space discussion. Held-out testing is
   necessary, and interpretation is limited to supplied features. M0's missing
   history cannot be certified by a classifier that never receives it.
   [Paper](https://arxiv.org/html/1610.06545v2).

Additional primary implementation/documentation checks: DIAMOND reward head and
noise conditioning; installed Mamba-2 Triton/reference kernels; per-token norm
in vendored I-JEPA/V-JEPA; sklearn's threshold-based AP definition; official systemd
WorkingDirectory documentation and live harmless unit tests.

No new scientific performance claim is inferred from matching a component precedent.
