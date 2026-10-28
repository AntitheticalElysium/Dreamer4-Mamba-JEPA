# Independent M0 implementation audit

Targets, in order: design closure e81302d6; implementation 17c2a39c; evaluator f4d1b672.
No production P1, P2, or actor run. Canonical files remain unchanged. Evidence and
bounded reproduction scripts go here. Existing unrelated changes are preserved.

Before additional experiments, the separating checks are:

1. Independently run all existing tests with actual CUDA visibility. Record skips,
   runtime, numerical tolerances and exactly what each test establishes.
2. Validate split membership, frame/action/outcome alignment, target-array metadata
   and teacher hashes. Recheck vectorized teacher features against its original
   loop on held-out real frame pairs. No simulator labels enter training.
3. Check both long-context and reset paths, including recurrent-state and parameter
   gradients. Extend cache parity beyond the four-frame window and eight-frame reach;
   check branch/cache checkpoint round trips, not just a short reset example.
4. Test the deployed denoised-output map against actual encoder targets and a
   mathematically perfect conditional denoiser. Distinguish objective/sampler
   incompatibility from learned approximation error; do not infer actor performance.
5. Inspect BF16 and FP32 training/sampling differences, activation-checkpoint gradient
   parity, carried-state ownership, optimizer grouping, frame-indexed schedule and
   saved-state provenance. Use bounded checks, never extend the training budget.
6. Audit every evaluator section at full input shapes. Verify categorical missing
   support cannot silently pass a blocker; validate observable memory labels and
   available history in each control/C2ST tuple; check probe evaluation memory.
7. Inspect original reward-change evidence; recover the original implementation if
   present. Compare like-for-like bin probabilities/sampled values. Distinguish
   gradient clipping from actual Adam parameter updates and a practical workaround
   from a universal claim about logistic mixtures.
8. Verify external claims against JEDI, Dreamer 4, EDM, PixelCNN++, HarmonyDream,
   I-JEPA/V-JEPA 2 and classifier two-sample testing, using primary sources only.
9. Quantify real throughput only after correctness checks if still needed; do not
   treat reproduction of a timing, overfit loss, or passing test as scientific success.

Findings must state: measured defect, bounded numerical counterexample, deliberate
design adaptation, implementation gap, or unverified historical claim. No result
will trigger an automatic redesign, campaign, or actor launch.
