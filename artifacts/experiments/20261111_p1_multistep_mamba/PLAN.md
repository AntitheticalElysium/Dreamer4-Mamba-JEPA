# P1-MultiStep-Mamba: Multi-Step Mamba Belief Self-Consistency

Date: 2026-11-11  
Objective: Directly supervise Mamba's belief evolution across multiple horizons ($H \in \{2, 4, 8, 16\}$) to encourage it to retain persistent hidden-world information, particularly hidden timers (thirst, hunger, fatigue), without relying exclusively on the denoiser's one-step G1 gradients.

## 1. Context & Hypothesis

In P1-HUD-Change (11-09), the post-action belief reads health at linear $R^2$ 0.948 and drink 0.639 (exact MLP accuracy 0.99 and 0.35), and zombie readiness and hits are linearly present in it ($AP \in [0.78, 0.91]$). The awake timers remain absent (AP at the prevalence, about 0.04), and on DEV Episode 376 the world generates the unchanged digit in 127/128 futures in both the due and the not-due history.

**Hypothesis:** Because Mamba in P1 is only trained one step at a time with ground-truth observations injected at every single frame, it never develops internal recurrent clock dynamics. By training a blind student branch to predict future beliefs $b_{t+H}$ given only actions and a learned masked-percept token, Mamba is forced to internalize timer phase and counters into its recurrent state ($\text{conv}$ and $\text{SSM}$).

## 2. Architecture & Objective Specification

Keep F4's architecture, full-frame change grounding (from 11-09), and all existing training objectives (G1, G2, G3). Add only the following:

1. **Observed Teacher:** Run Mamba normally over a real chunk, obtaining observation-informed beliefs $b_0, \dots, b_L$.
2. **Blind Student:**
   - Fork Mamba's complete recurrent carry at time $t$ (immediately after percept $t$).
   - Advance through the next $H$ recorded actions without intermediate observations. Process action $t$ first. Do not double-process the starting percept $pt_t$.
   - Interleave actions with a single learned `mask_token` $\in \mathbb{R}^W$:
     $$x_{\text{student}} = [a_t, pt_{\text{mask}}, a_{t+1}, pt_{\text{mask}}, \dots, a_{t+H-1}, pt_{\text{mask}}]$$
   - This maintains Mamba's 2-token cadence, short-convolution window ($d_{\text{conv}}=4$), and RMSNorm residual stream.
3. **Target:** The teacher's observation-informed belief at $t+H$, $b_{t+H}$, with `stop_gradient`.
4. **Phase Alignment:** Compare the student's post-mask belief at $t+H$ ($\hat{b}_{t+H}$, output at token $2H-1$) directly against the teacher's post-percept belief at $t+H$ ($b_{t+H}$).
5. **Loss:** Plain Mean Squared Error (MSE) between belief vectors:
   $$\mathcal{L}_{\text{multi}} = \frac{1}{|B_{\text{valid}}|} \sum_{i \in B_{\text{valid}}} \|\hat{b}^{(i)}_{t+H} - \text{sg}(b^{(i)}_{t+H})\|_2^2$$
   Weighted by a modest fixed weight $\lambda_{\text{multi}} = 0.1$.
   No cosine similarity, no extra predictor head, no uncertainty heads, and no representation regularizers.
6. **Horizons:** Sample $H \in \{2, 4, 8, 16\}$ on a logarithmic scale. Single sampled endpoint target per update.
7. **Gradient Flow:** Do NOT detach the starting carry. Gradients propagate forward through the student rollout AND backward through the starting convolution/SSM carry into the historical percept-processing path prior to time $t$.
8. **Episode Boundaries:** If an episode reset (`first[i, t : t + H + 1].any()`) occurs anywhere in $[t, t+H]$, that stream is masked out (`valid_rollout = False`).

## 3. Preflight Protocol (`checks.py` -> `checks.json`)

1. **Learned Resume Determinism across Bootstrap Start:**
   - 4-update straight run vs 2+2 resumed run across update 2.
   - Strict bitwise equality on streams, carry, cache, state, rng, chooser, gaze bytes.
   - Optimizer state and model parameters $< 1\times 10^{-11}$.
2. **Update 1 Alignment vs F4 / HUD-Change Control:**
   - Verify bitwise identical `g1`, `tsp`, `rec`, `noise`.
   - Verify `multi` loss is computed, non-zero, and bounded.
3. **Launcher Process Test:**
   - Subprocess execution of `train_multistep.py` for 2 updates with `--no-checkpoints`.
4. **Sustained Benchmarks & Memory Profiling:**
   - `sustained64` (200 updates @ 64 steps): throughput and monitor loss.
   - `sustained64_bootstrap` (100 updates @ 64 steps, bootstrap on).
   - `phase256` (3 updates @ 256 steps).
   - `phase256_bootstrap` (3 updates @ 256 steps, bootstrap on).
   - Verify peak memory $\le 2.8$ GiB on RTX 3060 (6 GiB card).
   - Projected training time: $\sim 14.5$ - $15.0$ hours.
