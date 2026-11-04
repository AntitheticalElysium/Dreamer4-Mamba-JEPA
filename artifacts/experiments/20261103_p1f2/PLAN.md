# P1-F2: learned fixation on the unchanged F1.6 world

2026-11-03. Preparation authorized; **overnight launch NOT authorized yet**.
The user explicitly retained the original combined-score softmax. No training
service or timer is installed by preparation. Mechanical smokes are disposable,
not short efficacy gates. Negative results do not suppress full evaluation.

## Scientific question and matching

Does endogenous learned fixation improve the F1.6 spatial-fusion world when learned
jointly from scratch? Control: `../20261102_p1f16/run/ckpt_10.pt`, SHA256
`34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045`.

World cfg must equal the actual saved F1.6 cfg, byte-for-value, and world seed=1.
Same mandatory joint64 spatial fusion, capacity, initialization, losses (including
faithful pixel objectives), harmoniser, optimizer/clipping, corpus, data order,
monitor, 25.6M-frame budget, 16 streams, 64/256 chunks and accumulation, cascade
sampler. No data repair, extra supervision, P2, actor, or architectural repair.
Only gaze acquisition changes. This is one matched seed, not seed-robust evidence.
The head's auxiliary objective does not train the world or enter its clipping.

The canonical trainer has no controller hook. `trainer.py` is a compact local copy
with six explicit lifecycle hooks; an AST test proves that removing these hooks
recovers the canonical loop. World loss/monitor/diagnostics are calls to the
original functions, not rewritten implementations. No canonical source changes.

## Adopted chooser and causal timeline

Use the ORIGINAL 101,584-parameter Head in `../20261102_direct_chooser/direct.py`,
not E3's 118,032-parameter ablation head. Eight coarse frames, lighting, previous
actions/rewards/termination/validity; no Mamba belief or auxiliary sketch route.
Head outputs last coarse frame + learned residual. At frame t:

```
(p,v) at t-8..t-1 + a[t-1] -> cheap forecast of p[t]
               same history + NOOP -> counterfactual forecast
current coarse p[t] is observed -> C and S maps -> sample eye -> actual crop
                     original full P1 objective -> original world update
                     detached coarse examples -> separate head update
```

C is squared forecast difference, S squared forecast error, summed across 16
channels and each 3x3 window. Normalize each 49-window map by its mean (zero map
stays zero). Sample from **0.8 softmax(0.5 norm(C)+0.5 norm(S)) + 0.2/49**, T=1.
No separate probability mixture, argmax, hand placements, reference model, learned
noise floor, adaptive weighting or recency trace. Frame zero stays uniform.

Forward prediction sees neither current p/v/reward/termination nor the crop.
Surprise deliberately sees current coarse p; this is the approved same-frame
periphery-before-fovea timeline. No current fine pixels, simulator labels,
achievement fields, event masks, or dev labels reach the chooser. The encoder's
existing light/periphery operations are evaluated without the crop; full P1
perception is recomputed with gradients as before.

History resets at episode boundaries. Coarse history is re-encoded with current
weights, not fetched from stale targets. Head is frozen while a chunk's gazes are
chosen. Its update uses only that chunk AFTER selection/world update; it cannot
train on an example before selecting its look. This batches a coarse-only policy
without any sequential fine-dependent belief collection.

## Explicit online learning clock (new working choice, not a settled past result)

`chooser.json` declares every setting. Fresh head seed611601 (first declared
research seed, not chosen for best performance); zero residual output init.
AdamW, lr1e-3 to1e-4 cosine, 100 head-update warmup, wd1e-4, clip1, B256 match
the original head fit's optimizer family/settings. One head update per1,024 new
world frames:25,000 updates/6.4M sampled examples at the full budget. At256-step
chunks this means four updates per world update, preserving the same per-frame
rate. Minibatches sampled from current detached chunk targets; no replay cache or
pretrained head. No LR reset at phase/checkpoint boundaries.

The online rate is a prospective implementation choice; frozen-head E1/E2/E3 did
not validate it on an evolving encoder. Online loss, C/S magnitude, entropy,
window distribution and actual gaze trajectory are retained. Low MSE is not a
success criterion. A useful policy/world benefit must be demonstrated.

Head CPU init, gaze sampling and head minibatch sampling have isolated RNGs.
Original uniform stream draws are still consumed so episode reshuffling/data
order are identical to F1.6. New gazes replace only the fixation fields. Actual
past gazes are written back to Streams.hist, so reused prefixes never acquire a
new observation. World gradients/Adam state are independent of head gradients.

## Checkpoints, resume and selection

Ten numbered world checkpoints plus latest, same canonical schedule. Each adds
head parameters, Adam state, per-frame clock, both RNGs and gaze/log byte offsets.
Resume rejects changed world/source/policy identity; restores original stream,
carry/cache and RNG state, and truncates only its own uncheckpointed gaze/log tail.
`gaze.bin` is16 bytes/frame (~410MB at full budget), with target row, episode UID,
sampled and paired-uniform windows, component argmax and sampled probability.
A selected earlier checkpoint's gaze analysis stops at its saved trace offset.

Monitor uses the SAME fixed uniform gaze and checkpoint rule as F1.6. The final
checkpoint is read unless three consecutive monitor increases trigger the
existing global-minimum rule. Full budget always completes; no train-longer rule.
The one-shot worker validates budget/source/checkpoint identity before launching
evaluation, records completion/failure explicitly, and never launches P2/P3.

## Automatic evaluation, same grounds plus the useful fovea audit

`queue.sh` retains F1.6's full standard battery on the selected checkpoint:

1. Full S1 with newly fitted checkpoint-specific token readers.
2. Actor interface under common random AND guided gaze, paired with F1.6;
   M0.5 evaluator/geometry; trained pixel-decoder rendering.
3. Fixed denoising/event panels, CPU counters/light, cascade sampler factorial;
   reuses paired F1.6 reference results and contexts. Euler3/churn1 stays primary,
   alternatives secondary, no best-sampler selection.
4. Real/generated conditioning and all-query consistency;250-episode CPU past-
   glimpse uptake. Existing event-locus C/S reading.
5. EXTRA: same predeclared58-scene night genuine-look local uptake audit used in
   F1.6's review: identical49 real crops, levels/noise, local and whole-map next
   predictions, next-fine TSP, newly fitted fine readers and common-readable
   subset. Reuse SHA-verified existing F1.6 raw baseline; do not alter its outputs.
6. EXTRA: frozen ONLINE head on the existing saved and confirmation dev cohorts,
   using its own checkpoint encoder. No head refit; report the adopted combined
   policy against uniform, remote/contact coverage and C/S diagnostics. Component
   C alone is not an adoption gate. This measures whether frozen-head findings
   transfer to online acquisition, without changing the primary world evaluation.
7. EXTRA: actual training gaze trajectory across deciles, HUD vs map and entity
   contact on the already labelled train subset, paired with its original uniform
   draws. Descriptive acquisition, not held-out teaching benefit or new labels in
   training. First frames with no transition target are excluded from label reads.

Fixed gaze comparisons separate training benefit from merely selecting easier
queries at evaluation. Do not apply F2's head to F1.6 encoder coordinates. The
head's own gaze is described separately; there is no claim yet about P2/P3 VoI.

Read the full panel: semantic consequences/reward/death, entity retention/counts,
real/generated gaps and uptake/regressions, not only averaged denoising MSE or
counter reconstruction. Gains in teaching and gains in using fine evidence are
separate claims. All new outputs live here; previous evidence stays untouched.

## Preflight and launch

Mechanical requirements: causal-history/reference-policy CPU checks; exact
world cfg; uniform-control parity with canonical P1 (including nonzero world
updates and actual auxiliary training); resumed learned trajectory; both chunk
phases; sustained200-update64-step smoke; uniform monitor and native sampling;
source pins and output-isolated evaluation/selection dispatch checks. Tiny GPU
arithmetic discrepancies must be measured against canonical repetition, not
silently called bit-for-bit parity. Preserve failed checks and their explanation.

`checks.json` / CHECKS.md will record final qualification. `run --dry-run` prints
commands only. No overnight launch until the user requests it. Proposed command
is in CHECKS.md once preparation is complete. No scientific result has selected
or rejected the run during smoke testing.
