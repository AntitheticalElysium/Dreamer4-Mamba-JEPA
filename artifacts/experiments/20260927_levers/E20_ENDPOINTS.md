# E20 endpoint implementation supplement — before optimization

Training and E20.md are unchanged. The historical evaluators save generated and
teacher token grids in their hash-bound teval batch stores at windows5/15. The
dual-reader endpoint reuses those grids on CPU, after exact input/checkpoint/source
checks, instead of rerunning the worlds. It reports the stored fp16 precision.
Window4's old health cache saved only the historical reader's teacher scalar; a
compact additional pass records both readers at once. No head is refitted.

Health endpoints retain the declared1.5-unit threshold and separate0.5 sensitivity,
ordinary/fresh/unchanged/negative/recovery/starvation strata and real-successor
controls. Teacher comparisons use identical roots; self-fed comparisons use the
intersection of each world's correctly aligned camera prefixes, with denominators
reported. These are reused diagnostic roots, not a new judgement block or actor.

The paired memory criterion uses **sample0**, selected before any E20 world update,
of the original diagnostic futures. Sample0 allows exact reuse of teval's teacher
states and independent per-root physical sums; the original all-five-key recall
point estimates are also measured by the main queue. The paired subset has its
own counts and episode-cluster bootstrap; it is not presented as the all-five-key
interval. Same-slot ages6..15 must have lower95% capture difference from the own-
seed parent greater than−0.05. Insufficient uncertainty remains a hold.

One-step cost uses saved all-action squared-error sums over the identical copy
denominator, paired by episode seed. Its upper95% arm-minus-parent bound must be
<=0.005. Depth16 cost is reported separately, normalized by the same true-state
variance, on identical living successors. Health wins cannot substitute for these
physical and memory checks. Parent continuations are not a pure alignment contrast.

The post queue waits for the main queue's completed endpoints. Missing parent
teval baselines are generated at5/15; all other saved grids are reused. All new
steps are source/input-bound and resumable, with logs/status under EDA only.
