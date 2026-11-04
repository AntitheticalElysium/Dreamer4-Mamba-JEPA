# One-step semantic fine-evidence test (authorized 2026-11-04)

Frozen F1.6 ck10, SHA256
`34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045`.
No fitting, world updates, successor architecture, or F2 results. Source selection
and simulator qualification precede model consequence outputs.

## Bounded cases and why there are two candidates

1. Arrow heading, 24 source-selected first-visible adjacent-arrow scenes:
   first16 with light<.25 and first8 with light>=.5 in dev0:400. The simulator
   renders toward/away headings and applies the same NOOP and environment key.
   These are valid rendered states, NOT proven to share a feasible episode past.
   The initial source check finds a health difference of2 but coarse inputs also
   differ. Therefore this is a semantic sensitivity/capability control, not a
   pure fine-only memory diagnosis. Preserve all24 and report that limitation.
2. Natural night threats: source-selected visible arrows/zombies within two
   Manhattan cells, first dev400 episodes, t>=8, awake, health falls >=2 on the
   recorded next action. At most256 source frames per kind, in corpus order.
   Current gaze centers the threat, versus a same-row disjoint crop. Existing
   frozen F1.6 readers identify the threat: require both fine readers correct;
   split by coarse reader correct/missed. Keep at most12 per kind/coarse stratum,
   one case per episode within each stratum. No new reader fit. This is achieved
   readability, not a proof all coarse/history information is absent. Replay and
   remove the source entity under the same action/key to verify its causal
   contribution; exclude DO revivals and other interactions only if this source
   qualification fails, with every reason retained. Include arrows/zombies,
   not just one rare symbol. If a stratum is empty, report it; no threshold changes.

## Measurements fixed before model consequence outputs

- Replay complete real prefix through the actual full Mamba carry. Same actual
  source frame, reward so far and next action; only current genuine gaze changes.
  Both gazes produce identical current periphery/lighting. Fix the next query
  at HUD center(7,1), independent of the current look. No actual future percept,
  reward or termination enters the prediction.
- Counterfactual heading control: both headings, both gazes, shared prefix,
  source reward, NOOP and all random draws. Its artificial-history limit above
  applies; do not use it alone to diagnose native memory failure.
- Primary: sampled NEXT PHYSICAL health value via the existing frozen F1.6 HUD
  nearest-mean reader, with its error and support on these real successors.
  Report probability of lower health, expected health and support/unreadability;
  no unsupported generated token silently counts as correct. Reference outcome
  heads on the actual successor check whether the task can be expressed.
- Secondary: native reward head negative-reward probability and expectation on
  generated successors and real successors. It also reads the generated fine
  stream, so it does not isolate the memory route.
- Sampler: production3 steps/churn1 and fixed secondary8 steps/churn0. 64 shared
  random futures per condition (smoke4), batches16. Report both, never select
  the sampler after seeing the result. No rollout beyond the one P1-trained step.
- Capture raw fine, local joint, mixed input, full conv/SSM carry, corrected and
  predicted emissions, exported memory. Check exact collisions, plus RMS
  differences; a nonzero delta is sensitivity, not semantic retention. At fixed
  gaze also compute the fine-only component of the heading contrast with all
  other input coordinates controlled, in native precision and FP64. This is an
  algebra/control reading, not an in-distribution generation experiment.
- Assert physical cache parity across genuine gazes with common prior memory.
  Factorized physical prediction cannot read fine cache/coordinates: a correct
  health effect across these gazes demonstrates the fine-to-belief-to-physical
  route. A fine/reward effect alone may use the direct fine cache.
- Case-level paired effects and episode-bootstrap95% intervals; difference of
  focused vs control gaze, and heading-by-gaze interaction for candidate1.
  An interval including zero is inconclusive, not evidence of exact absence.

## Interpretation and stop

Positive health benefit demonstrates native consequence-specific fine use on
that cohort. No projected/full-carry collision excludes exact pair erasure only,
not practical inaccessibility. Clear real-successor reference failure prevents
blaming the fine route alone. A coarse-missed/fine-readable source is stronger
than the old arbitrary crop comparison, but does not certify unique fine-only
information; a null there cannot uniquely separate Mamba from its consumer.
Do not patch architecture after a null. Preserve every case and stop for user
discussion after this bounded two-candidate campaign.

Source counterfactuals and semantic metrics follow the cautions in Zhang & Nanda
(ICLR2024), https://arxiv.org/html/2309.16042v2 and Heimersheim & Nanda(2024),
https://arxiv.org/html/2404.15255v1 : locality claims depend on the behavior and
contrast tested; redundant routes make necessity-only nulls insufficient.

## Source-only amendment, before ANY native consequence output

The initial survey found nearby threats coarsely readable:74/75 arrows and
256/256 zombies; the fine readers correctly identify all331. Only one coarse
miss exists there. Accordingly also qualify distant night arrows (distance>=3,
visible now and next frame, moving closer under the recorded action): first256
source frames in dev0:400, both fine readers correct, fixed coarse reader must
recognize the arrow at its NEXT position. Split by source coarse correct/missed,
at most12 episodes per stratum. Primary semantic event is now next-step arrow
occupancy at that simulator-verified destination, read by the SAME existing
physical reader. Removal under the same simulator key/action must remove that
destination occupancy. This is a one-step deterministic movement consequence,
not a claim of immediate damage, long memory, or behavior value. Health remains
primary for nearby/heading cases. No native output has been inspected when this
source-based fallback is added. Preserve the initial survey separately.

## Same-gaze content control, before inspecting native consequence numbers

Genuine gaze changes also change a location mask and may provide an oracle
location hint. Therefore on the12 coarse-missed distant-arrow cases, cross the
TWO valid coarse streams with the TWO valid fine streams at the SAME focused
gaze, same prior, action and randomness: (pA,gA),(pA,gB),(pB,gA),(pB,gB).
Only g changes within each coarse pair. Run64 matched futures per condition,
both fixed samplers, same next-destination reader. This tests fine-content causal
sensitivity beyond gaze/mask, with identical physical cache as an assertion.
The two matched endpoints are real valid renders; the two crossed combinations
are internal interventions and need not occur naturally. Thus this control can
show the network responds correctly to the fine cue, but cannot establish
calibrated world probabilities or make a null into a proof of information loss.
This addition is motivated by a design confound, not by any observed prediction
result. It is the last control in this bounded campaign; no architecture patch.
