# Decision levels: does the noise level at which a consequence is decided govern what P1-F3 learns and generates?

User 11-06, after the adversarial audit (`../20261106_conditioning_audit/AUDIT.md`): "discuss what this means ... and
what we need to test in order to prove or disprove this (both are equally important) ... run what you need."
Written and committed before any of the measurements below exist.

## The hypothesis, stated so that it can fail

The audit's oracle result (a sleeping target + noise is denoised asleep 100%; pure noise at the same level wakes 45%;
an energy-9 awake target is followed at sigma 5, 91%, and ignored at sigma 20, 7%) suggests one mechanism with four
parts. For a consequence whose two outcomes differ by d in token space:

- **H-dec (decision level).** The denoiser follows the noisy target wherever the target still distinguishes the
  outcomes and falls back on its context below that; the switch-over level sigma* grows with d, in step with what an
  ideal two-outcome denoiser could read (sigma*_ideal).
- **H-ctx (context-only prediction).** Above sigma* the output is the context-only prediction: the same as the
  pure-noise first evaluation at that level. It is right for frequent regularities (persistence, wood +1,
  movement) and wrong for the failing rules (energy 9 -> wake: ~the marginal; coal +1: copy).
- **H-start (inference).** Sampling that starts below a global state's sigma* lets the initial noise decide it (the
  audit: starting at 20 fixes most false wakes). Being an inference mismatch, it should not shrink with more
  high-noise TRAINING at the old start.
- **H-learn (training).** A context-only prediction is learned in proportion to the training signal above its
  sigma*: frequency x noise mass there x loss size. Rules that are rare AND decided high (energy) get almost none
  (P(sigma > 10) = 1.2% of draws); rules decided low (coal, d ~ 3.7: sigma* ~ 2) get noise mass but little loss
  signal. So adding mass above sigma = 10 should teach the energy rule and NOT coal.

Competing explanations: the energy digit cannot be routed to the tokens that must change (exp 3: not recovered at
the generated map / lighting tokens), a capacity / architecture limit that more high-noise signal would not fix;
or the rules simply need more training of any kind.

## T1 (frozen F3): decision levels and context-only predictions

Pairs: the audit's 208 (its `select`, seed 110607: 96 SLEEP, 64 ENERGY, 24 COAL, 24 WOOD) plus 64 MOVEMENT pairs
built here the same way (dev, exact replay; awake; the recorded action moves the player; no zombie / skeleton within
6, no arrow; d0 = the recorded move, d1 = NOOP on identical frames; both successors from the recorded key; natural
successor = the stored frame). First evaluation only, each member's own context / memory / action, K = 8 fixed
Gaussian draws, sigma in {0.25, 0.5, 1, 2, 3.5, 5, 8, 12, 20}: (i) pure noise, (ii) the member's own true successor +
noise, (iii) the OTHER member's successor + noise (conflict). Readout: the projection f between the pair's two
successors (0 = d0's, 1 = d1's) over the consequence tokens (map + lighting; counts: the slot).
- Reliance r(sigma) = mean over both members of |f(conflict) - f(own)|: 1 = follows the target, 0 = follows its
  context. sigma*_net = where r falls through 0.5 (log-interpolated). sigma*_ideal: the same for an ideal two-point
  posterior with equal priors at the pair's own d.
- **H-dec SUPPORTED** if across the five families sigma*_net rises with d (Spearman over pairs > 0.5, and the family
  medians order as their d) and each family median is within a factor 3 of its sigma*_ideal; **REFUTED** if sigma*_net
  does not rise with d (Spearman <= 0.2) or a family is off by more than 10x.
- **H-ctx SUPPORTED** if the pure-noise f at sigma = 20 matches the audit's prefix-20 generated rates within 0.10 per
  family (persistence, energy 8 / 9, coal, wood) and movement's is >= 0.8 toward the correct outcome; **REFUTED** if the
  energy-9 context-only prediction is correct (>= 0.5 toward wake) while generation is not.
- Movement, the case that could break H-start: generated movement (nearest successor, K = 16) at the original start
  (5, 16 steps) and at the audit's prefix-20 recipe. Read as: >= 0.9 correct at 5 -> the start mismatch does not
  touch an action-driven large change (H-start then needs the reason, read from T1's context-only predictions).

## T2 (frozen F3): which of the determinant-curve verdicts were the sampler's start?

The determinant curves (all four populations, same tuples and seeds) re-generated with the audit's prefix-20 recipe
(four deterministic evaluations 20 -> 5, then the original 16-step ladder and churn, the glimpse stage unchanged),
patched into the unchanged actor interface; sham: the patch with no prefix reproduces model.sample bitwise.
Predictions: false wakes from asleep histories (energy <= 8) fall from ~0.50 to <= 0.15; the energy value contrast
stays < 0.10; cooldown, lethality, sleep-attack damage, kills and per-count increments change T by < 0.10 (small-d
or hidden-state decisions). **The start lever is broader than global states** if any of those T moves by >= 0.10.

## T3 (training, causal): does high-noise training signal teach the missing rule?

Three fine-tunes of F3 checkpoint 10, identical except the PHYSICAL sigma draw of each training frame (glimpse
untouched; same data stream, same Gaussian noise for the original draw, band choices from a separate generator):
- **C**: the original lognormal(-0.4, 1.2) capped at 20.
- **H**: with probability 0.25 per frame, log-uniform in [10, 20] instead (mass above 10: 1.2% -> ~26%).
- **F**: frames whose state t - 1 is asleep (labels; a diagnostic allocation, not a recipe) take the band with
  probability 0.8; others original (energy-9 exposures above 10 ~2.7x H's).
F3's own loop (F2 trainer: online gaze controller, canonical x-loss, harmoniser, all-parameter clip), 750 updates of
16 x 256 frames, LR linear 1e-5 -> 5.8e-5 over 100 updates then constant (the M0.5 fine-tune's schedule),
**perception frozen** in all three (tokens and the frozen readers stay valid; the belief, denoiser and heads train).
In diffusion forcing each frame is also context for later frames, so H / F also noise more contexts: declared, part
of the intervention.

Evaluation per arm (and F3): the audit's 208 pairs with its base-5 and prefix-20 recipes and readers; the pure-noise
first evaluation at sigma 20 for the energy members; the P1 monitor.
- **Primary: the energy 9-minus-8 wake contrast at prefix-20** (required 100 pp; F3 0.2 pp).
  - H >= 33 pp and C < 10 pp: **H-learn SUPPORTED** (high-noise signal unlocks the rule).
  - H and C < 10 pp, F >= 33 pp: learnable when exposed; the natural mix at this budget is too thin (frequency).
  - H, C, F all < 10 pp: **H-learn REFUTED at this budget**; points to routing / capacity (the competing explanation).
  - C >= 33 pp: more training of any kind, not allocation.
- Coal (prefix-20): |H - C| <= 5 pp predicted; H better by > 5 pp **refutes** the decision-level split (coal decided low).
- False wakes at the ORIGINAL start (base-5): |H - C| <= 15 pp predicted (H-start is an inference mismatch); H lower by
  > 15 pp: the base-5 failure is also high-noise under-training.
- Costs, descriptive: wood, persistence at prefix-20, the monitor's low-noise levels.

## Limits

One world seed and one fine-tune seed per arm; 750 updates is ~5% of P1's updates; perception frozen; diagnostic
populations, not corpus frequencies; token-space readouts. A supported H-learn says the allocation is a sufficient
lever for this rule, not that it is the recipe to adopt (that is a design decision with costs, discussed after).

## Crafting at the frame level (added 11-06 after the need analysis; user: "For the measurement, run it sure")

The need table's one gap. Correction first: crafting is NOT "own past + a specific action". Craftax Classic
`do_crafting` makes a tool only if a table (iron tools: a table AND a furnace) is among the 8 cells around the player
and the materials are in their slots (wood; + stone; iron tools + coal + iron; 1 of each spent). That rule reproduces
every awake attempt in dev and train400 (0 mismatches), and 87-98% of attempts are infeasible. So the tool slot's
change is decided by OTHER tokens (map cells around the player, other HUD slots) plus the action. Made events per
training pass: wood pickaxe 6.9k, stone pickaxe 3.9k, iron pickaxe 1.4k, wood sword 5.0k, stone sword 4.3k, iron sword
1.4k (need.json).

Frozen F3, dev, one-step futures with the prefix-20 start (T2's recipe and readers, K = 4, seed 4200), plus frozen
readers for the six tool slots (s1.fit on train400 tokens, same validity rule). Populations: awake attempts at t >= 8,
per tool, stratified by the rule: feasible from count 0 (all), feasible from count >= 1 (all), infeasible from count 0
with the station missing only (cap 250), the materials missing only (cap 250), both (cap 150).

Declared readings:
- Per tool, the feasibility contrast of the tool slot drawing n -> n + 1 (truth 1 feasible, 0 infeasible): T, Q, false
  and TRACKS / PARTIAL / BLIND as in the determinant curves. Beside it: false appearances split by what is missing
  (station = map tokens; materials = HUD tokens), and the materials-spent contrast (all cost slots drop by 1).
- **Location-law prediction**: the iron tools (1.4k events per pass, below coal counts' 4-6k where the count drew
  0.10-0.25) are NOT TRACKS, and their feasible appearance is <= 0.3. **Refutes the law's exposure threshold**: an iron
  tool TRACKS (T's lower bound >= 2/3), i.e. a cross-token determinant used at 1.4k events per pass.
- Pattern, descriptive: high Q with high false = drawn from the action alone (blind to the other tokens); low Q with
  low false = the majority outcome of an attempt (nothing), blind as well.
- Heads beside: the unlock head's contrast on generated and on recorded frames (first craft only), and its agreement
  with the frame (WORLD-DRAWN / HEAD-READ).
Limits: one world seed; dev histories; token-space readers; feasible-from-n >= 1 is small (26-203 per tool).
