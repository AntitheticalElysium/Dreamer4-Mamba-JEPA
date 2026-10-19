# Does persistence make imagination more useful for decisions? H16 decision value, preregistration

2026-10-16. Written before any code for this study exists. The WHITEBOARD's end goal is imagination useful for actor
training. The decision proxies so far (adjacency / health AUC16) score readouts, not choices.

## Question

When a world imagines the futures of each of the 17 first actions and picks the action it imagines safest, how good is that
choice under the real outcome distribution? Does PI's persistence improve it over IC's soft persistence and the D parent's
regeneration?

## Data

- **Roots:** the deeppanel DEV store (`deeppanel_dev_v1`, replayed seeds; the judge block is not read), kept by deepeval's rule.
  Restricted to roots whose real P(dead by 16) varies across the 17 first actions (opp16) and whose 8-frame context carries no
  episode-start sentinel action.
  - 200 roots, seeded (20261030).
- **Labels:** P(dead by 16 | first action) over the 32 real key sequences under the recorded continuation, as
  `check_h16_signal.load`. Labels are used only to score choices.
- **Context:** the root's 8 stored frames, encoded with the frozen Raw encoder. This is shorter than the 15 the transport
  worlds were trained with, equally for every world.

## Worlds

- `pi_s{1,2}` and `ic_s{1,2}`: sampled, M = 4 futures per first action.
- `D_parent_ref`: deterministic, M = 1.
- Each future: the first action, then the recorded 15-action continuation, 16 steps, rolled out as in `transport_recursion`
  (windows of the last ≤ 15 frames).

## Risk readouts (label-free; frozen readers on imagined frames, identical for every world)

- **Primary, HD:** share of a first action's futures in which the imagined health (bank HUD reader) drops by ≥ 0.5 units below
  the root's within 16 steps.
- **Secondary:**
  - **HL:** mean max imagined health loss;
  - **ZA:** share of futures with a zombie adjacent to the player at any step.

## Decision and score

- Choose the argmin-risk action, uniform over ties.
- **Expected safe** = 1 − P16(chosen), averaged over roots.
- References on the same roots (`check_h16_value.arms`): uniform, FIT prior action, one_real_future, oracle31.

## Readings

- **P1:** PI's expected safe (HD) > IC's, at both seeds.
- **P2:** PI's expected safe (HD) > uniform, at both seeds.
- **Secondary:**
  - each world vs prior / one_real_future / oracle31;
  - the HL and ZA readouts;
  - the paired root-level difference PI − IC, with a bootstrap 95% interval over roots;
  - the share of roots where a world's risk is tied across all 17 actions (no imagined damage).

**Verdict:**
- **"Persistence improves decisions"** if P1 and P2 hold at both seeds.
- **"Imagined damage does not separate actions"** if no world beats uniform.
- Anything else is reported as is.

## Known limits

- Imagined health drops are rare in these worlds (R23: within-16 drop rate 0.07-0.26 vs real 0.26), so ties are expected. The
  tie share is reported.
- 8-frame context; 200 roots; M = 4. One protocol, no tuning.
- Fresh confirmation stays sealed. The judge block is not read.
