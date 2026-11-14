# Can Mamba learn the pixel-only clocks, and under which pressure? (2026-11-14 night)

User, 11-14: would rebalancing the loss do it; would more denoiser pressure push Mamba to learn the clocks; can we
know; or is it a Mamba problem, the model not realising it has access to the information (as with mining counts
before tokens could attend to each other)? This localises the failure by substitution: Mamba is given one kind of
pressure at a time, with the whole optimiser to itself. A diagnostic: the event labels below come from the
simulator's HUD digits and are used only to locate the failure, never in a world model's recipe.

## Arms (`learnable.py`; F6 checkpoint 10's perception frozen; the training split; same data draws per arm)

Windows of 128 consecutive steps from random training episodes (T >= 128), uniform prior fixations, the belief run
from its zero state over each window; 32 windows per update, 3,000 updates (12.3M frames), Adam, belief lr 1e-4
(F6's peak), head lr 1e-3, 100 warm-up updates, gradient clip 1. Each arm's loss is on the predicted belief of each
frame (bp_t, before frame t), as every F5 / F6 forecast head:

| Arm | Belief | Loss |
| :--- | :--- | :--- |
| A0 frozen | F6's, frozen | events head only (reference: what F6's belief already holds) |
| A1 events | F6's, trained | log-loss on five binary events (below): a direct request, surprise-priced, undiluted |
| A2 digit MSE | F6's, trained | change-form token forecast (prev tokens + head(bp)), squared error on the 4 HUD digit cells only |
| A3 all-cell MSE | F6's, trained | the same forecast on all 81 cells: F6's token forecast with the optimiser to itself |
| A4 fresh events | random init | as A1: learnable from scratch? |
| A5 G1 only | F6's, trained with the denoiser | G1 alone (`train.p1_losses`' g1 term, F6's streams and schedule shape, 16 x 64, 2,000 updates): more denoiser pressure, nothing else |

Events at transition t-1 -> t (each with its eligibility, applied to loss and reading): food drop (awake, food > 0),
drink drop (awake, drink > 0), energy drop (awake, energy > 0), energy rise (asleep, energy < 9), recovery (+1
health, awake, health < 9; positive control, already carried by F6).

## Reading (fixed before results)

Every arm's trained belief encodes all DEV episodes from the zero state (F6's evaluation fixations); one linear
logistic probe per event on bp, fitted on 4/5 of DEV episodes, AP on the other 1/5 (eligible frames), at updates
750, 1,500 and 3,000 (A5: 1,000 and 2,000). A0 is the reference. Interpretation:

- A1 far above A0: Mamba can learn the clocks from these inputs when asked; the failure is the request, not the
  architecture. A1 near A0: an access limit (input route or recurrence), the kind the mining counts had.
- A4 against A1: learnable from scratch, or only on top of F6's representation.
- A2 against A1: squared error without dilution. A3 against A2: dilution with the optimiser to itself. A3 far
  above A0 means that giving F6's visual forecast the whole budget would have formed the clocks (rebalancing works
  in this regime); A3 near A0 with A2 above it means dilution inside the term, which no between-term weight fixes.
- A5 against A0: what the denoiser's own request does with the whole budget.
A fine-tuning result from F6's checkpoint is not a from-scratch training result; the budget is bounded.

## Revision (user, 11-14 night, before any arm finished)

The user's question was only whether a larger, even equal, share of the loss for the denoiser and the other world-model
objectives would make a difference. Arms that use simulator event labels as training targets (A1, A4) or score only
the HUD digit cells (A2) cannot become part of Saccade and were stopped (A1 had logged updates 750 and 1,500, kept
in `events.log` as one fact: the belief can learn the energy clocks from its inputs when a loss asks for them; it says
nothing about a recipe). A3 (the token forecast alone) was dropped with the token forecaster.

Kept, because it answers the question at its extreme: **A5, G1 with the entire loss budget** (every other loss off),
continued from F6 checkpoint 10. Claim: G1's request for the clocks is too weak because of its share. Competing
explanation: G1's own objective prices the clocks at about 1% of itself, so even all of the budget teaches them
little. A5 now saves the adapted world at 1,000 and 2,000 updates and runs F6's own readings on it
(`trajectory.py`: exact-history pairs drawn natively, due against not due, for recovery, cooldown, energy and hunger;
the belief probe), against F6's checkpoint 10 and the unchanged F6 belief under the same linear probe
(`reference`). If even the whole budget forms no pixel clock in generation, a rebalanced joint run is not worth
running; if it does, an equal-share joint run is the next test.

## Result (A5, 11-14 21:22)

G1 alone with every update for 2,000 updates (belief and denoiser trainable) **forms no pixel clock**. Belief probe
(AP; prevalence in brackets), reference F6 checkpoint 10 / after 1,000 / after 2,000 updates: food drop .037 / .037 /
.037 (.034), drink drop .047 / .048 / .047 (.041), energy drop awake .035 / .037 / .039 (.027), energy rise asleep
.170 / .169 / .176 (.083). F6's own native exact-history pairs on the adapted world at 2,000 updates (due / not due,
drawn): hunger .154 / .154 (F6 .160 / .159), energy .007 / .007 (F6 .014 / .013), so no timing either way. G1's loss
fell only from .00420 to .00386. The reward-requested clocks decay once the reward forecast is off: recovery probe .541
-> .326 -> .280, recovery drawn .192 / .000 (F6 .466 / .000); cooldown is unchanged (.911 / .053, F6 .912 / .048) and
the zombie reads stay at .94-.99. Reading: the competing explanation holds. Giving the denoiser the entire budget does
not form the pixel clocks within this budget, so a rebalanced or equal-share joint run is not worth running; the
denoiser's objective itself does not ask for them strongly enough. Bounded by fine-tuning from F6 (about 1M frames per
1,000 updates here) and by the frozen perception. Saved worlds `g1_run/run/ckpt_0{1,2}.pt` (not in git), readings in
`g1_run/readings/`, probe in `g1.json`, reference in `reference.json`.
