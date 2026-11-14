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
