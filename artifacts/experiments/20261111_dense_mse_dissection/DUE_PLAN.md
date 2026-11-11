# Can the belief tell whether the next decrement is due?

Frozen readout on the same 192 DEV episodes and folds as the original belief audit. Exact-phase regression can miss a useful threshold signal. Use current awake states with an old visible reset, non-extreme digits, and no intervening sleep, as before. Exclude DO and SLEEP using the current action, so the exact timer rule is applicable without intervention. Require equality to actual next-frame digit transitions throughout the cohort. Use exact phase only to check truth, never as reader input.

Fixed linear logistic and small MLP readers with the recent semantic features alone, or the same features plus the actual post-action belief, for all five checkpoints. Report AP and proper Brier score, with paired episode intervals on Brier differences. Recent semantics is a control for probe accessibility and a forecast covariate, not the real denoiser cache. No world updates.

Reuse the existing native thirst diagnostic with its original qualified reader protocol on the two 2,000-update exports. One previously selected history alias is a controlled comparison within each checkpoint, not an estimate of thirst forecasting on all play. No new native energy, healing, or fatigue-asleep bank in this turn.
