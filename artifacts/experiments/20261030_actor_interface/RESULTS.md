# Actor interface (10-30; corrections 10-31)

First run (d39aec70, `results.json`, `results_random_split.json`; P1 world 1 ckpt 10 and M0.5 guided_1500, random
fixations): death in risky contexts 0.04% generated vs 12.8% recorded (context-defined: stands); damage 1/4-1/3 of
recorded; per-achievement unlock on outcome-selected rows (sensitivity diagnostics, recall for deterministic unlocks).
CORRECTED (Codex, 10-31): "half of the imagined achievement reward is false" is WITHDRAWN. The split of generated
P(unlock) by the recorded outcome puts a calibrated world's mass on histories whose recorded draw missed (independent
draws: a perfect 10% predictor shows 90% there); and the head's event (reward > 0.5) differed from the achievement
label on 123 dev transitions. `interface.py` now uses the same event on both sides and a held-out per-history
reliability (`check_reliability.py`: a perfect world reads calibrated, a 30% world under-predicted; my first version,
binning and reading the same 4-draw mean, failed that check), with --policy=random|guided. World 1 and P1-F are to be
read through the corrected evaluator under both policies.
