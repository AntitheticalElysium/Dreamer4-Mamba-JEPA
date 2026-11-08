# Broad collection coverage

**Collection status:** complete.

Committed 128/128 batches, 3,072 episodes, 3,880,121 transitions; 2,932 episodes/3,711,420 transitions after fixed monitor exclusion.

All original dev/final data remain unchanged. These are acquisition counts, not learning or model-quality results.

| Achievement | Old train unlocks | Added train unlocks | Combined | Old / combined per million transitions |
|---|---:|---:|---:|---:|
|COLLECT_WOOD|6,671|2,931|9,602|2742.9 / 1562.9|
|PLACE_TABLE|5,804|2,930|8,734|2386.4 / 1421.7|
|EAT_COW|4,466|2,888|7,354|1836.3 / 1197.0|
|COLLECT_SAPLING|4,868|2,824|7,692|2001.5 / 1252.0|
|COLLECT_DRINK|4,105|2,847|6,952|1687.8 / 1131.6|
|MAKE_WOOD_PICKAXE|5,102|2,926|8,028|2097.8 / 1306.7|
|MAKE_WOOD_SWORD|3,976|2,823|6,799|1634.8 / 1106.7|
|PLACE_PLANT|4,507|2,811|7,318|1853.1 / 1191.2|
|DEFEAT_ZOMBIE|4,300|2,872|7,172|1768.0 / 1167.4|
|COLLECT_STONE|4,750|2,919|7,669|1953.0 / 1248.3|
|PLACE_STONE|4,466|2,911|7,377|1836.3 / 1200.8|
|EAT_PLANT|184|778|962|75.7 / 156.6|
|DEFEAT_SKELETON|2,509|2,683|5,192|1031.6 / 845.1|
|MAKE_STONE_PICKAXE|3,047|2,826|5,873|1252.8 / 956.0|
|MAKE_STONE_SWORD|2,986|2,813|5,799|1227.7 / 943.9|
|WAKE_UP|7,416|2,911|10,327|3049.2 / 1681.0|
|PLACE_FURNACE|4,169|2,899|7,068|1714.1 / 1150.5|
|COLLECT_COAL|3,398|2,802|6,200|1397.1 / 1009.2|
|COLLECT_IRON|2,249|2,696|4,945|924.7 / 804.9|
|COLLECT_DIAMOND|187|626|813|76.9 / 132.3|
|MAKE_IRON_PICKAXE|1,045|2,221|3,266|429.7 / 531.6|
|MAKE_IRON_SWORD|1,096|2,268|3,364|450.6 / 547.6|

| Event | Old train | Added train |
|---|---:|---:|
|damage|47,618|40,211|
|death|7,621|2,311|
|risk|28,891|12,194|
|risk_death|3,701|759|

## Additional danger collection

Status: running, partial; 156/512 batches, 3,557 train episodes/855,688 train transitions.

| Event | Old train | Advanced addition | Danger addition | Old / all combined per million transitions |
|---|---:|---:|---:|---:|
|damage|47,618|40,211|19,091|19578.8 / 15276.0|
|death|7,621|2,311|3,553|3133.5 / 1926.6|
|risk|28,891|12,194|12,527|11878.9 / 7659.7|
|risk_death|3,701|759|2,065|1521.7 / 932.2|

The all-combined rate is an accounting diagnostic, not an adopted sampling mixture. No model run has been launched on the concatenation. Preserve or deliberately change danger frequencies before choosing the training recipe; do not let long competent episodes dilute them silently.

Absolute counts and rates both matter. A rare unlock count can increase while its mixture frequency barely changes if the collection adds many ordinary transitions. Different source episodes and later-state action coverage matter beyond repeated events. Whole episodes are kept; there is no post-outcome acceptance filter.

The manifest also reports all inventory increases and prior-value histograms, body increases/decreases, day/night unlocks, visible creature frames, component-specific counts and the monitor partition. Per-transition exogenous flags and behavior probabilities are in audit sidecars.

Remaining limits: this frozen expert may still miss behaviors; an absent state/action combination is not fixed merely by collecting more expert play. Compare the final report and propensities before selecting the training mixture. Fixed-frame-budget training will give fewer repetitions of a larger corpus. No sampling weights, model losses, acuity, architecture or overnight training are changed by collection.
