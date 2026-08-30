# The `gate` field in this directory is the SUPERSEDED criterion. Do not read it.

`report.json` and `run.log` here were written before S80 changed the primary actor gate.
They record:

    "gate": {"criterion": "paired 95% lower bound of official-score actor-minus-BC > 0",
             "passed": false, "decision": "oracle_phase3_did_not_beat_bc"}

That criterion is the **official geometric score**, which S80 demoted to "mandatory beside
it". The primary causal gate is now **mean achievement count**, and on this same run the
oracle PASSES it:

    achievements   actor 7.939   bc 7.113   gap +0.826  [0.537, 1.119]   (S80, 512 DEV)
    geometric      actor 12.917  bc 12.824  gap +0.092  [-1.011, 1.143]

So the correct reading of this run is: **the h=2 fully-oracle actor beats its BC prior on
the primary metric.** It is a live positive control, not a failure. S80's own words: "The
oracle result establishes a live but weak local actor path, not satisfactory control."

`run_oracle_phase3.py` was subsequently fixed and now reports the S80 criterion directly
(`passed = comparison["achievements_beats"]`, with the geometric score co-reported). Any
report whose `contract.version` is `oracle-phase3-v2` and whose `gate.criterion` mentions
"official-score" predates that fix. This directory is one of them; it is kept unmodified
because it is evidence.

Note also that this stored report carries the achievement *means* but no achievement
*interval* — the [0.537, 1.119] above comes from S80, not from this file. The file alone
cannot be evaluated against the current gate.

Why this note exists: the superseded `passed: false` was read as "even a perfect world
model at horizon 2 fails the gate", and that false premise was used to argue for making an
h=16 oracle run a prerequisite for world-model work. It is not. Horizon is a separate
Phase-3 question (S82: "The h=16 oracle is a separate Phase-3 question and is not evidence
about the world model").
