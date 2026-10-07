# October 5 reboot recovery

Recorded before relaunch. Reboot inspection at 22:42 UTC October 4 (09:42 AEDT October 5): no research Python processes
or user services; GPU 474 MiB desktop / 5,375 MiB free. This is a recovery of declared comparisons, not a new treatment.

| Work | Recoverable evidence | Action and scientific question |
|---|---|---|
| E17 A16 s7/s8 | Final 6,000-update worlds and training states | Preserve; evaluate with supervised input window 15 |
| E17 M16 s7 | Only a separate 500-update smoke state; historical full run logged 1,000 without saving | Restart from its own 36k M6 parent; never substitute smoke weights |
| E17 M16 s8 | Not started | Continue own 36k parent, identical declared 6k/L16/b40 recipe |
| E18 canvas s7 | Full optimizer/order/CPU+CUDA RNG state at 13,000 | Archive state, verify source/default treatment and dataset, GPU interruption check, resume to 36k |
| E18 canvas s8 | Not started | Same recipe/36k; both seeds are required for the moved-cell retention rule |
| M6 H16 s7 | Finished legacy report | Preserve; do not regenerate this world just to populate a new cache |
| M6 H16 s8 | Unjournalled partial FIT feature file, no DEV features or head/report | Preserve metadata/hash and file; recompute in a fresh resumable namespace and result directory |
| E16 s7 | Corrected full evaluation complete; ordinary damage 0/305, starvation 0/32, recovery 0/112 | No blind s8 prior replication: reconstruction fails before prior prediction |
| E19 | Design only | First isolate death-layout effect from health loss dose. Do not apply a combined repair or modify live training source |

Resource order amendment, before canvas has resumed: actual M16 + desktop uses ~3,599 MiB, leaving ~2,251 MiB,
below canvas admission 2,300+256=2,556 MiB. Canvas therefore waits until the E17 lane finishes. H16 evaluator memory
falls during head training and grows again inside its process; starting canvas in that gap risks a later collision.
Only canvas's waiting shell is restarted to load this scheduling guard; no computing research job is stopped.
Completed M6 seed8 H16 evaluation waits for the E17/E18 services to stop, avoiding an expensive feature pass starving comparisons.
E17/E18 evaluation commands keep their declared order and existing GPU admission lock. Training checkpoints every
1,000 updates; evaluator caches bind source/checkpoint/input/label contracts and checkpoint complete root batches/heads.
Complete legacy reports and partial files are retained. Contract drift, corrupt state, non-OOM exceptions fail closed.

Recovery manifest pins actual sources, parents, original recovery state and dataset bytes. Historical A16 source hash
belongs to the original trainer; the only trainer change is the already-validated checkpoint cadence, not loss/math.
The manifest is an external recovery record, not a retroactive claim that historical training states sealed their inputs.
GPU interruption numerical differences are reported as measured, rather than assumed bitwise zero.

Future recovery entry point: `/bin/bash artifacts/experiments/20261005_recovery/launch.sh`. It refuses duplicate services,
rechecks pinned source/parent and all dataset bytes, verifies working states and final checkpoints, then reloads the
same queued lanes. Changed code/data fail closed; the current tree is not silently rebound to old experiments.
Working training states advance and are checked dynamically; immutable historical copies retain their old identity.

E17 tests trained use of longer history; parent versus continuation also changes targets/update, ledger and optimizer.
E18 tests coordinate registration on the six-frame recipe, retaining the historical absent-frame convolution behavior.
Neither is an ordinary-health repair or an imagination-trained actor result. New health work must separate unchanged
continuation, de-alignment alone, and dose added on identical de-aligned batches, before combining with longer memory.
