# P1-F2.1: F2 on the union corpus (user 11-04: diversity arm approved)

**Question.** Does the broader corpus (original support-v2 + coverage-v1 + risk-v1: `../20261104_rare_collection`,
census `../20261104_dataset_eda`) improve the world at F2's compute? This is diversity at fixed compute (Codex's
framing), not more optimisation pressure on rare failures (their per-frame exposure falls) and not fovea use (data
cannot create fine-information need at 3 x 3).

**Prior from tonight's train-vs-dev reading (`../20261104_f2_train_dev`).** F2 does not generate first coal, first
iron, risky death or adjacent damage even on histories it trained on ~10.5 times (gap closure -0.03 to -0.005, all
upper bounds < 1/3). This weakens a simple memorisation/generalisation explanation;
it does not establish that diverse contexts cannot teach those relations. Improvement
on any of the four would be compatible with that reading; all remain scored.
Drink and furnace showed a train advantage (diversity could help them). What the reading does not touch: the new
data's states that F2 never trained on (late game: frames past step 1,023 10.3 -> 19.0%; off-expert actions and failure
tails), i.e. the actor's own future distribution, read on the new held-out episodes below.

**What changes.** The corpus only: every source once per pass (streams play whole episodes from one shuffled queue,
so frames are weighted by length, as in every P1 run): 22,267 training episodes / 8,914,902 transitions, 2.86 passes
at 25.6M frames (F2: 7,672 / 2,432,119, 10.49 passes). The recipe differs from F2's checkpoint cfg in `data.corpus`
and `p1.out_dir` only; saccade's model / train / data / m0.yaml are byte-identical to F2's (checked); F2's trainer and
chooser, budget, seed 1, checkpoint rule unchanged. Consequences of the data change, by design: the reward bins'
decode values are the union train rewards' bin means (`set_reward_values`); the chooser head trains on the union
stream.

**Construction (`union.py`; for review).**
1. A virtual store `artifacts/craftax_union_v1/manifest.json`: the three complete source manifests' shards in order
   (original 420 first, then 128 + 512), files relative to the union directory, hashes copied; no shard copied.
   Every existing (shard, slot) label keeps its meaning (dev / final are original-store episodes, unchanged).
2. `use(cfg)`, called by the launcher before any data access: points `replay.STORE` at the union and wraps
   `data.load_split` (launcher process only; both files untouched) so the ORIGINAL store's 5% monitor stays the
   checkpoint-selection monitor (comparable with F2) and the new stores' monitor-hash episodes (140 coverage, 625
   risk) move to `held_out`: never trained on, kept for reading.
3. Change-map targets (G2's change loss) for every new training frame from the frozen teacher that wrote the
   original targets (`artifacts/saccade_m0/targets/change_teacher.txt`, `data.change_targets` unchanged), in 5
   frame-balanced parts, appended to the original targets (bitwise copy) in `artifacts/saccade_m0_union/targets`.
   The teacher must reproduce the original targets bitwise on 12 spread original episodes before assembly.
4. `verify`: monitor / dev / final identical to the original store's split; original train list = the union's
   prefix; ids unique; train 22,267 / 8,914,902 and held-out 140 / 625 = the census; dev / final from the original
   store only; every shard file's sha256 = its source manifest's; all 1,060 shards mapped at once in one process
   (open descriptors recorded); 24 union reads = direct source reads; targets shape and hash.

**Declared readings (automatic after selection, implemented before launch).** Against F2, the
same instruments: actor interface (random gaze and own closed-loop gaze), M0.5 evaluator, S1, the chooser handover
(coal / iron counter forecasts). Per class, the change vs F2 read against the event's exposure ratio and unique-
instance growth (EDA tables). Expected, if exposure governs: up for plant (1.60x), diamond (1.45x), drink (1.19x), iron
increments (1.08x) / iron tools (~1.1x); slightly down for most first unlocks (0.71-0.83x) and danger (0.77-0.86x).
Expected, if distinct instances govern: gains can also occur where exposure fell, including the four primaries.
New-distribution reading: F2 and F2.1 on the held-out coverage / risk episodes using
S0-format labels assembled before launch from the hashed collection sidecars
(`heldout.py`, `heldout.json`), with per-transition exogenous flags retained.
Each model uses its own saved head and closed-loop query gaze on the generated
coarse percept; recorded successor crops are not provided to that choice.
F2 has never trained on those policies' states. One seed: per-class differences
of a few pp are uncalibrated (F2 vs F1.6 moved classes 2-10 pp both ways); read patterns and large effects.

**Preflight.** `union.py manifest | part I 5 | assemble 5 | verify` -> `union_checks.json`; `checks.py` (exact
learned-gaze resume over the union stream, the real launcher in a fresh process, 200 updates at 64 steps and 3 at
256, the monitor, the trained frames' source composition) -> `checks.json`; `pin_sources.py` -> `SOURCE_PINS.json`.
**Launch** (user's approval after review): `systemd-run --user --unit saccade-p1f21 -p MemoryMax=24G -E
MAMBA_DETERMINISTIC=1 .venv/bin/python -u -m artifacts.experiments.20261104_p1f21.run` (gates, training, F2's rule;
then `evaluation.py`, selected-checkpoint path passed explicitly). Per-stage logs
and failure status are recorded; no P2/P3 training is launched. The unchanged
original monitor chooses the checkpoint, all 10 numbered checkpoints retained.
One GPU job at a time; the CPU uptake reading may overlap evaluation.

The complete usual battery is retained: S1; random/guided actor interface; M0.5
glimpse readings and geometry; decoder renders; Codex panel, sampler factorial
and paired summaries; consistency; locus; source-local fine uptake; online-head
readings and gaze trace. Closed-loop own-gaze interfaces on old dev and each new
held-out source are added, reported separately. The historical guided interface
remains a descriptive reference with its privileged gaze caveat, never a claim
of honest danger competence. `evaluation --dry-run CHECKPOINT` lists the queue
(CLI order: `evaluation CHECKPOINT --dry-run`).

**For the reviewer, explicitly.** (a) the launcher-scoped replay / load_split patches instead of editing shared code;
(b) original monitor kept, new monitor-hash episodes held out; (c) raw union, no weighting, fixed budget; (d) reward
bin values from the union; (e) the store grows from 36 to 113 GB read through mmap under the 24 GB cgroup (page cache
is reclaimable; throughput is measured by the 200-update check against F2's checks).
