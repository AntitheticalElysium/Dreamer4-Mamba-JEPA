# P1-F1.5: the architectural baseline, everything but the learned fixation (11-01 night; for review before launch)

**Why (user 11-01; Codex supports).** The learned chooser (decision 2) is not settled (H3: the endogenous floor as
built fails; the design needs discussion, not another ablation). Decision 1 is built and checked
(`../20261101_factorised_denoiser/PLAN.md`, results 1-7). P1-F1.5 trains the system with every agreed piece except the
learned chooser, so the chooser's later run has its control: P1-F2 = P1-F1.5 + the chooser, nothing else changed.

**Recipe** (`train_p1f15.py`, saccade.train.p1 itself): P1 world 1's config, data streams and world seed 1, with three
declared changes written into the checkpointed config: (1) P1-F's faithful pixel decoder (mean by squared error, scale
by NLL on the detached mean); (2) decision 1's factorised denoiser (physical tokens never attend to glimpse tokens in
spatial layers; independent physical / glimpse noise levels and adaLN, UniDiffuser); (3) the cascade's glimpse
conditioning level 0.1 (used only by the sampler, i.e. by readings). Fixations: P1 world 1's, uniform over the 49
windows (data.fixation_prior, seeded per stream); no hand placement, no learned chooser, no reference model, no floor.
Budget, schedule (25.6 M frames, 64- then 256-step chunks), optimizer, harmoniser, checkpoints and monitor unchanged.
MAMBA_DETERMINISTIC=1 (required by the launcher; mamba_ssm otherwise autotunes per process, so a resume would not
continue the same computation; deterministic mode costs no time here: 1.32 s / update joint, as P1-F's 1.36).
Expected ~11 h (P1-F 9.5 h x 1.18-1.21, decision 1's measured training cost).

**What it can and cannot be compared with.** Against P1-F it changes the fixations (rule-placed on 13% of frames ->
uniform) AND the generator (joint -> factorised): neither change is isolated (Codex). Against P1 world 1 it changes the
decoder and the generator. Its role is the control for the learned chooser, and the first trained factorised world.

**Pre-launch checks** (`checks.py` -> `checks.json`): the real launcher for 25 updates (every logged term finite; time,
peak memory); resume with TR.p1 at a shrunken budget (4 straight vs 2 + resume + 2 against a second straight run,
bitwise); identity (validate_resume refuses P1-F's checkpoint under this config and this run's checkpoint without the
factorised key; the launcher refuses a directory with another identity).

**Not addressed tonight, before any adaptive chooser or reading:**
- Readings must use the cascade: saccade/eval/s1.one_step and Codex's sampler factorial sample the periphery and the
  glimpse jointly (Codex exercised the S1 path: zero cascade calls). Fixed and checked before the readings run.
- The chooser hook's inputs (Codex): `choose(p, v)` reads the clean generated periphery, the glimpse stage its
  conditioning-augmented version; with an adaptive chooser the window can carry information the conditional lacks.
  Inert under uniform fixations; to be reconciled with the real step's inputs before decision 2 is built.
**Readings:** planned and reviewed after the run, before any is run (declared there); candidates: the cascade-based
S1 / sampler readings, consequences and HUD, whether informative past glimpses reach later physical predictions
through the belief (decision 1's standing reading, TEACHING 10a), the reward head's consistency across fixations.

## Pre-launch checks: results (11-01 night; `checks.json`; all pass)
- First attempt: the real launcher ran out of memory after update 1 (default CUDA allocator: 4.0 GiB allocated +
  0.76 GiB reserved unused, the factorised path's temporaries fragmenting the 6 GB card). Fixed by expandable
  segments, set by the launcher and recorded in the identity (allocation only; no arithmetic changes).
- Launcher, 25 updates: every logged term finite. Resume: 4 straight = 2 + resume + 2 = a second straight run,
  bitwise (max abs 0, state equal). Identity: P1-F's checkpoint refused under this config, this run's refused
  without the factorised key, the launcher refuses a directory whose identity differs.
- Memory measured on fresh processes (nvidia-smi, every second): 64-step phase (60 updates) process max 4,624 MiB,
  card max 5,245 MiB of ~5,850 usable (desktop ~620 MiB): ~0.6 GiB headroom; 256-step phase with 3 checkpoints
  (encoder diagnostics) process max 4,492 MiB. Time: 1.66 s / update over the first 60 at 64 steps, 6.6 s at 256
  (the GPU's thermal state moves absolute times; decision 1's measured cost is +18-21%): ~11 h.
