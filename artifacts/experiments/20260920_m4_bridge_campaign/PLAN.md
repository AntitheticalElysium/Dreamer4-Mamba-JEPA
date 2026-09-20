# M4 bridge + fresh paired Raw/TC through real actor evaluation — 2026-09-20

**Primary result:** actor achievements vs **its own BC**, on real Craftax, for each arm.
Everything else is diagnostic.

**Standing rationale.** Direct reached fork retrieval `0.878` and geometry `0.989` and its actor
still lost to its own BC by 0.6–1.5 achievements on 512 real DEV episodes. Good fork geometry and
good semantic probes are not sufficient selectors. M03 and the fork/semantic probes run throughout
as **diagnostics that explain failures — never as authorization barriers.**

**Not in scope.** No SIGReg replacement, no MSE replacement, no Sub-JEPA / JEDI / V-JEPA2 objective
port. The 2026-09-19 arms gave no downstream reason to touch the objective, and their "loss form"
verdict was retracted (`fc6c008a`). Counterfactual branch supervision stays **separately declared**
where reward/continuation heads need action coverage; it does not silently become the LeWM
objective.

---

## Blocking findings from the pre-flight audit

**B1 — Source closure does NOT orphan sealed checkpoints. There is a designed re-seal path.**
*(Corrected 2026-09-20 after measuring the actual failure rather than predicting it.)*

Appending one comment to `world_api.py` and reloading Raw-10k gives, not the `verify_sources` drift
error I expected, but an earlier and more informative guard:

```
ValueError: m03_frozen_eval: proof does not describe the current tree
```

`load_m03_bundle` tries exact source equality first, and **on failure falls back to a measured
parity proof** (`d4mj/m03/frozen_eval_compat.json`). That file already holds **two** proofs from
previous tree changes, and proof[1] already covers **13 changed runtime files including
`d4mj/world_api.py`** — so this path is not theoretical, it has been walked twice.

The proof is a real measurement, not a bypass. `frozen_eval_parity` (gate.py:738) dumps an 8-tap
surface — `z, cls, patch_grid, patch16, prefill_latent, prefill_history, advance_latent,
advance_history` — from fixed seeded inputs; `frozen_eval_proof` (gate.py:805) compares two trees
and passes only when `cross_tree_max_abs <= max(tolerance, within_tree_max_abs)`. The repeats
matter because `advance` is not run-to-run reproducible, so a cross-tree gap is meaningless until
each tree's own spread is measured. The standing proof sits at **5.96e-07 against a 1e-05
tolerance**.

Crucially `scope: "frozen evaluation only; training resume keeps exact full-source equality"`, and
`test_frozen_eval_never_relaxes_training_resume` enforces that ordering in source. Training resume
from an old checkpoint stays hard-blocked — which is correct, and irrelevant here because Stage 4
trains from zero.

→ **Resolution:** tag the pre-M4 tree (done: `lewm-closure-m3`), and after the in-closure edits
re-measure parity from a worktree at that tag and append a third proof. M03 diagnostics on sealed
checkpoints keep working. `d4mj/m03/*` is itself outside the closure, so the tooling is free to use.

**B2 — M4 is mostly wiring, not construction.**
`agent.py`, `imagination.py`, `actor_critic.py`, `execution.py` already exist from the Direct line.
Genuinely missing for LeWM: the `z+h` readout surface, head fitting, and the unblock. This is
smaller than "build M4" implies — which the code contract (no new files, no new functions) requires
us to exploit rather than rebuild.

**B3 — The corpus merge is real work, not a flag.**
Archive `d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt` is 8.6 GB, stored CHW and sliced
`[:, :, :63, :63]` then permuted to HWC (`eda/corpus.py:frames_of`). Support-v2 is a sharded
`d4mj_episode_store_v1`. Split provenance differs: archive via `episode_splits(len, seed+0)`
*permutation order*; support via its on-disk `sha256(seed:round:slot)` 80/10/10 field. Both must be
preserved, not re-derived. LeWM's `data.py` + dataset contract currently accept one manifest.

**B4 — BC's support is the expert archive alone.**
`craftax_support_v2/manifest.json` has `bc_eligible: false`; only the archive is eligible. Since
the primary result is "actor vs **its own** BC", the BC is trained on the archive subset while the
world sees both. State this in the result; do not let it become an unexamined asymmetry.

---

## Checklist

### Stage 0 — seal the past, unblock the future
- [x] `0.1` Tag the pre-M4 tree — `lewm-closure-m3`. Manifest digest
      `76cb5826…` (18 closure files); `world_api.py` at `032b2a0d…`.
- [x] `0.2` Break demonstrated, not assumed. Raw-10k loads clean at the tag (2,503,496 world
      params). One appended comment in `world_api.py` →
      `m03_frozen_eval: proof does not describe the current tree`. Reverted. **Finding: this is
      recoverable by re-measuring the parity proof — see B1.**
- [x] `0.2b` **DONE** — parity 5.96e-07 cross-tree, equal to the within-tree floor, against a
      1e-05 tolerance, 5 changed runtime files, 2 runs per tree. Both sealed 10k checkpoints now
      load by measured proof and stay correctly blocked from control. Superseded detail: After the Stage-2 in-closure edits: worktree at `lewm-closure-m3`, run
      `frozen_eval_parity(..., allow_drift=True)` **twice in each tree**, then `frozen_eval_proof`
      at tolerance 1e-5, and append the result to `frozen_eval_compat.json`. Must pass on its own
      measurement — if parity fails, the edit changed frozen-evaluation numerics and the design is
      wrong, not the guard.
- [x] `0.3` **DONE** — see DESIGN.md. Plan every signature before writing (code contract): name each function, module, inputs
      and outputs for the readout + head surface; check against the architecture draft. **No new
      files, no new functions** without re-opening the design here first.

### Stage 1 — corpus
- [x] `1.1` **DONE** — 10,400 episodes, 8,325 TRAIN, 3,875,808 transitions. Build the merged corpus contract: archive TRAIN + support TRAIN, separate split
      provenance, `bc_eligible` carried per episode.
- [x] `1.2` **DONE** — `validate_episode` ran on all 10,400: dtype, 63x63x3 geometry and the
      no-early-reset invariant checked per episode, not by inspection. Verify frame geometry agrees between sources (dtype, range, 63×63, channel order) —
      byte-level, on real samples, not by inspection.
- [x] `1.3` **DONE** — splits preserved per source; `JointSampler` draws only `split=='train'`. Confirm no DEV/FINAL episode from either source can enter joint training.
- [x] `1.4` **DONE** — 256 BC-eligible TRAIN episodes / 552,998 transitions, vs 0 before. Record episode/transition counts and the BC-eligible subset size.

### Stage 2 — M4 bridge (out-of-closure first, in-closure last)
- [x] `2.1` **DONE** — bridge.py, observed phase. `z+h` readout + BC head, trained on observed paths.
- [x] `2.2` **DONE** — heads fitted; fork coverage separately declared in train_joint_pair.py. Reward + continuation heads, with counterfactual branch coverage **separately declared**.
- [x] `2.3` **DONE** — depth schedule verified ramping 2 -> 9 -> 16 on the same world. H2 → H16 recursive generated-prefix training, continuing the **same** jointly trained
      world — no Mamba restart.
- [x] `2.4` **DONE** — `require_control` passes only on a recipe declaring `agent`; no bypass flag. Unblock `require_control` / the CLI phase gate under a real authorization condition
      (trained readout + heads + validated horizon), not a bypass flag.

### Stage 3 — predeclaration (before any fresh training)
- [x] `3.1` **DONE** — PREDECLARATION.md. Predeclare the primary comparison: panel (real Craftax DEV episodes), episode count,
      seeds, the interval, and what counts as "actor beats BC". Commit it **before** Stage 4.
- [x] `3.2` **DONE** — PREDECLARATION.md. Predeclare the staged stop points and what each failure assigns:
      - poor observed-path BC → representation / readout / data
      - good BC, poor generated heads → world / recursive bridge
      - good generated heads, actor < BC → critic / policy / imagination (the Direct failure)
      - actor > BC → genuine signal; replicate training seeds

### Stage 4 — the run  *(built and smoke-tested end to end; awaiting review before launch)*
- [ ] `4.1` Raw from step 0, merged corpus, no frozen encoder, no continuation.
- [ ] `4.2` TC from step 0, identical in every respect but the SIGReg centering.
- [ ] `4.3` Continue each into M4: readout, BC, heads, H2→H16.
- [ ] `4.4` Train and **execute** the actor. Real Craftax, actor vs its own BC.

### Stage 5 — read out
- [ ] `5.1` Primary result per arm against the Stage-3 predeclaration.
- [ ] `5.2` M03 / fork / semantic probes as explanation of whichever stage bound.
- [ ] `5.3` Only if a stage is identified as binding: name the objective change it implies, and
      only then consider Sub-JEPA / JEDI / V-JEPA2 evidence.

---

## Budget note
6 GB card. The original paired run was 10k updates per arm on support-v2 alone. Merged corpus is
larger and M4 adds head fitting, H2→H16 and actor rollouts on top of two arms. Stage 4 is to be
scoped against measured step time before launch, not assumed.
