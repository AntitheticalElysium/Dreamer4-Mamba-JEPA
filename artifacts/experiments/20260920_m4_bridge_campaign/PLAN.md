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

**B1 — Source closure orphans every sealed LeWM checkpoint.**
`sources.py:lewm_source_manifest` hashes an 18-file runtime closure; `checkpoint.py:56` calls
`verify_sources`, which **raises** `checkpoint source drift` for LeWM on any mismatch.
M4 requires editing at least `world_api.py` (`require_control`) and `experiments.py` (the CLI phase
gate) — both in closure. Editing them makes Raw-10k, TC-10k, every m03 gate and every arms parent
unloadable on the working tree.
*In closure:* `__main__ cache checkpoint config data diagnostics execution experiments gates
imagination lewm lewm_config mamba_recurrence sources state train world_api`
*Out of closure (free to edit):* `actor_critic agent backbone counterfactual env expert
lewm_transformer representation time_mixer`
→ **Resolution: tag the tree before the first in-closure edit.** Historical checkpoints stay
verifiable at the tag; the fresh run seals under the new manifest. Reversible, cheap, recorded.

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
- [ ] `0.1` Tag current tree `lewm-closure-m3` + record the manifest digest; note which artifacts
      remain verifiable only at that tag.
- [ ] `0.2` Verify a sealed checkpoint (Raw-10k) loads at the tag, and record the exact failure it
      will give afterwards — so the break is demonstrated, not assumed.
- [ ] `0.3` Plan every signature before writing (code contract): name each function, module, inputs
      and outputs for the readout + head surface; check against the architecture draft. **No new
      files, no new functions** without re-opening the design here first.

### Stage 1 — corpus
- [ ] `1.1` Build the merged corpus contract: archive TRAIN + support TRAIN, separate split
      provenance, `bc_eligible` carried per episode.
- [ ] `1.2` Verify frame geometry agrees between sources (dtype, range, 63×63, channel order) —
      byte-level, on real samples, not by inspection.
- [ ] `1.3` Confirm no DEV/FINAL episode from either source can enter joint training.
- [ ] `1.4` Record episode/transition counts and the BC-eligible subset size.

### Stage 2 — M4 bridge (out-of-closure first, in-closure last)
- [ ] `2.1` `z+h` readout + BC head, trained on observed paths.
- [ ] `2.2` Reward + continuation heads, with counterfactual branch coverage **separately declared**.
- [ ] `2.3` H2 → H16 recursive generated-prefix training, continuing the **same** jointly trained
      world — no Mamba restart.
- [ ] `2.4` Unblock `require_control` / the CLI phase gate under a real authorization condition
      (trained readout + heads + validated horizon), not a bypass flag.

### Stage 3 — predeclaration (before any fresh training)
- [ ] `3.1` Predeclare the primary comparison: panel (real Craftax DEV episodes), episode count,
      seeds, the interval, and what counts as "actor beats BC". Commit it **before** Stage 4.
- [ ] `3.2` Predeclare the staged stop points and what each failure assigns:
      - poor observed-path BC → representation / readout / data
      - good BC, poor generated heads → world / recursive bridge
      - good generated heads, actor < BC → critic / policy / imagination (the Direct failure)
      - actor > BC → genuine signal; replicate training seeds

### Stage 4 — the run
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
