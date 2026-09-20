# Response to the 2026-09-20 audit

**Every blocker is confirmed. None was denied.** The audit was right that the shared suite stayed
green through all of it because nothing exercised the campaign drivers — that gap is now closed by
`test_campaign.py` (18 tests targeting exactly these contracts).

## Blockers — all CONFIRMED and repaired

| # | finding | verification | repair |
|---|---|---|---|
| 1 | 10k launch cannot cross G1 | `train.py:623` requires a sealed screen past `screen_step`; my `pair.json` held only `initial_identity` and `axis`, while `require_joint_screen` reads `dataset_id` and `screen_settings_id` from it | `train_joint_pair.py` now runs 2k → canonical pair seal → `screen_joint_pair` → accepted budget, and passes `screen_report` through |
| 2 | bridge is not the specified Phase-2 bridge | `ARCHITECTURE.md` §5 and `DECISIONS.md` specify it in detail; the implementation had **no dynamics loss at all** | rebuilt: `mean(teacher SE) + mean(recursive SE)`, heads `0.5·prefix + 0.5·(0.5·obs suffix + 0.5·gen suffix)`, continuation 0.8/0.2, independent group RMS 0.99, batch 16+4, 32 frames with 128 every fourth, 2k H2 → DEV gate → 8k H16, warmup 1000 then constant |
| 3 | actor not optimization-equivalent to Direct | it summed raw losses at constant lr | now uses the inherited `_balance` (RMS 0.99) and `_update` (warmup 1000 then constant); budget is the spec's 500-update screen / 5,000 total, batch 16 |
| 4 | recipe presence authorizes control | **reproduced**: a freshly constructed, never-trained M4 bundle passed `require_control()` | authorization now reads a **recorded capability** (`readout_trained`, `validated_recursive_depth`); recipe intent alone is refused, and `validated_recursive_depth` is written only by the DEV gate |
| 5 | resume/identity incomplete | confirmed across bridge, actor, evaluation and the grouped fork stream | bridge/actor checkpoints carry optimizer, sampler RNG, RMS state, parent and dataset digests; both refuse to start over existing output and refuse to overwrite numbered snapshots; the fork stream is reseeded **from the update index**, so a post-G1 resume reproduces its draws with no extra state |

## File-by-file — CONFIRMED

- **`data.py` single-source returned a plain `list`** — reproduced (`list False 320`). This was a
  regression I introduced: it silently disabled `EpisodeCorpus.pools` for every existing caller.
  A single source now returns the object `load_episodes` built, unchanged.
- **Bridge did not seed `Heads`** — confirmed; Raw and TC differed by an ambient RNG draw before a
  single update. Now `torch.manual_seed(seed + 2)`, with the resulting head identity recorded and
  checked on resume.
- **`evaluate.py` cache not bound** — confirmed. Episodes are now keyed to an identity over actor,
  joint, horizon, seed base and protocol; a mismatched directory is refused.
- **Fork pool manifest had no shard hashes** — confirmed (`['file', 'roots']`). Now hashed.
- **`run_campaign.sh` MODE typo silently selected grouped** — confirmed. Now `flat|grouped` only,
  exit 2 otherwise; the false "resumable / refuses clobber" claim is replaced with what is true.
- **`reseal.py` reused cached dumps and an unchecked worktree** — confirmed. Dumps are now keyed by
  the manifest digest of the tree that produced them, and the worktree's HEAD is verified against
  the tag before use.
- **`PREDECLARATION.md` understated exposure** — confirmed: it cited 8,325 TRAIN episodes, which is
  the two-source figure. Flat mode's world sees **236,924**. Corrected, with the bridge/actor
  figure (8,325) and BC figure (256) stated separately.
- **`DESIGN.md` stale** — confirmed: it named functions never written. Rewritten to the code.
- **Draw share** — confirmed **12.35%**, not the 12.1% predicted or the 11.6% I reported; mine was
  a 40-update finite sample. The audit's corpus figures reproduce exactly: 265,672 episodes,
  4,872,668 transitions, TRAIN/DEV/FINAL 236,924 / 27,714 / 1,034.
- **G1 excludes the short flattened episodes** — confirmed: `screen_windows` needs
  `len(e) + 2 - span >= windows_per_episode`, i.e. ≥ 6 steps, and flattened forks carry 3–4. G1
  therefore screens expert and support only. Now **recorded in `campaign.json`** rather than silent.

## What the audit did not cover, and I found while repairing

Implementing the DEV gate surfaced something worth stating before the run: **persistence is a very
strong baseline in this latent space.** On DEV, repeating the anchor latent for every step gives
MSE 0.0016 at H=2 and 0.0051 at H=16. Consecutive `z` are nearly identical, so "predict no change"
is already close to optimal under MSE, and a trained world must beat that to have its depth
validated. This is the same geometry the retracted allocation work ran into from the other side —
the directions that carry outcomes hold a very small share of δz variance. It means the H2 gate is
a real risk, not a formality, and a failure there is informative rather than merely blocking.

## Status

Full suite **311 passed, 4 skipped, 0 failed** (d4mj plus the new campaign tests). Pipeline
re-verified end to end at smoke scale: joint → bridge (both DEV gates, H2→H16, 128-frame long
batch) → actor (RMS, warmup, horizon from the recorded capability) → real Craftax evaluation.
Smoke artifacts are self-declaring: a smoke bridge refuses to feed a non-smoke actor, and a smoke
actor refuses to feed a non-smoke evaluation.

**Not launched.** The remaining item is the audit's request to either implement the spec or
predeclare a replacement — implemented, so the predeclaration stands as written with the corpus
figures corrected.
