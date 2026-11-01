# Disk cleanup of the closed LeWM / D4MJ line (2026-10-27, user-approved)

Freed 870.3 GiB (disk 97% -> 51%). Rule: delete only regenerable bulk, keep everything that carries information.
- Deleted (26,011 files): caches, pools, tapes and rollout tensors (`.pt` / `.f16` / `.npy` / `.npz` / `.sqlite3` >= 1 MiB
  not recognised as checkpoints), and intermediate checkpoints (all but the last of each model's numbered series).
- Kept: every git-tracked file; all text / results (json, jsonl, md, log, csv, png, py, ...); every binary < 1 MiB; the
  final checkpoint of each model (series grouped by folder and file-name pattern): `kept_final_checkpoints.tsv`.
- Scope: `artifacts/eda/*` except `s0` and `system_design_v1`; 47 old top-level `artifacts/*` run folders; `data/`,
  `d4_mamba_jepa/artifacts`, `.archive` (unused by saccade or anything current). Untouched: the support-v2 corpus,
  the expert store, `artifacts/experiments`, `third_party` (lock files), `.venv`, `saccade`, `d4mj`.
- Every listed path was re-checked before removal (inside the approved scope, untracked, size unchanged since the
  inventory). 140 entries were the same files reached through `forkset_*` symlinks to `forkset_s1_n64`, so the true
  total is 870.3 GiB, not the inventory's 880; the links remain (their target keeps its small / text files).
- Consequence: old results can no longer be re-verified from their raw caches; their numbers, logs and final models
  remain. `deleted_by_folder.tsv` (per folder), `deleted_paths.tsv.gz` (every path, bytes, reason).
