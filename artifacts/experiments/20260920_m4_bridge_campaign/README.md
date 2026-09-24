# M4 bridge campaign — historical

**Superseded.** M4 is now canonical architecture inside `d4mj/`; run it with
`python -m d4mj paired-run`, `bridge`, `actor`, `evaluate`. The drivers that once lived here
(`bridge.py`, `actor.py`, `evaluate.py`, `train_joint_pair.py`, `test_campaign.py`) were moved into
the package, which is where they belong: they implement TC-14–18, and only code inside `d4mj/` is
pinned by the `sources.py` runtime closure into a checkpoint's source manifest. While they sat
here, the code computing the headline result was not hash-bound to any artifact it produced.

What remains, and why it was not deleted:

| file | why it stays |
|---|---|
| `PATCH_CONTROL.md` | the designed control for RISK_REVIEW risk 7. **Not yet run.** The open question about `z` versus CLS-plus-patch-grid is settled by this measurement, not by argument |
| `AUDIT_RESPONSE.md` | the 2026-09-20 audit and the repairs, including two defects of mine that no test caught |
| `PREDECLARATION.md` | historical. The canonical protocol is `spec/lewm/EVALUATION.md`; this records what was predeclared before any number existed |
| `build_corpus.py` | the **only** provenance for `artifacts/craftax_expert_store_v1`, which the canonical path consumes. `d4mj` has no corpus-build command |
| `flatten_forks.py` | provenance for `artifacts/craftax_forks_flat_v1`. TC-17 excludes fork corpora from canonical training, so that store is now unused — see the note below |
| `reseal.py` | the driver for re-measuring frozen-evaluation parity after an in-closure edit. `gate.py` has the primitives; this is the procedure around them |
| `evidence/` | corpus, fork-pool and flat-fork audits for the stores that exist on disk |

Deleted: the moved drivers, the disabled launcher, the superseded plan and design notes, the
campaign recipes (canonical ones are in `d4mj/recipes/`), and the 6.4 GB grouped fork pool, whose
condition was never selected and which TC-17 excludes.

**Still on disk, not deleted here:** `artifacts/craftax_forks_flat_v1` (15 GB). TC-17 makes it
unused by canonical training. Delete it if the flattened-exposure question is not going to be
revisited; keep it if it is, since rebuilding costs about 20 minutes.
