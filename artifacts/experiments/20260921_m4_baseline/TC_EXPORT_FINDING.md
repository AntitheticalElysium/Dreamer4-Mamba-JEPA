# TC: a low-rank / scale training pathology, and a failed cache export

**Raw-only baseline.** TC never produced a TC-14 compliant latent cache, so this run yields **no
Raw-versus-TC H2 architecture comparison**.

## The pathology

Effective rank and scale on the **sealed G1 DEV window ledger** — the same 512 windows G1 screened,
2,048 frames, reproduced by `rank_trajectory.py`; the 2,000-update raw value reproduces the sealed
screen's own `23.44675` exactly.

| update | raw rank | tc rank | raw mean \|z\| | tc mean \|z\| |
|---:|---:|---:|---:|---:|
| 2,000 | **23.45** | 9.87 | 0.731 | 0.959 |
| 4,000 | **31.50** | 3.93 | 0.740 | 1.568 |
| 6,000 | **34.68** | 3.69 | 0.739 | 1.604 |
| 8,000 | **38.09** | 3.79 | 0.746 | 1.577 |
| 10,000 | **38.32** | 3.94 | 0.742 | 1.596 |

**TC was already 2.4× behind raw at G1** (9.87 against 23.45, with normalized prediction MSE 0.681
against 0.059), and fell to 3.93 by update 4,000 while raw rose to 38.3. Scale inflates as rank
falls: TC's mean |z| goes 0.959 → 1.568 over the same interval while raw's holds near 0.74.

This is a **training pathology of the TC arm** — low rank with inflated scale — not a gate defect.

## Why G1 passed it

G1's criteria are learning progress against each arm's own initialization and a limited
proxy-retention rule. TC improved over its initialization and did not trip that rule, so it passed
**as written**. Two gaps are exposed:

- **Criterion gap.** G1 records the covariance spectra but does not gate on rank or scale, so a
  representation 2.4× poorer than its pair advanced on equal footing.
- **Schedule gap.** G1 screens at 2,000 and the next representational check is the export at
  10,000. TC's collapse lands at 4,000, in the unobserved interval.

## The export failure: correlated, not causal

The export's batch-invariance check compares a frame encoded alone against the same frame encoded
in a chunk, at `atol=1e-5, rtol=1e-4`. TC fails it; raw does not.

A matched 60-frame check across checkpoints shows the failure is **sample- and batch-dependent**:

| checkpoint | failing coordinates (matched 60-frame sample) |
|---:|---:|
| 2,000 | 0 |
| 4,000 | 73 |
| 6,000 | 4 |
| 8,000 | 8 |
| 10,000 | 0 |

while the full exporter found 14 failures at 10,000 on different frames. Scale inflation makes an
absolute-tolerance breach more likely — larger activations carry larger absolute error from the
same relative precision — but **collapse does not mechanically cause export failure**. The honest
statement is that export failure is a numerical-contract failure *associated with* the pathological
representation, on a batch- and sample-dependent basis.

An earlier version of this file claimed the cause was near-zero coordinates left by temporal
centering. That was wrong — TC has *fewer* near-zero coordinates than raw — and it is removed
rather than annotated.

## Rank is a symptom, not a target

Raising rank does not restore capability. The 2026-09-16 centering-window experiment lifted TC's
rank from **5.14 to 19.16**, and the projected-`z` behaviour cloning nonetheless *"collapsed to the
action prior"*, with all-action outcomes showing no significant improvement. Any TC follow-up that
selects on rank is selecting on the wrong quantity.

## What a TC follow-up needs

1. Measure critical retention and an action-relevant capability proxy at 2k / 4k / 10k, alongside
   prediction-versus-SIGReg gradient norms and their cosine — is the regularizer overwhelming the
   prediction term as rank falls?
2. **Do not** rerun the same recipe, and **do not** select widened centering because it restores
   rank.
3. Add a sealed **4,000-update review** carrying normalized prediction, rank and scale, critical
   retention, and an action-relevant proxy. Rank alone must never authorize continuation.

## Evidence

`evidence/rank_trajectory.json` — every checkpoint hash, the ledger hash, the script hash, and the
raw measurements. Script: `rank_trajectory.py`.
