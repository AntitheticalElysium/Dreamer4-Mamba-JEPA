# Canonical M4 baseline — launched 2026-09-21

| | |
|---|---|
| command | `python -m d4mj paired-run --dataset craftax_expert_store_v1 craftax_support_v2 --out artifacts/lewm_m4_canonical` |
| recipes | `d4mj/recipes/lewm_mamba_raw_m4.json` / `lewm_mamba_tc_m4.json` (CLI defaults) |
| pair axis | `variant` — raw vs tc differ in nothing else |
| recipe digests | raw `a740b7feefbb119a`, tc `bf9175a0644cef25` |
| dataset_id | `11a0e733d2067a0f` |
| corpus | 8,325 TRAIN / 1,041 DEV / 1,034 FINAL, 3,875,808 transitions; 256 BC-eligible TRAIN episodes (552,998 transitions) |
| forks | **excluded** — TC-17 refuses both fork stores by manifest kind (verified) |
| joint | B128, 10,000 updates, G1 at 2,000, lr 5e-5, SIGReg 0.09, stride 1 |
| bridge | batch 16 + 4 terminal, 32 frames (128 every 4th), H2 2,000 → H16 8,000, depth 2→16 |
| actor | horizon 16, batch 16, screen 500, total 5,000; lr 1e-4, warmup 1,000, RMS 0.99 |
| evaluation | 512 paired seeds, actor vs its own immutable BC |

## What this run reaches, and where it stops

Stages 1 and 2 only: the 10,000-update paired joint world for both arms, then a 2,000-update H2
bridge for each. **It then stops at the G2/G3 gate.**

`require_bridge_gate` needs a `d4mj_lewm_bridge_gate_v1` report whose seven components each carry
non-empty metrics and at least one byte-bound evidence file. **No evaluator produces one** —
`STATUS.md:42` states that producing the empirical G2–G4 evidence is a separate job. Writing a
status-only JSON to get past it would be manufacturing the evidence the gate exists to demand.

So H16, the actor and real evaluation are blocked until the G2/G3 evaluator is built from
`spec/lewm/EVALUATION.md` §G2 and §G3.

## Resumability

Every stage resumes in place; `run.sh` is idempotent and re-running it continues.
- `paired-run --resume` re-verifies the screen recipe, each arm's recipe, the dataset identity and
  the pre-training pair seal before continuing, and refuses if any changed.
- `bridge --resume <checkpoint>` continues from the last snapshot.
