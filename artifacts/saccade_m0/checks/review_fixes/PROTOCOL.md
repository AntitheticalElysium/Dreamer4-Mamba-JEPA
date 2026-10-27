# Pre-P1 correction verification (2026-10-27)

Scope: close the implementation audit's R1-R7 and smaller completeness findings without changing the M0 architecture,
production budgets, dev/final split or scientific stopping rules. Production P1 is not launched in this turn.

1. Run all S0/M0/S1 regression checks on CUDA, including exact-target sampler consistency, full-history cache parity,
   resume identity and unchanged gradients/RNG under periodic diagnostics. Failure is a code defect to repair.
2. Use audit seed 1898 for 20 bounded real-data updates, 16 streams x 64 steps, first 64 train episodes. For this
   mechanical run only, cap frames at 20,480, warmup at 1,024, checkpoint every half-budget, monitor 4 held-out windows.
   Preserve all model shapes, losses, routing, optimizer and sampler settings. Save a fresh provenance-bound checkpoint.
3. Run the full S1 entry point on that checkpoint with 15 dev and 30 train-400 episodes, as in the historical smoke.
   Require every section and reader artifact to be written, finite scores where exposed, correct schema/identity and
   explicit unmeasured blocker status where necessary. No smoke score is a P1 gate or semantic success claim.
4. Measure one full 4 x 4 x 256 accumulated update with the periodic gradient diagnostics enabled. This verifies the
   additional diagnostics fit the hardware; no optimization or learning conclusion follows from one update.
5. Recheck source/data identities, whitespace, the explicit-working-directory launch command, and the committed diff.
   Record commands and outputs; do not rerun historical mixture/ablation training or inspect final-split examples.

LightGBM 4.6.0 is installed in the persistent project venv and pinned in requirements.txt. Original audit evidence and
historical negative results remain intact. Any unexpected failure and correction is retained in RESULTS.md.

Implementation correction during verification: the initial full retained-graph diagnostic OOMed at the first
production-size checkpoint (smoke_initial_oom.out). It is replaced by one stream capped at 64 new frames, recomputed
after training backward using saved incoming history, before optimizer.step, with RNG restored and .grad untouched.
The same seed/data/20-update smoke is rerun from scratch. This changes diagnostics only, not a learning rule or budget.

Evaluator correction during verification: after completing every S1 section, the full-input tree classifier scored
0.5 on shuffled histories and mismatched actions. On a balanced controlled pair (x,x) versus (x,-x), it also scored
0.5 despite completely different dependence and exactly equal marginals. Reuse the existing S1 MLP/fit schedule,
with train-only standardisation and equal expected energy per input group (history, belief, action, fixation,
outcome). Unbalanced MLP input caught history but not action; retain both negative outputs. The single group-scaling
correction is checked on the same fixed tuples, then S1 is rerun on the same unchanged 20-update checkpoint. No
production/world training budget is extended, and no failed power check is treated as a pass.
