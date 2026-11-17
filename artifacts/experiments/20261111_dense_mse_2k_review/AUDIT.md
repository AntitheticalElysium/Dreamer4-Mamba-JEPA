# Live 2,000-update extension audit

Read-only checks on the active run, 2026-11-11 Sydney. No GPU work, process changes, or edits to the run.

- Endpoint resumed at 384 (startup log; exact copied 384-update log prefix), then completed exactly 2,000.
  CPU replay from the original endpoint-384 stream and horizon RNG matches the saved 2,000 state exactly:
  all trace target/episode/position fields, stream history, and horizon draws. Adam counters are 2,000.
- Dense starts from its own byte-identical original 384 checkpoint and log. Verified before its next
  saved checkpoint replaced that seed copy. Observed live handoff and update385; it selected dense MSE.
- All 19 extension pins match. Loss and readout code match the short run byte for byte. AST comparison
  verifies identical model constructor and training loop; new cap is 2,000 total, not 2,000 additional.
  LR does not ramp again after 384. Parent, target store, manifest and original gaze trace agree.
- Endpoint's 2k inference export binds to the 2k training checkpoint SHA; only the mask token is removed
  and every native model tensor compares bitwise equal.
- Minor provenance limitation: the launcher did not record a separate seed-checkpoint digest at endpoint
  startup. The log, restored optimizer count and exact stream/horizon replay support the continuation;
  the already-replaced endpoint seed file cannot now be compared byte for byte.

After dense finishes, worker runs native hunger pairs for both 2k exports with SHA/reader qualification,
then the same risky, adjacent, coal and iron guards. Do not infer memory repair from training loss.
For analysis compare due/not-due rates and paired dense-vs-endpoint contrast, alongside both 384
checkpoints and the parent, and check factual guards. Still one training seed.

At verified handoff: endpoint complete; dense running from384. Endpoint median update2.287s after384,
so remaining dense training from the handoff estimated ~62 minutes plus readout. See live STATUS.json.
