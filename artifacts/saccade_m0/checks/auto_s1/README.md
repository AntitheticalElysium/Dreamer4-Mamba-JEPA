# One-off S1 after P1 (2026-10-27)

User requested automatic S1 after the current P1 completes. This adds a native runtime completion hook, not a timer
or recurring Codex automation. P1 code, configuration, losses and frame budget remain unchanged.

## Selection and retention

- Train to the full 25.6 M frames. Save and keep 10 numbered checkpoints, every 2.56 M frames, plus latest.pt
  (a full duplicate of the newest checkpoint). Current files imply about 2 GB per world seed at completion.
- Default selection: checkpoint 10.
- Exception: if fixed monitor G1 loss rises on three successive comparisons anywhere in the run, use the global
  minimum over all ten readings. Three increases require four observations; equality breaks a streak; earliest
  checkpoint wins an exact minimum tie. This makes the existing CURRICULUM wording executable before final results.
- Semantic probes/dev outcomes do not choose the checkpoint. All checkpoints remain available.

## Execution

`install_hook.py` creates the runtime-linked saccade-s1-world1.service and adds OnSuccess to the current running
saccade-p1-world1.service via a runtime drop-in and daemon-reload. It asserts the P1 PID and invocation stay unchanged.
The evaluation unit is inactive until P1 exits. It is bound to the original invocation and the current S1 source hash.

`python -m saccade.eval.after_p1 RUN_DIR 1` accepts only systemd's successful, normal exit metadata for that invocation,
then checks all monitor readings/checkpoints, completed budget and cfg/seed/source/data identity. It writes selection
and checkpoint hash, then execs the existing full S1 entry point with no episode limits. Exec frees validation tensors
and makes the unit exit status S1's status. S1 has 24 GB host RAM and six CPU threads; no evaluation GPU work overlaps P1.

Outputs, under artifacts/saccade_m0/p1_world1/s1/:

- selection.json: fixed-rule decision, ten monitor readings, checkpoint hash and command;
- results.json and results.reader.pt: full readings and exact frozen reader;
- eval.log: S1 stdout/stderr, including any failure.

Hook validation errors before S1 is exec'd appear in `journalctl --user -u saccade-s1-world1.service`; eval.log is
created only when the full S1 child is about to start.

No P2, second world seed or actor training is launched. A failed evaluator is not automatically retried; an existing
selection also blocks repeated attempts. A completed evaluator is not itself a scientific pass: inspect its readings
and blockers. Runtime hooks do not survive a machine reboot; neither does this transient P1 training process.

## Checks

- CPU regression tests: selection/default/three increases/ties/later recovery; missing, duplicate, reordered, nonfinite
  or incomplete monitor readings; fixed window count; bound successful exit vs failed/stopped/wrong invocation;
  end-to-end handler selection and full-S1 command, checkpoint retention and repeated-attempt refusal (`tests.txt`).
- Live CPU-only throwaway service test: runtime hook attached while producer was running, preserving PID/invocation;
  successful natural exit starts consumer with the exact expected MONITOR_* metadata; exit 7 does not start consumer.
  Both test cases clean up their units and runtime files. Production P1 PID 30881 and invocation
  a467ea58282948e5883e02590cf03ca2 were unchanged (`systemd_hook.json`).
- A float-order defect in the first selector draft was caught by the tests; frame thresholds now use the same
  `frames * fraction`, then checkpoint index multiplication, as train.py. No production run used that draft.
- `installation.json` records the actual attached production hook and inactive evaluator. No GPU tests or S1 were
  run alongside production P1. This change does not alter model training or claim anything about learned quality.

Primary documentation checked against installed systemd 262:
[OnSuccess](https://www.freedesktop.org/software/systemd/man/latest/systemd.unit.html#OnSuccess=),
[handler exit metadata](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html#%24MONITOR_SERVICE_RESULT),
[daemon-reload](https://www.freedesktop.org/software/systemd/man/latest/systemctl.html#daemon-reload).

Reproduce CPU checks with `.venv/bin/python -m pytest saccade/tests/test_after_p1.py -q`. The live hook test is
`.venv/bin/python artifacts/saccade_m0/checks/auto_s1/test_systemd_hook.py` (requires user-systemd access). Installer
requires the current P1 still running: `.venv/bin/python artifacts/saccade_m0/checks/auto_s1/install_hook.py`.
