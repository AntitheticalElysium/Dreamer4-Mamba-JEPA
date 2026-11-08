# P1-F1.6 pre-launch checks

2026-11-02. Approved joint64 replacement, not the earlier optional16D prototype.
No scientific success claim is made from this smoke.

- Full regression suite:63 passed in33.97s, including all five new fusion checks.
  Existing parameter initialization and RNG preserved; no raw-fine bypass;
  locality/crop ordering, input gradients, strict reload and post-warm-up native
  future-G1 gradients checked. `all_tests_gpu_final.log`.
- Queue wrapper/selection tests:3 passed in2.72s. Output isolation, world labels,
  same checkpoint-selection rule and refusal of duplicate evaluation attempts.
- Actual launcher ran2 updates. Exact recipe difference versus F1.5 is the saved
  fusion config only. No world parameters frozen, no objective term removed.
- Sustained native trainer:200 real64-step updates, all terms finite. Last interval
  1.614s/update. Original25.6M-frame schedule/warmup, actual data/targets/fixations.
- Full2,000-window monitor executed after the sustained smoke. S1 encoding and
  actual cascaded generation executed on the trained smoke model: finite periphery,
  fine and lighting outputs with the expected shapes.
- 256-step phase:3 real updates with4-way gradient accumulation and checkpoint
  diagnostics/monitor. Recorded6.06,6.81,7.09s/update (includes varying checkpoint
  overhead between records). All terms finite.
- Interrupted2+2 versus uninterrupted4: maximum model/carry/cache discrepancy0.0;
  frame/update state, CPU RNG and CUDA RNG equal. Missing-fusion resume refused.
- Old F1.5 checkpoint loads strictly. New checkpoint loads and runs on CPU.
- Initial real-token RMS:raw fine0.92847, joint0.92336, so the replacement does not
  start with an accidentally attenuated fine stream. Standard existing out_map,
  no new learned noise/normalization parameters.
- Native future-G1 gradient norms after200 full-objective updates: first weight
  2.27e-6, first bias2.57e-7, output weight2.68e-6, output bias7.58e-7. Connected,
  finite, nonzero; not evidence of learned semantic benefit.
- Entire sustained/checkpoint/reload check450.34s. GPU peak allocated4.251GiB,
  reserved4.434GiB. Source pinning/import validation covers48 repository files;
  all fixed evaluation inputs exist. Bash syntax and Python compilation pass.

Earlier failed checks are preserved: the optional prototype's initialization-G1
test incorrectly expected a gradient through zero output heads. The first joint
test retained the carry graph across chunks, unlike production `carry_put`; it
failed on second backward. The harness was corrected to production detachment;
no native trainer change. The final suite and sustained real trainer pass.

Raw evidence: `checks.json`, `trainer_smoke.log`, test logs, `dispatch.json`,
`SOURCE_PINS.json`. Smoke checkpoint mechanics use a shortened budget/eight monitor
windows only in the separate `checks/` directory. The sustained smoke/full monitor
and production run use the actual recipe; no check checkpoint initializes P1-F1.6.
