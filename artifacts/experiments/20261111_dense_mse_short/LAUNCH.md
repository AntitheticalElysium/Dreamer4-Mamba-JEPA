# Capped comparison launched

User explicitly ruled out a full run. Serial systemd unit `saccade-dense-mse-short.service`,
invocation `2b4fc9c100874b0bb67d850193adf061`. Endpoint 384 updates, dense 384, then native readouts.
No escalation or full P1 run. Two-arm estimate ~30 minutes training plus evaluation (2.29–2.43s per
update on the real-update smoke after startup); not a measured completion time.

CHECKS.json: rebased fork RMS ~.0039 against continuous CUDA pass on affected rows versus ~.25/.26
unrebased. The unaffected row retains numerical-only split drift. Both real two-update arms succeed,
first-update ordinary terms bitwise equal, different auxiliary reductions, peak ~2.50 GiB.
Both inference exports load strictly. Parent native readout reproduces all saved digit counts.
The factual guard bank qualifies with 442 risky, 2523 adjacent, 48 coal and 44 iron histories.
RUN_PINS.json validated before every worker phase. Production and historical files unchanged.

Current status in run/STATUS.json, per-arm logs in run/. Native paired results go to run/SUMMARY.json,
with factual guard rates in run/guards.json. Report learned benefits conditional on short continuation,
not a verdict on all budgets or on the eventual choice of loss.
