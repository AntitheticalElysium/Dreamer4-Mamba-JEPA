# P1-F6.1 dissection: results (2026-11-15)

## Check 1: the sleep phase where no loss asks for it (`sleep_phase.py`, `sleep_phase.json`)

F6.1 and F6 at checkpoint 10, all DEV encoded on CPU (F6's natural APs reproduce its GPU reference: food 0.0372,
asleep rise 0.1703 against 0.1696), the gaze test's probes, folds 0-3 fit, fold 4 read; "beyond time and light" is the
held-out log-loss gain of a probe with Codex's time-and-light reader as a fixed offset over that reader alone, nats
per eligible frame; F6.1 minus F6 with episode-bootstrap 95% intervals.

| Asleep energy rise | Frames / rises | Time-light AP | AP F6 / F6.1 | Beyond time and light F6 / F6.1 | F6.1 - F6 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| All asleep frames | 10,153 / 842 | 0.097 | 0.170 / 0.401 | 0.021 / 0.060 | 0.040 [0.030, 0.052] |
| WAKE_UP not yet paid | 2,664 / 227 | 0.106 | 0.306 / 0.823 | 0.050 / 0.169 | 0.119 [0.100, 0.137] |
| WAKE_UP already paid | 7,489 / 615 | 0.088 | 0.118 / 0.228 | 0.012 / 0.027 | 0.015 [0.010, 0.022] |
| Paid, 8 constant asleep frames (the gaze test's `after_reset`) | 2,510 / 609 | 0.254 | 0.254 / 0.338 | -0.001 / 0.011 | 0.012 [0.003, 0.025] |

- **The claim mostly holds, with a small general part.** The belief carries the phase strongly where the reward
  forecast prices it (first sleeps: AP 0.82) and weakly where nothing asks (repeat sleeps: 0.23, against F6's 0.12);
  on the strict repeat-sleep cohort it adds 0.011 nats beyond time and light, where F6 adds nothing. Competing
  explanation (i) is therefore partly right: some of what was learned on first sleeps transfers. Explanation (ii)
  (elapsed time, lighting) is ruled out for both cohorts: the time-and-light reader alone reaches AP 0.09-0.11 on
  asleep frames.
- **Awake pixel clocks: nothing, in either world.** Food, drink and awake energy drops add at most 0.0009 nats beyond
  time and light (natural, steady8, after_reset), F6.1 minus F6 within +-0.0005 everywhere. Recovery, the
  reward-priced reference: 0.070 against 0.061 (F6.1 minus F6 0.009 [0.006, 0.011]).
