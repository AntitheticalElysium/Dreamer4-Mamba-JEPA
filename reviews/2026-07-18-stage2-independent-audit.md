# Independent audit: Stage-2 A/B full-world generated supervision

Date: 2026-07-18  
Outcome commit audited: `300b1fc0aeca3efdd863f619f4bea60515d40ab8`  
Runner commit: `2c30a66538cfdc1cd09738388732d4e8fcd50841`  
Verdict: **implementation-correct combined intervention; full acceptance
fails; broad repair claim refuted**

## Copy-ready response to the companion

I independently reproduce the Stage-2 artifacts and accept the negative
decision, but I do not accept the filed mechanism claim.

1. **Provenance and transition correctness pass.** Protocol, dev/final bundle
   pinning, runner, pre-outcome terminal-pool amendment, and result commits are
   chronologically valid. All recorded hashes match. Arm A and B have the same
   initial state and main replay schedule. Synthetic tracing confirms that
   action 7 predicts observation/reward/continuation 7 at K1, action 8 predicts
   index 8 at K2, and the alive mask removes only post-terminal losses. I find
   no future leakage, stale recurrent cache, target-encoder drift, or
   transition off-by-one.
2. **The narrow K8 reward-event result is real.** B-A K8 AUROC is `+.0593`,
   paired episode-cluster CI `[+.0083,+.1106]`, and decoded event magnitude
   rises by `+.0567`. This is a valid one-seed dev result for the complete
   Arm-B package.
3. **"Deep reward repaired" is refuted.** K8 Pearson does not improve; AP moves
   only `.119 -> .128`; overall MAE worsens by `.0171`; zero-target magnitude
   worsens by `.0196`; and K0 AUROC falls by `.0754`, CI
   `[-.1172,-.0304]`. Removing terminal-reward rows makes the K8 AUROC CI cross
   zero.
4. **The experiment does not isolate per-step supervision.** Every B update
   adds a natural generated-state latent+reward+continuation loss. Every tenth
   update adds a second terminal-pool generated loss rather than replacing the
   natural batch. B therefore changes loss routing, data distribution, example
   count, and compute simultaneously.
5. **The terminal curriculum is implicit event oversampling.** The main
   generated positions contain about `3.56%` reward events and zero terminals.
   The terminal pool contains about `53.70%` reward events across K1/K2; K2 is
   exactly `100%` terminal and `100%` reward event with mean reward about
   `-.25`. It cannot diagnose continuation without also changing reward
   supervision.
6. **The dynamics representation deteriorates.** Direct last-predictor cosine
   error to the identical frozen target is worse for B at every evaluated
   depth:

   | depth | A | B | paired B-A |
   |---|---:|---:|---:|
   | K1 | .02269 | .03060 | +.00791 |
   | K2 | .02845 | .03589 | +.00744 |
   | K4 | .03871 | .05114 | +.01243 |
   | K8 | .05598 | .08739 | +.03141 |

   Every paired CI excludes zero. Main JEPA loss also worsens
   `.02181 -> .02976`.
7. **Continuation is not flat after K1.** AUROC is rank-only. Arm-B Brier
   skill reaches `-2.745/-12.030/-10.499` at K2/K4/K8. Nonterminal
   false-terminal rates reach `2.64%/9.77%/8.78%`. This is a severe
   calibration failure directly relevant to continuation-gated planning.
8. **The registered false-reward gate fails decisively.** On truly zero-return
   fork suffixes, raw predicted cumulative reward grows `.0095 -> .1219`;
   B-A is `+.1125`, CI `[+.0631,+.1854]`. Continuation gating reduces but does
   not remove it: delta `+.0669`, CI `[+.0466,+.0941]`. The permitted delta
   was `+.02`.
9. **The implementation is causally implicated by design, not by a conventional
   bug.** On a fixed initialization batch, shared-dynamics gradient norms for
   generated latent/reward/continuation are approximately
   `2.52/23.54/14.52`. Reward and continuation gradients are 6-9x larger and
   mildly oppose the latent gradient. The checkpoint trade-off matches that
   mechanism.
10. **The test suite is insufficient for this stage.** The full existing suite
    passes (`108 passed, 3 deselected`), but no committed test mentions Stage
    2, its terminal pool, generated loss, component routing, latent readout, or
    complete acceptance calculation. Passing legacy tests therefore does not
    validate this experiment.

The corrective route is not "repeat B without continuation while retaining
the terminal pool." Use only uniform main replay, remove generated
continuation and the terminal pool, and separate:

- base + per-step generated latent;
- base + the same latent term + a fixed gradient-balanced reward term.

Continuation remains teacher-forced through the unchanged base objective.
The fixed Arm-A checkpoint can be reused after its hashes, initial digest,
configuration, and schedule are asserted. The new evaluation must include
paired reward/continuation/ranking intervals, zero-suffix false reward, and
direct latent accuracy. Mamba, replication, and the final tier remain blocked
unless the complete candidate passes.

## Source-grounded boundary

- SPR source `mila-iqia__spr@0b9dd4e` unrolls a transition model and predicts
  reward at every jump (`models.py:438-469`), then masks SPR losses after
  terminals (`algos.py:269-303`). It does not provide a continuation head or
  justify this experiment's coupled terminal batch.
- Official V-JEPA 2 source
  `facebookresearch__vjepa2@204698b` scores the entire autoregressive latent
  sequence (`app/vjepa_droid/train.py:425-449`). It does not apply reward or
  continuation losses.
- The next control is consequently **SPR-shaped and locally
  gradient-balanced**, not a reproduction of SPR, V-JEPA 2, or Dreamer 4.

## Ruling

- Arm-B checkpoint: retain as a diagnostic artifact.
- Broad Stage-2 repair claim: withdraw.
- Full Stage-2 acceptance: fail.
- Existing final tier: untouched.
- Planner, online policy, reliability weighting, Mamba replication: no-go.
- Next action: the preregistered Stage-2C uniform-data latent/reward factorial.
