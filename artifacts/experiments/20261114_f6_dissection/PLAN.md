# F6 dissection (2026-11-14): why the token forecast taught no pixel clock

F6's standard readings (../20261113_p1f6/readings, checkpoint 10): food / drink / energy drops from the belief are at
their base rates as in F5 (0.038 / 0.046 / 0.034; asleep energy rise 0.180 against F5's 0.184), hunger pairs are drawn
untimed (0.160 due / 0.159 not due), and the clocks the reward prices are kept (recovery 0.590, cooldown drawn 0.912 /
0.048). The primary reading is negative. Frozen readings only, no training; plan and code committed before any result.

Two observations from the queue's own clock stage set the questions (gate3, arm `native` against the fresh change-form
head fitted on the same frozen belief):
- F6's own trained head reproduces 2.7% of recovery's due / not-due token difference at checkpoint 10 and 0.02% at
  checkpoint 2; a fresh head on the same beliefs reproduces 40% and 13%. Cooldown: 6.7% against 64%.
- Its separation gradient on recovery is 5.7e-4 at checkpoint 2 and 2.6e-2 at checkpoint 10 (fresh head: 0.28, 1.2).

## Readings

1. **Dose (`dose.py pairs`, checkpoints 1-10).** Claim: as trained, the token forecast's request for a clock was far
   smaller than the reward forecast's, in the units both were trained in (both are per-frame means inside G4, one
   harmoniser weight). Competing explanation: the request was of comparable size and the belief failed to follow it.
   On the exact-history pairs, per checkpoint: the token forecast's own separation gradient and rate
   (-grad sep . grad loss, belief parameters) for recovery, cooldown, energy and hunger, beside the reward forecast's
   on recovery and cooldown (`../20261111_p1f5/credit.py`'s terms, unchanged). The ratio on recovery and cooldown is
   the dose the token forecast delivered where a clock existed; energy and hunger are read against that scale.
2. **Gradient budget (`dose.py batch`, checkpoints 1, 2, 5, 10).** The same question on ordinary training frames: each
   loss term's gradient norm into the belief parameters on the checkpoint's own next training chunk (stream state,
   carried Mamba state and cache from the checkpoint; 64 new frames, one stream at a time, eight streams), raw and
   divided by its group's sigma, with the cosine of each term to the weighted total. The stream's prior fixations are
   used (the chooser's trace cannot be replayed outside the run); stated as a limit.
   Control, added after F6's table was read (same instrument, `DOSE_RUN=artifacts.experiments.20261111_p1f5.train_p1f5`):
   F5's checkpoints 1, 2 and 10, where the TSP forecast holds the token forecast's place.
3. **What the token forecast's loss pays for (`lossmap.py`, checkpoints 2 and 10, all DEV).** Claim: most of the
   term is error the head cannot remove (scene scroll when the player moves: the next token of a cell is another
   cell's previous token, which a head reading only that cell's previous token and the belief cannot supply), so the
   digit events are a small share of the term and of its gradient. Per frame class (player moved / still awake /
   asleep) and cell group (map / HUD digits / other HUD): the term's share, and the head's lift over copying
   (1 - head / copy). At each HUD digit's cell, on frames where the digit changes and where it does not: head error
   against copy error (what share of each kind of change the head predicts).

Not established by these: whether a larger dose would have formed the clocks (a training question), and the cause of
the head's lag (learning rate, joint non-stationarity or capacity are not separated here).

## Second pass (user, 11-14 afternoon): what decides each loss's share of the belief's gradient

Question (user): why does the reward exert more pressure than the other terms, and what decides the share? No
intervention is selected here. Frozen readings on F6 (checkpoints 2, 10) and F5 (control), `share.py`.

A term's gradient into the belief is, per frame, (how wrong the prediction is, in the term's units) x (how much the
head's prediction moves when the belief moves), summed over frames (where per-frame gradients can add or cancel),
times the term's weight (its reduction, e.g. a mean over 1,296 numbers against one categorical, and its group's
harmoniser multiplier). Each factor is read separately on the checkpoint's own next training chunk (8 streams, 64 new
frames, as `dose.py batch`), per term (G1, reactive TSP, reward, reward forecast, token or TSP forecast):

1. **Per-frame pull on the belief vector:** norm of d(frame's term)/d(predicted belief of that frame): mean, median,
   and the share of the total carried by the top 5% of frames (is the pull spread out or concentrated on a few
   surprising frames?). For the reward terms, the share carried by frames with a non-zero reward.
2. **Addition across frames:** norm of the summed per-frame pulls over the sum of their norms (1 = all frames pull
   the belief vector the same way, 1/sqrt(frames) = random directions).
3. **How much the term depends on the belief at all:** the term's value with each frame's belief replaced by the
   belief of a frame half a chunk away, over its true value.
4. **Weights, by arithmetic:** the reduction of each term and the harmoniser's multiplier, with the multiplier its
   stationary point would give (L = sigma^2 / (1 + sigma)).
Known-answer check: the instrumented terms equal `train.p1_losses`' terms on the same chunk and seed.

`share.py align` (verification of Codex's alignment reading, `../20261114_f6_balance_audit/`, on all DEV): on awake
moved frames, token error on overlapping map cells for copying the same cell, F6's head, and copying the cell the
recorded displacement points to.

`share.py light` (added after the factor table; a data fact, checkpoint 10, all DEV): on awake frames where the player
did not move, the map cells' copy and head error by light level, with and without entities in view. With no entity and
unchanged terrain, the remaining error is what no history predicts (night noise, lighting drift).

## Third pass (user, 11-14 evening): research pass on questions 1 and 2

Question 2 asks which prediction should teach the memory. The claim I made in discussion, untested: the denoiser is
the system's action-conditioned forward model, so its error is cleaner of self-caused change than the token head's.
Competing explanation: the denoiser's loss is spent on the same things (moved frames, night noise), so it is no
cleaner a teacher. `g1map.py`, frozen, F6 checkpoints 10 and 2, DEV, the monitor's protocol (8 context frames at the
context level, the target frame at each of the 8 signal levels, fixed noise per target, prior fixations):

- **Targets:** 3,000 frames drawn uniformly from DEV (t >= 9) plus up to 300 frames of each digit event (health
  down / up, food down, drink down, energy down awake, energy up asleep), kept separate in every reading.
- **What the denoiser's loss is spent on:** x-space error per signal level and cell group (map cells, HUD digit
  cells, other HUD cells, glimpse, lighting), by frame class (moved / still awake / asleep; night), unweighted and
  with the trained ramp weight, beside copying the previous frame's tokens. Reading: on moved frames, is the
  denoiser's periphery error near its still-frame error (it handles transport) or near copying's?
- **Where it uses the memory:** the same errors with the window's beliefs replaced by those of a window at least 32
  frames away in the same episode, per level, class and cell group; at each digit's cell on that digit's event frames.
- **Pull by level (`g1map.py pull`, 300 uniform targets):** norm of d(target frame's loss at one level)/d(the
  window's belief vectors), unweighted and ramp-weighted: at which signal levels does the denoiser ask the memory,
  and what does the ramp (0.1 at pure noise, 1.0 at full signal) do to that request?
Known-answer check: the instrumented ramp-weighted per-frame loss equals `model.shortcut_loss` on the same inputs.

Question 1 (the unit) is a reading pass plus arithmetic on the saved gradient norms (`weights.py`, CPU): the
belief-gradient shares that each published weighting rule would give on F6's measured raw norms. No rule is selected.
