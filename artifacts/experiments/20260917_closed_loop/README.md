# Does the richer-state gain survive repeated prediction?

Status: **complete**, 2026-09-17. **No.** Frozen encoder, the worlds trained by
[`20260917_state_transition`](../20260917_state_transition/), rolled closed-loop — every
step consumes the previous *prediction*, never a fresh image. Runner:
[`closed_loop.py`](closed_loop.py), evidence in
[`evidence/closed_loop.json`](evidence/closed_loop.json).

512 held-out DEV windows, 2,048 TRAIN, 4 context frames then 8 closed-loop steps.

## The one-step `u→u` advantage does not survive

Fidelity is normalized MSE against the truly-encoded latent, divided by that latent's own
variance, so `z` space and `u` space sit on one scale. **Skill** is that divided by the same
measure for persistence — holding the last observed latent still. Below 1, the world beats
doing nothing; above 1, it loses to it.

| skill (model ÷ persistence) | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 |
|---|---|---|---|---|---|---|---|---|
| `z→z` | 0.097 | 0.089 | 0.094 | 0.114 | 0.133 | 0.144 | 0.164 | **0.179** |
| `u→u` | 0.687 | 0.910 | **1.026** | 1.135 | 1.180 | 1.184 | 1.259 | **1.190** |

**`z→z` is five to ten times better than holding still, at every depth.** `u→u` is barely
better at one step and **loses to holding still from depth 3 onward**.

## Why: `u` barely moves

Persistence MSE at depth 1 — how far each latent actually travels in a single step:

| space | one-step persistence MSE |
|---|---|
| `z` | 0.929 |
| `u` | **0.089** |

`u` moves about **ten times less per step** than `z`. The label-free patch PCA is dominated
by slowly-varying scene content, so "predict `u`" is a much easier task than "predict `z`" —
and a large part of the one-step `u→u` advantage was that easiness, not better dynamics. The
2×2 did not control for it, and this is the control.

## Decoding agrees

A probe fitted on TRUE latents at each depth, read on the PREDICTED latent, mean AUC over
the three supported labels (`positive_reward`, `negative_reward`, `achievement_event`):

| depth | `z→z` predicted | `u→u` predicted | action-only |
|---|---|---|---|
| 1 | 0.8145 | 0.7841 | 0.7767 |
| 2 | 0.7689 | 0.6689 | 0.7793 |
| 4 | 0.7382 | 0.5605 | 0.7536 |
| 6 | 0.6119 | 0.5237 | 0.5849 |
| 8 | 0.7136 | 0.5574 | 0.7234 |

**`z→z` beats `u→u` at every depth**, and `u→u` falls to near chance (0.52–0.56) by depths
4–6. The ordering is the reverse of the one-step 2×2 result.

Note also that **neither world clearly beats action-only** on these labels. Reward and
achievement are strongly action-driven, so action-only is a hard baseline here; that is a
property of the label family, not a verdict on the worlds.

## What this does not establish

The labels here are corpus outcome labels — reward, achievement, termination — **not** the
rich M03 successor-state labels the 2×2 used. So the decoding comparison is not
apples-to-apples with that experiment, and the fidelity result is the load-bearing one: it
is scale-free, needs no labels, and is unambiguous.

One seed, 4,000-step worlds, consecutive checkpoint, frozen encoder. A `u` trained *jointly*
rather than fitted as a fixed PCA might have different temporal structure, and nothing here
tests that. Nor does this rehabilitate `z`: `z→z`'s good fidelity coexists with decoding that
does not clearly clear action-only.

## What it changes

**It argues against a joint retrain on this `u`.** The one-step gain was real but is
substantially explained by `u` being a slow-moving target, and it inverts under exactly the
repeated prediction that imagination requires. Before any retrain, a candidate richer state
should be shown to beat its own persistence baseline under closed loop — a check this
experiment now makes cheap, and one the one-step probes cannot substitute for.
