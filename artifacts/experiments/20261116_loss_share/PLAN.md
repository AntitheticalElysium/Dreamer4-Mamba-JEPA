# Does reward pressure keep the encoder moving late in training? (2026-11-16, user: the loss question)

User 11-16: some events kept falling as the reward heads' share grew; will that hinder what G1 can learn from the
spatial history, and the encoder? Facts on file: the reward heads' gradient into the encoder rose to 45x TSP's by
F6.1's checkpoint 10 (F6.1 RESULTS 5); across F4 / F5 / F6 / F6.1 effective rank 5.96 / 5.22 / 5.20 / 4.88 and coal
0->1 0.99 / 0.57 / 0.48 / 0.35 (correlational). New today: on F6.1's FROZEN encoder, G1-only denoiser fine-tunes
draw coal 0.67-0.78 and iron 0.46-0.70 (F6.1 itself 0.20 / 0.12, same instrument), so that content was not erased
from the tokens; the jointly trained denoiser under-drew it.

Claim measured: under reward pressure the tokens keep drifting late in training, so the jointly trained denoiser
learns rare events on a moving input. Competing explanation: every run's tokens settle late alike (the learning rate
decays to 0.1x), so drift does not separate the recipes. Separating comparison: F4 (no forecast heads; reactive
reward and continuation only) against F5, F6, F6.1 (forecast heads), same frames.

Measurement (`drift.py`, CPU, perception only): 40 DEV episodes, 30 frames each at seeded positions (1,200 frames);
each run's coarse periphery tokens at checkpoints 1-10; drift(k) = mean squared token change from checkpoint k-1 to k,
divided by the run's own token variance at checkpoint 10. Reading fixed before results: late drift = mean drift over
checkpoints 8-10, as a ratio of early drift (checkpoints 2-4), per run. The claim is supported if F6.1's late drift
(absolute and as a ratio) is at least twice F4's; not supported if within 1.5x. Nothing is trained; one seed of each.
