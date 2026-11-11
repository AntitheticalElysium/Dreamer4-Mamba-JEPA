# World-belief role: discussion follow-up, 2026-11-08

User clarification: retain the denoiser's recent cache. The immediate question is whether Mamba supplies
history-dependent information that improves world prediction, not whether its exported vector alone can replace
vision for a future actor. Actor inputs are a separate P3 question. This note adds no training run or model change.

## What the cache does and does not establish

The native cache is not an independent eight-frame visual baseline. Denoiser.embed appends Mamba memory tokens;
dense temporal attention and commit retain keys/values for all physical tokens, including memory, and visual
features can also integrate memory before being cached. Thus earlier current-memory/cache interventions cannot
measure the total contribution of Mamba against a memory-free recent observation window.

Current HUD values and many immediate interaction determinants are already visible. Zombie readiness has a strong
five-step visible-history heuristic. Conversely, timer reset cues often precede the nominal cache window: the
training census gives median previous-visible-reset gaps of 26 for food, 21 for drink, 31 for awake energy and 11
for sleeping energy. Those gaps establish absence of a direct recent reset cue, not impossibility of predicting
phase from recent proxies, actions or other observations. Achievements already earned and histories of occluded
objects are further candidates for useful recurrent information.

We do not have a clean numeric estimate of Mamba's conditional predictive benefit beyond all other inputs. That
requires comparing recent available evidence with the same evidence plus recurrent information, using an older
history reference to establish that the target actually needs history. Recent fine evidence must be controlled:
the physical denoiser cannot directly read glimpse tokens, whereas Mamba receives them. Past memory already in the
native cache must also be accounted for. This is a criterion for a diagnostic, not a selected or launched protocol.

Higher-order information need not mean learned rules reside in recurrent state. Dynamics parameters can encode a
general rule; recurrence should retain this episode's relevant phase, past events and uncertainty about hidden
state. Its success criterion is the resulting conditional forecast, not a human-named register for every fact.

## What the present losses ask of memory

G1 trains next-percept prediction. In principle an informative old fact should be retained when other inputs
cannot supply it and fitting the conditional prediction rewards it sufficiently. In finite training, this is a
joint optimisation problem: the generator must learn to use a memory signal while the recurrent model learns to
provide it. A weak route can reinforce itself. Training also provides a corrupted future target; high-signal
examples can be denoised without inferring the event from history. F4 includes pure-noise cases, so this is a
possible pressure imbalance, not an absent forecasting objective.

The completed 150-microbatch reading demonstrates active G1 teaching in flow and bootstrap branches. On ordinary
frames, mean belief-gradient norms in units of 1e-6 are G1 4.61/5.49, reward 45.21/46.25 and change 18.84/17.80
(flow/bootstrap). These are checkpoint sensitivities, not percentages of learning, parameter-update norms or
causal attribution of training history. Relevant G1 sensitivities rise on event frames and reach older inputs;
neither measurement certifies learning elapsed time.

The harmoniser's G1/G2/G3 weights at this checkpoint are 2.625/2.716/2.582: it is not giving the consequence group
an order-of-magnitude larger coefficient. Group loss balancing does not balance derivatives at the shared belief.
G1 averages 1,880 percept scalars per frame, while reward has one categorical target. Learned head/denoiser
Jacobians and alternative conditioning routes also differ. These are structural differences, not a measured
decomposition of why the gradient ratio takes its observed value. Continuation is negligible in the earlier
full-term gradient reading; saying all outcome heads dominate is inaccurate.

Reward legitimately needs historical context: previous health for reward deltas, and whether achievements were
already earned. Its large signal need not teach anticipation of hidden timer phase. ChangeMap sees both real
frames, so it can explain an observed change without predicting it. Its raw-observation-derived target also
grounds encoder content independently of the learned G1 target. That grounding role does not demonstrate useful
memory training, and overlap with G1 does not justify deleting grounding without a replacement.

## Confirmed region-specific change target

The current implementation really excludes the HUD and fixed player cell:

- `saccade/m0.yaml`: change_map.rows = map_only.
- `saccade/data.py:pair_features`: crops the first 49 image rows, estimates a global camera shift on the 7x9 map,
  and excludes the player cell and its aligned source. `change_targets` writes [frames,7,9].
- `saccade/model.py:ChangeMap`: emits 7x9 predictions and reads only those target positions.
- `saccade/train.py:p1_losses`: loads 7x9 teacher targets.

This came from the A9b camera-aligned map-content-change instrument. It is a documented implementation exception,
but conflicts with the user's region-general design principle. The next target design must remove semantic HUD
and player masks. Historical F4 remains unchanged; this note does not claim a correction has been implemented.

Changing 7 to 9 alone is invalid: world terrain scrolls while the HUD is screen-fixed, and the teacher was trained
and calibrated on map cells. A replacement must handle correspondence across the full observation generically.
Simply accepting the smaller of a same-position and camera-shift residual is not established either: independent
object motion can be explained away by one of those alignments. Geometric validity masks for unavailable source
pixels are distinct from excluding a known semantic region. No replacement teacher or new objective is chosen.

## Actor question, explicitly deferred

Peers do not require a bare recurrent export to serve as the complete actor input:

- DreamerV3 uses recurrent h plus observation-conditioned stochastic z as model state:
  https://arxiv.org/html/2301.04104v2 .
- DIAMOND's actor/critic encodes the current frame through a CNN and its own LSTM:
  https://arxiv.org/html/2405.12399v1 , Appendix D.
- JEDI uses the current learned spatial latent with actor/critic recurrence:
  https://arxiv.org/html/2605.13013v1 , algorithm and architecture table.

The user's preference for acquired vision plus belief is a reasonable P3 candidate. For Saccade, acquired vision
means the actual periphery and selected glimpse with coordinates, not unrestricted full-resolution access that
bypasses the fovea budget. Actor and comparison policies would need matched evidence. No actor configuration or
P3 implementation is changed by this discussion.
