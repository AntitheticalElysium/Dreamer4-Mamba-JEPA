# P1-F3 on its own training histories vs dev, at 16 sampler steps (user 11-06: "the coal reading")

**Why.** P1-F3 lifted coal / iron only partly (HUD field at sigma 20: coal 0.124 -> 0.155, iron 0.046 -> 0.089; coal
increments unchanged at 0.10; `../20261105_p1f3/RESULTS.md`). The user's reading: the model does not learn the
relationship (it should, from the examples it has), plus less coverage than the common counts. The 11-04 reading
(`../20261104_f2_train_dev`) found F2's coal / iron / death / damage NOT memorised, but at the production 3-step
sampler, which decodes minority outcomes by majority vote (consequence trace RESULTS.md 5, 7): a vote can hide a
training-side advantage. This repeats that reading on F3 at 16 steps, where worlds are now compared.

**Claim (instance learning).** F3 generates the recorded consequences on histories it trained on (~10.5 exposures each)
clearly better than on dev: it learns instances, not the rule. **Competing (learner-limited).** It fails on its training
histories too: even seen instances are not produced, so the limit is the learning signal / route / representation,
not the number of distinct instances. **Separating observation.** Same checkpoint, instrument, gaze and sampler on
training vs dev histories.

**Instrument.** `../20261104_f2_train_dev/train_dev.py` unchanged in substance (`train_dev_steps.py` imports its
`subset` / `compare`; the actor interface `../20261030_actor_interface/interface.py`: classes rng 2031, K = 4, seeds
3000 + i, the fixed random evaluation gaze), on P1-F3 checkpoint 10 with edm.sampler.steps = 16. Training side: the
same 1,009 training episodes (seed 612403, 5% monitor excluded; asserted equal to 11-04's list). Dev side: F3's own
16-step reading (`../20261105_p1f3/readings/interface_steps.json`, f3_16steps). R0: COLLECT_COAL on dev recomputed in
this process must equal that reading exactly. Unpaired episode-bootstrap SEs, combined.

**Declared reading (11-04's rule, unchanged).** Per class, gap closure = (rho_train - rho_dev) / (1 - rho_dev),
rho = generated / recorded event rate; MEMORISED: closure >= 1/3 and the train - dev interval above 0; NOT MEMORISED:
closure's upper bound < 1/3; otherwise UNRESOLVED. Primary: COLLECT_COAL, COLLECT_IRON, risky death, adjacent damage;
every other class reported by the same rule. Reading of the primary coal verdict:
- MEMORISED -> instance learning: F3 can produce coal where it has seen it, does not generalise the rule; more distinct
  instances would help, a learner that shares structure across counts would help more cheaply.
- NOT MEMORISED -> learner-limited even at 16 steps: more data is not predicted to help cheaply; the deficit is in
  how the consequence is learned (signal, route or representation).
One world seed; outcome-head readings on one-step futures; the history glimpses are the random prior (conservative
for memorisation through the belief).
