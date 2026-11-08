# F2's generation on its own training histories vs dev (user 11-04: approved before the F2.1 mixture decision)

**Why.** F2.1 is to test whether broader data improves generated consequences. At the fixed 25.6M-frame budget no
mixture of the three stores raises the core consequences' exposure much (first coal / iron: the original store is
densest, 1.00x at best; death 1.34x, damage 1.14x by going all-danger; `../20261104_dataset_eda`), against F2 deficits of
4-80x (generated vs real-successor heads: coal 24 vs 99%, iron 9 vs 98%, risky death 1.3 vs 12.8%, adjacent damage
0.3 vs 22%). So if data helps them, it must be through more DISTINCT instances (the union: ~2.7x unique unlocks at
0.75x exposure), not more repeats. Whether F2 is limited by distinct instances is readable from F2 itself.

**Claim (diversity-limited).** F2 memorised what it saw: on histories it trained on (~10.5 exposures each) it generates
the recorded consequences far better than on dev. **Competing (learner-limited).** It fails on its training histories
too: the deficit is in learning / sampling, not the number of distinct instances. **Separating observation.** The same
checkpoint, instrument and gaze on training vs dev histories. Rules out: train ~ dev rules out memorisation as the dev
deficit's explanation (more distinct instances at fewer repeats is then not predicted to help; exposure or the learner
remain); train >> dev rules out a learner that cannot produce the consequence at all (it can, on seen data).

**Instrument.** The actor interface unchanged (`../20261030_actor_interface/interface.py`: classes with rng 2031, K = 4
futures, class seeds 3000 + i, summaries) on F2 checkpoint 10 under the fixed random evaluation gaze (S1-seeded), as
the archived F2 dev reading (`../20261103_p1f2/readings/interface_random.json`, which is the dev side; its in-process
reproduction is checked by `../20261104_f2_own_gaze`). Training side: 1,009 episodes (dev's count) sampled with seed
612403 from the original corpus's train split, the 5% monitor excluded (`data.load_split`, the sha256 rule), from
`artifacts/eda/s0/s0_train_labels.npz`. Unpaired: different histories; episode-bootstrap standard errors combined.
Caveat (conservative for memorisation): F2 saw these histories under its learned gaze; here, as on dev, the history
glimpses are the random prior. The physical stage never reads glimpses, only the belief does.

**Declared reading.** Per class, the event metric is: risky -> P(termination), adjacent -> P(reward < -0.05), every
other class -> P(reward > 0.5) (the head's event), each beside the recorded rate r of the same event in that
population. Relative recall rho = generated / r per population; gap closure = (rho_train - rho_dev) / (1 - rho_dev), its
interval from the generated difference's 95% interval on the training side. MEMORISED: closure >= 1/3 and the
difference's interval above 0. NOT MEMORISED: closure's upper bound < 1/3. Otherwise UNRESOLVED. Primary: COLLECT_COAL,
COLLECT_IRON, risky death, adjacent damage. Every other class is reported with the same rule; none is dropped. The
real-successor heads' train - dev is reported beside (the heads' own memorisation). One world seed.
