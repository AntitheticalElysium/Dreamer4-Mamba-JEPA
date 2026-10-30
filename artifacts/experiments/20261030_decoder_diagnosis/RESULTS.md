# Why the shared pixel decoder is digit-blind (10-30; `checks.py`, `results.json`, P1 world 1)

1. Information flow (drink 65 at (7, 2) / wood 68 at (7, 5); MLP / linear probes, train-400 -> dev): glimpse token
   0.9995 / 0.9994 and 0.996 / 0.997; reconstruction adapter output (the decoder mean's input) 0.998 / 0.999 and
   0.986 / 0.993; decoder mean pixels 0.87 / 0.96 and 0.73 / 0.85; mean's nearest exact template 0.14 (draws "1" for
   96% of frames) and 0.42. The digit reaches the mean head and dies there; the adapter keeps it (even linearly).
2. Gradient of the reconstruction NLL w.r.t. the decoder mean, per pixel (drink window / wood window): HUD digit pixels
   0.29 / 0.98 at error 2.31 / 1.38 and 1/s 0.72 / 4.5; HUD icon-background pixels 0.50 / 19.7; map pixels 33.9 / 58
   at error 0.024 / 0.016 and 1/s 113 / 201. Digit pixels are ~100x more wrong and pulled 60-117x less.
3. Trajectory, P1 checkpoints 1-10 (first 60 dev episodes): log-scale at initialisation ~0 everywhere (fresh model,
   seed 1); body stats -0.65 at checkpoint 1 and -0.66 at 10 (never narrowed), inventory -2.1 -> -2.8, map -3.6 ->
   -4.6. Decoded drink cell: sapling = 1 (checkpoints 1-2), energy = 1 (3-10), never the drink icon (0% correct at
   every checkpoint); wood: wood = 1 throughout.
Facts beside it: HUD cells are pixel-identical day and night (max |difference| 0 over ~13k night frames per cell);
Craftax-Classic adds uniform random static to MAP pixels when daylight < 0.5, with a position-dependent intensity
texture (renderer.py, "Apply night"); 33% of dev frames have light < 0.5.
Reading: the position x lighting scale (TSP's noise model, designed for real night static) also absorbs model error;
cells not fitted early (body stats: varied values) kept the initial scale, so their mean's pull stayed ~100x weaker,
a state stable from the first 10% of P1 (Seitzer et al. 2022's premature convergence; Lucas et al. 2019). Not
established: which alternative renders digits (matched refit on frozen tokens, not run).
