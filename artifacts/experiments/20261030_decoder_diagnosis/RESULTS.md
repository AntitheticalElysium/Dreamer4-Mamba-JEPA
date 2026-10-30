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

## Check 4, TSP, beyond increments (`refit.py`, `refit_results.json`; dev first 300 episodes, 92,191 frames,
random-prior fixations; refit on train-400, 128,964 frames, 6,000 steps x 512 frames, one seed per arm)
- TSP IS AFFECTED, measured: its features carry the digit (probe drink 1.000, wood 0.997, health 1.000; reconstruction
  features 0.993 / 0.983 / 0.999) yet its decoded mean draws the same wrong cells: body-stat digits correct 0.014-0.165
  and icons 0.000 (health, food, drink; energy 0.000-0.005), identical to the reconstruction path; digit-pixel gradient
  0.94 vs map 42.7. The two paths share the mean head and the scale (log-scale identical by construction).
- BEYOND INCREMENTS (trained decoder, day MAE 0-255): body-stat ICONS are never drawn (icon correct 0.000); rare map
  content is badly rendered: crafting table 9.1, furnace 17.0, coal 6.4, iron 19.1, diamond 36.0, plant 13.5, zombies
  23.8, skeletons 26.5, arrows 10.4, cows 7.2, against grass 1.4, tree 1.0, stone 0.8, water 0.9.
- MATCHED REFIT (same init, data, budget, optimizer) on frozen good tokens:
  | body-stat digits correct (health / food / drink / energy) | wood | stone | icons | table / furnace / iron | zombies |
  nll (current)  0.035 / 0.033 / 0.165 / 0.025 | 0.50 | 0.40 | food 0.0, others ~1.0 | 3.0 / 3.5 / 3.3 | 22.6
  beta_nll       0.71 / 0.57 / 0.59 / 0.66     | 0.87 | 0.60 | 1.0 | 4.5 / 3.2 / 2.7 | 19.6
  faithful       0.985 / 0.954 / 0.976 / 0.983 | 0.93 | 0.82 | 1.0 | 5.0 / 4.3 / 3.9 | 10.6
  mse            identical mean to faithful (faithful's mean is trained by squared error); no noise model
  Day map MAE: nll 1.82, beta_nll 1.22, faithful / mse 1.94 (common blocks ~0.5-1 grey level blurrier: grass 1.46 vs
  1.60, stone 1.50 vs 0.68); night 4.1-4.5 in every arm (static). Not fixed by any arm: diamond (~35), plant (~13.5),
  skeletons (~27), arrows (~11), cows (~8).
Reading: the current loss reproduces the digit-blindness from a fresh start on tokens that carry the digit, so the
loss form (learned per-position scale in the NLL) causes it, not the joint training's dynamics; protecting the mean
(faithful: mean by squared error, scale by NLL on the detached mean) renders body-stat digits at 95-98% and keeps a
learned noise model; beta_nll is partial. The trained decoder's rare-block failures (table, furnace, iron, coal) are
recovered by refitting under any loss (a joint-training / budget effect, not the loss form); diamond, plant, mobs are
not recovered by any arm at this budget. Limits: frozen checkpoint-10 tokens (in joint training the decoder's loss also
shapes the encoder), one seed per arm, real tokens only.

## TSP's own path (`refit.py --tsp`, `refit_tsp_results.json`; user 10-31: would TSP prefer an uncertainty-weighted
loss?) A fresh TSP adapter + PixelDecoder trained through TSP only on frozen checkpoint-10 inputs (predicted belief,
periphery, lighting), same seed / data / budget as above, read on TSP's path:
  | body-stat digits | icons | wood / stone digits | table / furnace / iron / coal | zombies / skeletons | grass / stone |
  nll       0.014-0.165 | ~0  | 0.67 / 0.42 | 19.1 / 22.0 / 20.3 / 17.0 | 22.8 / 27.5 | 1.85 / 1.22
  beta_nll  0.78-0.94   | 1.0 | 0.98 / 0.88 | 19.1 / 20.1 / 19.1 / 12.7 | 11.7 / 28.0 | 0.99 / 0.98
  faithful  1.000       | 1.0 | 1.00 / 1.00 |  7.7 /  5.2 /  9.1 /  3.6 |  5.8 / 18.6 | 1.24 / 1.71
Night map NLL 3.03-3.14 in every arm. Reading: TSP under faithful renders every HUD value and the rare content best
(TSP's periphery features carry the digit prominently); beta_nll is only sharper on common blocks (~0.3-0.7 grey
level). One loss serves both paths: no second decoder is needed for the loss's sake. Limits as above (frozen inputs,
one seed per arm, no joint training).
