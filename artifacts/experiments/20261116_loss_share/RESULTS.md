# Results (2026-11-16)

## Token drift between checkpoints (`drift.py`, `drift.json`; 1,200 fixed DEV frames, CPU)

| Run | Forecast heads into the encoder | Token variance at ckpt 10 | Early drift (ckpts 2-4) | Late drift (ckpts 8-10) | Late / early | Coal / iron 0->1 (evaluations on file) |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| F4 | none (reactive heads; change map as a second owner) | 0.379 | 0.136 | 0.0030 | 0.022 | 0.99 / 0.92 |
| F5 | yes (change map removed) | 0.307 | 0.177 | 0.0091 | 0.051 | 0.57 / 0.32 |
| F6 | yes | 0.284 | 0.170 | 0.0104 | 0.061 | 0.48 / 0.19 |
| F6.1 | yes (45x TSP at ckpt 10) | 0.260 | 0.176 | 0.0067 | 0.038 | 0.35 / 0.14 |

Against the bar fixed before results (F6.1 at least twice F4's late drift, absolute and as a ratio): absolute 2.2x
(met), ratio 1.7x (not met, outside 1.5x): unresolved for F6.1; F5 (3.0x / 2.3x) and F6 (3.5x / 2.8x) meet both. The
runs with forecast heads keep moving their tokens later in training, and every one has less token variance than F4
(0.38 to 0.26, falling with each recipe that added reward pressure). Confounded: F5 also removed the change map, F4's
largest encoder owner (raw gradient 1.11 against TSP 0.54), so grounding's share fell from both sides. Late drift in
G1's units (variance x drift): F4 0.0011, F6.1 0.0017 mean squared token change per checkpoint interval, against G1
losses of 0.0032 and 0.0042: in both, the denoiser's targets move by about a third to two fifths of its loss level per
interval late in training.

## Harmoniser fixed point (F6.1 training log, last 1M frames)

| Group | Mean loss | Weight reached (1/sigma) | Weight at the harmoniser's own fixed point |
| :--- | ---: | ---: | ---: |
| G1 next percept | 0.0039 | 2.68 | 15.5 |
| G2 grounding (TSP + reconstruction) | 0.0021 | 2.73 | 21.3 |
| G3 reactive reward + continuation | 0.0057 | 2.68 | 12.8 |
| G4 forecasts | 0.30 | 1.35 | 1.4 |

The forecast group sits at its fixed point; the other three end 5-8x below theirs (the learned sigma is
rate-limited, as found 10-31). Arithmetic on F6.1's checkpoint-10 weighted gradients (RESULTS 5) at the fixed-point
weights, not a retraining prediction: the reward heads' gradient into the encoder would fall from about 41x to 7.6x
TSP + reconstruction's, and G1's share of Mamba's gradient would rise from about 0.5% to 2%.
