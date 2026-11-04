# One-frame history / native-belief ablation

All four arms completed12k updates with the same schedule, two head seeds, one frozen F1.5 world. Positive reductions mean lower factual token MSE; policy changes are probability points. Intervals are episode-clustered, conditional on these fits. No world-learning or production-adoption claim.

| Fresh-cohort slice | n | Belief at1 A / B (% MSE reduction) | Belief at8 A / B | 1B versus8C A / B |
|---|---:|---|---|---|
| all | 978 | 2.93 [1.97, 3.87] / -1.18 [-2.20, -0.10] | 0.01 [-1.06, 1.09] / 0.17 [-0.95, 1.26] | 0.03 [-1.20, 1.26] / -0.59 [-1.58, 0.42] |
| random | 128 | 0.66 [-1.55, 2.96] / -0.92 [-3.15, 1.28] | 0.42 [-2.15, 2.57] / 3.07 [-0.31, 7.23] | -1.11 [-4.23, 1.88] / -0.54 [-3.95, 3.70] |
| action_move | 51 | 2.57 [0.54, 4.75] / -0.17 [-2.12, 1.61] | 1.76 [-0.24, 3.86] / 2.18 [-1.90, 7.41] | 2.47 [-0.41, 6.02] / 0.31 [-3.93, 5.33] |
| near_entity | 359 | 0.66 [-0.59, 2.03] / 0.07 [-1.18, 1.37] | 0.67 [-0.92, 2.23] / 0.26 [-1.52, 1.94] | 0.48 [-1.07, 2.12] / 0.16 [-1.43, 1.74] |
| class_drink | 128 | 2.66 [-0.31, 5.06] / -2.03 [-5.11, 1.75] | -3.92 [-7.38, -0.57] / 1.85 [-0.83, 4.83] | -0.36 [-3.51, 3.01] / -2.60 [-6.24, 0.99] |
| class_wood | 128 | 11.50 [7.79, 15.97] / -7.72 [-13.55, -3.49] | -7.19 [-15.13, -1.04] / -4.49 [-8.30, -1.08] | -22.54 [-32.31, -16.09] / -4.90 [-9.22, -1.55] |
| class_stone | 128 | 10.70 [7.72, 14.28] / -4.53 [-7.91, -1.09] | 4.71 [0.65, 8.76] / -0.46 [-3.62, 2.73] | -2.55 [-6.25, 1.33] / -0.59 [-3.72, 2.85] |
| class_coal | 128 | -2.13 [-3.97, -0.60] / 1.45 [-0.67, 3.56] | 2.05 [0.43, 3.66] / 0.90 [-1.00, 2.96] | 6.00 [4.29, 7.79] / 2.41 [0.91, 3.93] |
| class_iron | 82 | -0.48 [-1.93, 1.17] / -0.69 [-2.24, 1.10] | 1.65 [0.10, 3.41] / 0.62 [-1.26, 2.53] | 3.28 [1.52, 5.39] / -0.14 [-1.88, 1.76] |

| Fresh combined gaze: remote counter coverage | 1C A/B | 1B A/B | 8C A/B | 8B A/B |
|---|---|---|---|---|
| drink | 32.9% / 29.9% | 31.7% / 29.3% | 32.4% / 31.5% | 30.4% / 35.8% |
| wood | 31.2% / 32.7% | 34.6% / 33.2% | 32.2% / 34.3% | 32.8% / 32.4% |
| sapling | 18.1% / 14.8% | 15.7% / 14.0% | 14.8% / 15.4% | 16.6% / 15.5% |
| stone | 26.0% / 32.2% | 29.7% / 32.1% | 34.9% / 30.0% | 35.4% / 28.9% |
| coal | 29.4% / 32.6% | 33.4% / 34.7% | 35.7% / 35.6% | 37.7% / 31.3% |
| iron | 34.3% / 35.9% | 34.8% / 35.1% | 37.3% / 36.2% | 41.7% / 35.7% |

Full grounded error/coverage/interaction: COMPARISONS.json. Learning trajectories: TRAJECTORIES.json. Cost: COST_V2.json (original failed strict compaction check preserved in COST.json and detail logs). No adaptive weighting or production world run launched.
