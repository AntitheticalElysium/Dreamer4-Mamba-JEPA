# Chooser acquisition, adaptation and input readings

All models are frozen-world diagnostic heads. No policy was used to train a world.
No result here establishes improvement from looking or causal identification.

Policies and budgets were fixed in PLAN.md before these fits.

## E1: fixed checkpoint10, existing-policy remote probability

| Head seed / update | Drink | Stone | Coal | Iron |
|---|---:|---:|---:|---:|
| 611601 / 0 | 38.3% | 25.3% | 20.6% | 32.4% |
| 611601 / 100 | 28.7% | 23.3% | 21.0% | 26.5% |
| 611601 / 500 | 29.9% | 22.3% | 21.0% | 26.6% |
| 611601 / 1000 | 29.2% | 18.6% | 18.3% | 27.1% |
| 611601 / 2000 | 29.5% | 19.0% | 22.3% | 23.8% |
| 611601 / 3000 | 29.7% | 20.2% | 25.0% | 25.8% |
| 611601 / 4000 | 29.2% | 21.6% | 24.5% | 26.5% |
| 611602 / 0 | 38.3% | 25.3% | 20.6% | 32.4% |
| 611602 / 100 | 29.8% | 23.8% | 20.9% | 27.5% |
| 611602 / 500 | 31.9% | 24.6% | 23.4% | 28.0% |
| 611602 / 1000 | 30.3% | 20.5% | 20.3% | 28.0% |
| 611602 / 2000 | 29.4% | 19.5% | 17.6% | 24.7% |
| 611602 / 3000 | 29.6% | 19.3% | 19.8% | 25.2% |
| 611602 / 4000 | 29.4% | 19.1% | 21.0% | 24.8% |

## E2: cumulative-budget-matched endpoint remote probability

| Seed / world / head | Drink | Stone | Coal | Iron |
|---|---:|---:|---:|---:|
| 611601 / 1 / carry | 27.0% | 19.6% | 23.6% | 32.0% |
| 611601 / 5 / carry | 27.3% | 28.5% | 28.6% | 29.0% |
| 611601 / 5 / fresh | 29.3% | 32.6% | 34.8% | 37.5% |
| 611601 / 10 / carry | 27.6% | 34.2% | 32.5% | 35.5% |
| 611601 / 10 / fresh | 49.6% | 34.2% | 30.3% | 41.9% |
| 611602 / 1 / carry | 25.8% | 28.4% | 34.0% | 34.6% |
| 611602 / 5 / carry | 45.7% | 32.2% | 25.8% | 43.7% |
| 611602 / 5 / fresh | 27.0% | 25.4% | 29.7% | 33.1% |
| 611602 / 10 / carry | 48.5% | 35.2% | 24.0% | 41.4% |
| 611602 / 10 / fresh | 31.8% | 26.2% | 31.0% | 35.1% |

## E3: capacity-matched information contract, endpoint remote probability

| Seed / input | Drink | Stone | Coal | Iron |
|---|---:|---:|---:|---:|
| 611601 / coarse | 28.0% | 18.3% | 15.5% | 25.2% |
| 611601 / rich | 29.6% | 17.9% | 18.3% | 25.3% |
| 611602 / coarse | 29.5% | 18.7% | 19.2% | 26.1% |
| 611602 / rich | 30.8% | 18.7% | 19.4% | 25.5% |

These tables only index the full evidence. Per-class C/S rankings, counter-only C,
counter error/copy/target-step energy, all policies, contact/union coverage and
episode-bootstrap intervals are in the matching JSON and NPZ files.
Raw token error is not comparable across changed representations without its
target scale; neither raw error nor a fitted negative proves absent information.
E3 rich history is a diagnostic, not a production collector recommendation.
Discuss the full readings before any learned-fixation world run.
