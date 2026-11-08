# Cached policy ablation

Two existing heads; frozen checkpoint10. No new training.
Values are exact remote-counter probability mass, not argmax coverage.

## saved

| Policy | Drink | Stone | Coal | Iron |
|---|---:|---:|---:|---:|
| uniform | 12.2%/12.2% | 12.2%/12.2% | 8.2%/8.2% | 4.1%/4.1% |
| existing | 30.4%/29.4% | 19.0%/16.8% | 25.5%/20.8% | 25.0%/23.4% |
| C_softmax | 9.1%/7.9% | 6.5%/4.2% | 3.2%/2.4% | 2.4%/1.6% |
| S_softmax | 44.8%/43.9% | 36.7%/38.0% | 47.7%/46.1% | 44.0%/43.3% |
| probability_mixture | 27.0%/25.9% | 21.6%/21.1% | 25.5%/24.3% | 23.2%/22.4% |
| C_direct | 9.7%/8.5% | 8.3%/4.5% | 4.2%/2.8% | 2.8%/1.7% |
| S_direct | 40.3%/40.0% | 32.2%/33.2% | 34.3%/32.5% | 21.2%/21.3% |
| direct_mixture | 25.0%/24.3% | 20.3%/18.8% | 19.3%/17.6% | 12.0%/11.5% |

Native reference under each matching rule and all contact/union, class, light and
paired episode-bootstrap intervals are in policy_results.json.

## confirmation

| Policy | Drink | Stone | Coal | Iron |
|---|---:|---:|---:|---:|
| uniform | 12.2%/12.2% | 12.2%/12.2% | 8.2%/8.2% | 4.1%/4.1% |
| existing | 29.2%/29.4% | 21.6%/19.1% | 24.5%/21.0% | 26.5%/24.8% |
| C_softmax | 8.3%/7.8% | 7.0%/3.9% | 2.8%/2.3% | 2.3%/1.6% |
| S_softmax | 44.1%/43.4% | 41.5%/43.5% | 49.1%/48.1% | 46.7%/44.8% |
| probability_mixture | 26.2%/25.6% | 24.3%/23.7% | 25.9%/25.2% | 24.5%/23.2% |
| C_direct | 9.1%/8.5% | 8.8%/4.3% | 3.7%/2.6% | 2.8%/1.7% |
| S_direct | 40.4%/40.1% | 35.1%/36.6% | 35.1%/33.8% | 22.1%/22.3% |
| direct_mixture | 24.7%/24.3% | 21.9%/20.4% | 19.4%/18.2% | 12.5%/12.0% |

Native reference under each matching rule and all contact/union, class, light and
paired episode-bootstrap intervals are in policy_results.json.

This is conditional allocation on the SAME recorded histories. It does not measure
the policies' own fixation histories, world learning, or adaptive emergence.
No rule or temperature was selected from these results.

CPU elapsed: 2.6s.
