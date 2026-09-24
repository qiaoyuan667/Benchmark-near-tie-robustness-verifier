# Five-benchmark MIRT-primary synthesis

Residual near-tie excess reversal is significant in 4/5 benchmarks and exceeds 5 pp in 4/5. Systematic family score-shift ranges exceed matched random controls in 5/5.

| benchmark | K by audit fold | tau-b | close reversal | random | excess | p | family shift range |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | 64/32 | 0.9236 | 47.1% | 18.5% | +28.6 pp | 0.0010 | 5.34 pp |
| BBH | 128/128 | 0.9153 | 42.1% | 25.2% | +16.9 pp | 0.0010 | 2.54 pp |
| MMLU | 128/128 | 0.9300 | 40.7% | 16.3% | +24.4 pp | 0.0010 | 3.76 pp |
| HellaSwag | 16/64 | 0.9483 | 30.9% | 11.4% | +19.5 pp | 0.0010 | 3.29 pp |
| WinoGrande | 8/8 | 0.8996 | 33.8% | 34.7% | -0.9 pp | 0.6893 | 2.76 pp |

WinoGrande is the boundary case for the primary near-tie statistic: its large raw reversal rate is explained by matched shorter-test noise, while its family score-shift range remains systematic. The other four benchmarks retain significant excess near-tie sensitivity after multidimensional ability adjustment.

The result supports local benchmark-dependent ranking resolution, not a universal family ordering or causal architecture effect.
