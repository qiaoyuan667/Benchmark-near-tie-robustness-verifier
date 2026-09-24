# Mandatory experiment A: score-gap sensitivity

Each point is cumulative over model pairs whose full-score gap is less than or equal to the displayed threshold. Low-DIF and matched-random subtests use the same item count and native source-by-easiness blueprint. Composition-bootstrap intervals resample native source blocks, except HellaSwag and WinoGrande, which use the frozen 100 contiguous item clusters. Pair sets are frozen from the observed full-benchmark score.

| benchmark | gap | eligible pairs | low-DIF flips | composition-resampling 95% percentile range | random median | random 95% interval | excess | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BBH | 0.25 pp | 558 | 49.1% | [16.2%, 45.3%] | 42.3% | [38.0%, 46.1%] | +6.8 pp | 0.0010 |
| BBH | 0.50 pp | 1099 | 46.6% | [16.2%, 44.5%] | 36.0% | [32.7%, 39.3%] | +10.6 pp | 0.0010 |
| BBH | 1.00 pp | 2226 | 42.1% | [16.5%, 42.9%] | 25.2% | [22.4%, 28.6%] | +16.9 pp | 0.0010 |
| BBH | 2.00 pp | 4327 | 33.9% | [16.1%, 39.1%] | 14.1% | [12.3%, 16.5%] | +19.8 pp | 0.0010 |
| BBH | 3.00 pp | 6385 | 26.9% | [15.4%, 34.3%] | 9.6% | [8.3%, 11.3%] | +17.4 pp | 0.0010 |
| BBH | 5.00 pp | 10313 | 18.0% | [13.3%, 26.8%] | 5.9% | [5.2%, 7.0%] | +12.1 pp | 0.0010 |
| HellaSwag | 0.25 pp | 1169 | 43.5% | [29.8%, 48.3%] | 32.8% | [29.2%, 36.0%] | +10.7 pp | 0.0010 |
| HellaSwag | 0.50 pp | 2347 | 39.3% | [30.9%, 43.0%] | 21.1% | [18.4%, 24.2%] | +18.2 pp | 0.0010 |
| HellaSwag | 1.00 pp | 4533 | 30.9% | [27.7%, 34.3%] | 11.4% | [9.8%, 13.3%] | +19.5 pp | 0.0010 |
| HellaSwag | 2.00 pp | 9016 | 19.0% | [17.8%, 22.3%] | 5.7% | [4.9%, 6.7%] | +13.3 pp | 0.0010 |
| HellaSwag | 3.00 pp | 14076 | 12.4% | [11.8%, 14.8%] | 3.7% | [3.2%, 4.3%] | +8.7 pp | 0.0010 |
| HellaSwag | 5.00 pp | 23777 | 7.4% | [7.0%, 8.8%] | 2.2% | [1.9%, 2.5%] | +5.2 pp | 0.0010 |
| MMLU | 0.25 pp | 829 | 48.0% | [27.7%, 56.2%] | 38.0% | [33.8%, 42.6%] | +10.0 pp | 0.0010 |
| MMLU | 0.50 pp | 1588 | 46.3% | [28.3%, 54.1%] | 28.8% | [25.2%, 33.8%] | +17.5 pp | 0.0010 |
| MMLU | 1.00 pp | 3235 | 40.7% | [28.5%, 47.2%] | 16.3% | [13.9%, 20.6%] | +24.4 pp | 0.0010 |
| MMLU | 2.00 pp | 6832 | 31.7% | [24.9%, 34.7%] | 7.8% | [6.6%, 9.9%] | +23.9 pp | 0.0010 |
| MMLU | 3.00 pp | 10531 | 24.1% | [20.2%, 26.9%] | 5.1% | [4.3%, 6.4%] | +19.1 pp | 0.0010 |
| MMLU | 5.00 pp | 17452 | 15.3% | [13.3%, 18.7%] | 3.1% | [2.6%, 3.9%] | +12.3 pp | 0.0010 |
| MMLU-Pro | 0.25 pp | 89 | 43.8% | [24.7%, 56.7%] | 37.1% | [27.0%, 47.2%] | +6.7 pp | 0.1139 |
| MMLU-Pro | 0.50 pp | 177 | 45.8% | [27.7%, 56.7%] | 28.2% | [20.9%, 35.6%] | +17.5 pp | 0.0010 |
| MMLU-Pro | 1.00 pp | 308 | 47.1% | [29.9%, 53.9%] | 18.5% | [13.6%, 24.0%] | +28.6 pp | 0.0010 |
| MMLU-Pro | 2.00 pp | 590 | 40.0% | [28.9%, 42.9%] | 9.8% | [7.1%, 12.7%] | +30.2 pp | 0.0010 |
| MMLU-Pro | 3.00 pp | 824 | 33.1% | [26.2%, 35.3%] | 7.0% | [5.1%, 9.1%] | +26.1 pp | 0.0010 |
| MMLU-Pro | 5.00 pp | 1299 | 23.6% | [20.1%, 26.0%] | 4.5% | [3.2%, 5.8%] | +19.1 pp | 0.0010 |
| WinoGrande | 0.25 pp | 1400 | 46.0% | [22.3%, 39.4%] | 45.1% | [41.9%, 48.4%] | +0.9 pp | 0.3007 |
| WinoGrande | 0.50 pp | 2790 | 40.5% | [22.2%, 38.2%] | 41.5% | [38.6%, 44.9%] | -1.0 pp | 0.7493 |
| WinoGrande | 1.00 pp | 5574 | 33.8% | [21.9%, 36.2%] | 34.7% | [31.6%, 39.3%] | -0.9 pp | 0.6893 |
| WinoGrande | 2.00 pp | 11688 | 22.6% | [19.5%, 30.7%] | 23.1% | [19.9%, 29.1%] | -0.5 pp | 0.6044 |
| WinoGrande | 3.00 pp | 17632 | 16.4% | [16.4%, 25.9%] | 16.3% | [13.7%, 21.4%] | +0.1 pp | 0.4675 |
| WinoGrande | 5.00 pp | 28247 | 10.6% | [11.5%, 18.8%] | 10.3% | [8.6%, 13.6%] | +0.4 pp | 0.3836 |

The primary 1 pp row exactly reproduces the five-family main MIRT-primary analysis. The curve is descriptive across thresholds; the confirmatory threshold remains the frozen 1 pp band.
