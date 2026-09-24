# V3 family/owner robustness

## Verdict

**PASS: at least three of five benchmarks retain the result under every specified family/owner perturbation.**

The gate asks whether at least three of all five benchmarks retain positive excess close-pair flips under each variant. The stricter significance gate also requires p<=.05 in at least three.

![Robustness matrix](robustness_matrix.svg)

## Gate summary

| variant | positive / 5 | significant / 5 | direction gate | significance gate |
|---|---:|---:|---:|---:|
| owner_cap_1 | 4 | 4 | pass | pass |
| owner_cap_3 | 4 | 4 | pass | pass |
| owner_cap_5_baseline | 4 | 4 | pass | pass |
| lexicographic_checkpoint | 5 | 4 | pass | pass |
| strict_lineage | 5 | 4 | pass | pass |
| omit_gemma | 5 | 4 | pass | pass |
| omit_llama | 4 | 4 | pass | pass |
| omit_mistral | 5 | 5 | pass | pass |
| omit_phi | 5 | 4 | pass | pass |
| omit_qwen | 4 | 4 | pass | pass |

## Benchmark x variant results

| benchmark | variant | close flips | random median | excess | p | families |
|---|---|---:|---:|---:|---:|---:|
| MMLU-Pro | owner_cap_1 | 48.1% | 16.9% | +31.2 pp | 0.0050 | 5 |
| MMLU-Pro | owner_cap_3 | 46.2% | 18.0% | +28.2 pp | 0.0050 | 5 |
| MMLU-Pro | owner_cap_5_baseline | 47.1% | 18.5% | +28.6 pp | 0.0010 | 5 |
| MMLU-Pro | lexicographic_checkpoint | 43.6% | 17.8% | +25.8 pp | 0.0050 | 5 |
| MMLU-Pro | strict_lineage | 45.8% | 17.7% | +28.1 pp | 0.0050 | 5 |
| MMLU-Pro | omit_gemma | 47.1% | 18.2% | +28.8 pp | 0.0050 | 4 |
| MMLU-Pro | omit_llama | 50.8% | 17.4% | +33.3 pp | 0.0050 | 4 |
| MMLU-Pro | omit_mistral | 44.0% | 16.5% | +27.6 pp | 0.0050 | 4 |
| MMLU-Pro | omit_phi | 38.4% | 17.4% | +21.0 pp | 0.0050 | 4 |
| MMLU-Pro | omit_qwen | 43.2% | 18.0% | +25.2 pp | 0.0050 | 4 |
| BBH | owner_cap_1 | 41.1% | 24.7% | +16.4 pp | 0.0050 | 5 |
| BBH | owner_cap_3 | 40.5% | 25.4% | +15.2 pp | 0.0050 | 5 |
| BBH | owner_cap_5_baseline | 42.1% | 25.2% | +16.9 pp | 0.0010 | 5 |
| BBH | lexicographic_checkpoint | 40.4% | 25.3% | +15.1 pp | 0.0050 | 5 |
| BBH | strict_lineage | 44.7% | 25.6% | +19.1 pp | 0.0050 | 5 |
| BBH | omit_gemma | 46.3% | 25.2% | +21.1 pp | 0.0050 | 4 |
| BBH | omit_llama | 41.1% | 25.1% | +16.1 pp | 0.0050 | 4 |
| BBH | omit_mistral | 46.3% | 25.6% | +20.7 pp | 0.0050 | 4 |
| BBH | omit_phi | 40.9% | 25.4% | +15.5 pp | 0.0050 | 4 |
| BBH | omit_qwen | 38.3% | 24.6% | +13.7 pp | 0.0050 | 4 |
| MMLU | owner_cap_1 | 41.9% | 16.5% | +25.4 pp | 0.0050 | 5 |
| MMLU | owner_cap_3 | 38.3% | 16.2% | +22.1 pp | 0.0050 | 5 |
| MMLU | owner_cap_5_baseline | 40.7% | 16.3% | +24.4 pp | 0.0010 | 5 |
| MMLU | lexicographic_checkpoint | 39.2% | 16.0% | +23.3 pp | 0.0050 | 5 |
| MMLU | strict_lineage | 41.0% | 16.7% | +24.3 pp | 0.0050 | 5 |
| MMLU | omit_gemma | 39.8% | 16.2% | +23.5 pp | 0.0050 | 4 |
| MMLU | omit_llama | 49.8% | 15.7% | +34.1 pp | 0.0050 | 4 |
| MMLU | omit_mistral | 42.0% | 16.7% | +25.3 pp | 0.0050 | 4 |
| MMLU | omit_phi | 43.7% | 16.5% | +27.2 pp | 0.0050 | 4 |
| MMLU | omit_qwen | 42.3% | 16.3% | +26.0 pp | 0.0050 | 4 |
| HellaSwag | owner_cap_1 | 30.1% | 11.5% | +18.6 pp | 0.0050 | 5 |
| HellaSwag | owner_cap_3 | 31.4% | 11.4% | +20.0 pp | 0.0050 | 5 |
| HellaSwag | owner_cap_5_baseline | 30.9% | 11.4% | +19.5 pp | 0.0010 | 5 |
| HellaSwag | lexicographic_checkpoint | 30.7% | 11.5% | +19.3 pp | 0.0050 | 5 |
| HellaSwag | strict_lineage | 29.1% | 11.1% | +17.9 pp | 0.0050 | 5 |
| HellaSwag | omit_gemma | 35.1% | 11.7% | +23.3 pp | 0.0050 | 4 |
| HellaSwag | omit_llama | 43.6% | 12.0% | +31.6 pp | 0.0050 | 4 |
| HellaSwag | omit_mistral | 21.7% | 11.7% | +10.0 pp | 0.0050 | 4 |
| HellaSwag | omit_phi | 37.0% | 11.4% | +25.7 pp | 0.0050 | 4 |
| HellaSwag | omit_qwen | 39.3% | 11.6% | +27.7 pp | 0.0050 | 4 |
| WinoGrande | owner_cap_1 | 33.5% | 34.7% | -1.2 pp | 0.7413 | 5 |
| WinoGrande | owner_cap_3 | 33.9% | 34.7% | -0.8 pp | 0.6716 | 5 |
| WinoGrande | owner_cap_5_baseline | 33.8% | 34.7% | -0.9 pp | 0.6893 | 5 |
| WinoGrande | lexicographic_checkpoint | 35.8% | 34.5% | +1.4 pp | 0.2388 | 5 |
| WinoGrande | strict_lineage | 35.1% | 35.0% | +0.1 pp | 0.4776 | 5 |
| WinoGrande | omit_gemma | 36.8% | 34.8% | +2.0 pp | 0.1990 | 4 |
| WinoGrande | omit_llama | 34.0% | 35.5% | -1.5 pp | 0.7562 | 4 |
| WinoGrande | omit_mistral | 42.8% | 36.4% | +6.4 pp | 0.0050 | 4 |
| WinoGrande | omit_phi | 37.5% | 34.0% | +3.4 pp | 0.0945 | 4 |
| WinoGrande | omit_qwen | 33.4% | 34.6% | -1.2 pp | 0.7612 | 4 |

All variants retain the original owner-disjoint cross-fit. The lexicographic policy chooses one checkpoint per owner-family without using benchmark accuracy. The strict-lineage filter is a conservative name-based sensitivity analysis, not a claim that all excluded models are invalid derivatives.

## Interpretation limits

WinoGrande is retained despite its null primary near-tie excess. The gate is deliberately based on replication across at least three of all five benchmarks rather than redefining a positive subset after results.

New robustness variants use 200 random-control replicates (minimum attainable p=1/201); imported baseline rows retain their original 1,000 replicates. These tests rule out the specified family/owner definition artifacts, but they do not establish a causal mechanism for the item composition effect.
