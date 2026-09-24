# Mandatory experiment C: item-DIF signature stability

## Verdict

**PASS: owner-disjoint item signatures are stable enough for content interpretation.**

The primary gate requires at least three of all five benchmarks to pass family-effect correlation, high-DIF overlap, and advantaged-family agreement after source-by-easiness control.

![Item-DIF stability and direct source attribution](summary.svg)

## Item-level stability

| benchmark | median family rho | top-20% overlap | advantage agreement | permutation p values | all gates |
|---|---:|---:|---:|---|---:|
| MMLU-Pro | 0.407 | 31.5% | 56.4% | 0.0020 / 0.0020 / 0.0020 | pass |
| BBH | 0.589 | 36.1% | 72.5% | 0.0020 / 0.0020 / 0.0020 | pass |
| MMLU | 0.501 | 38.5% | 71.9% | 0.0020 / 0.0020 / 0.0020 | pass |
| HellaSwag | 0.308 | 37.4% | 44.3% | 0.0020 / 0.0020 / 0.0020 | pass |
| WinoGrande | 0.465 | 38.5% | 66.0% | 0.0020 / 0.0020 / 0.0020 | pass |

## Discovery-selected source extremes

**CONTENT PASS: source directions replicate in the held-out owner half.**

| benchmark | selected source-family extremes | validation sign replication | exact binomial p |
|---|---:|---:|---:|
| MMLU-Pro | 20 | 100.0% | 0.0000 |
| BBH | 20 | 100.0% | 0.0000 |
| MMLU | 20 | 90.0% | 0.0002 |
| Pooled source-rich main | 60 | 96.7% | 0.0000 |

Effects are centered by each family's all-item mean separately in each owner half. Across all source-family cells, not just the selected extremes, the main-benchmark cross-half source correlations range from 0.232 to 0.953.

## Direct attribution of the score shifts

**SUPPORT: discovery-selected source contributions reproduce in the swapped cross-fit.**

Each source contribution is an exact additive component of the family's anchor-minus-full score shift; the components sum to the reported shift up to floating-point error.

| benchmark | selected source contributions | swapped-fold sign replication | exact binomial p |
|---|---:|---:|---:|
| MMLU-Pro | 20 | 65.0% | 0.1316 |
| BBH | 20 | 85.0% | 0.0013 |
| MMLU | 20 | 65.0% | 0.1316 |
| Pooled source-rich main | 60 | 71.7% | 0.0005 |

The source analysis is mechanism-adjacent evidence: it identifies which benchmark domains reproduce family-relative advantages, but does not by itself show which cognitive process or training-data difference caused those advantages. HellaSwag has one native source label and is therefore excluded from the source-domain confirmation.

WinoGrande is especially informative: its owner-disjoint residual item signatures can be evaluated even though its near-tie excess over matched random subtests is null. Stable item heterogeneity is therefore not sufficient for a large ranking-resolution effect; signed effects must also aggregate coherently under benchmark composition. MMLU's direct source attribution is also partial, so its specific source drivers should not be presented as confirmed even though its item signatures and ranking result replicate.
