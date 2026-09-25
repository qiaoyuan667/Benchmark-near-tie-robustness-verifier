# Released results

The primary 50% residual low-DIF audit is summarized below. Excess is the
low-DIF strict reversal rate minus the median of 1,000 item-group-by-easiness
matched random subtests, in percentage points.

| Benchmark | $K$ | Kendall $\tau_b$ | Close pairs | Low-DIF | Random | Excess | $p$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | 64/32 | .924 | 308 | 47.1% | 18.5% | +28.6 | .001 |
| BBH | 128/128 | .915 | 2,226 | 42.1% | 25.2% | +16.9 | .001 |
| MMLU | 128/128 | .930 | 3,235 | 40.7% | 16.3% | +24.4 | .001 |
| HellaSwag | 16/64 | .948 | 4,533 | 30.9% | 11.4% | +19.5 | .001 |
| WinoGrande | 8/8 | .900 | 5,574 | 33.8% | 34.7% | -0.9 | .689 |

Additional released checks:

- all four primary-positive benchmarks remain positive and nominally
  significant under all ten population specifications;
- WinoGrande is nominally significant only when Mistral/Mixtral is omitted;
- residual item signatures replicate across disjoint owner halves, with median
  within-cell correlations .308--.589 and all reported cell-permutation
  tests at the Monte Carlo resolution $1/501$;
- item-group-effect signs agree across owner halves in 58/60 discovery-selected
  checks; signs of item-group score-shift contributions agree in 43/60 checks;
- median binary inter-annotator Cohen's kappa is .715 and median ordinal
  Spearman correlation is .800; no binary content axis meets the pre-specified
  joint effect-size and statistical-significance criteria for both annotators,
  and the maximum absolute paired difference is 9.6 percentage points. The
  corresponding BH-adjusted value is $q=.106$, so it does not pass the
  dual-annotator confirmatory gate.

The underlying aggregate tables are indexed in `docs/PAPER_ARTIFACT_MAP.md`.

## Additional diagnostics

These later diagnostic analyses are not relabeled as pre-specified primary
tests. The frozen primary results above remain the primary estimand.

| Benchmark | Direct-MIRT excess (pp) | Discrimination-matched excess (pp) | Paired cross-minus-within specificity (pp) |
| --- | ---: | ---: | ---: |
| MMLU-Pro | +23.1 | +24.0 | +18.0 |
| BBH | +7.5 | +14.2 | +6.7 |
| MMLU | +9.2 | +21.3 | +13.9 |
| HellaSwag | +12.0 | +18.0 | +9.3 |
| WinoGrande | +2.7 | -0.8 | +0.0 |

In each column the first four comparisons have $p=1/1001$; WinoGrande has
$p=.087$, $.683$, and $.496$, respectively. With reversal rates expressed as
proportions, the specificity column is:

```text
100 * [(low-DIF cross-family rate - low-DIF within-family rate)
       - median(random cross-family rate - random within-family rate)]
```

Each random difference uses the same subtest for both pair types. The median
is taken after pairing; it generally differs from subtracting the two
separately calculated excesses. Rates pool counts across the two target halves.

The capability-profile experiment is an overlap diagnostic, not successful
identification of a causal family mechanism. `results/capability_profile/`
includes all five scenarios, random pairing, unrestricted optimal matching,
and increasingly strict calipers. Under strict matching some benchmarks have
very little replaceable item mass. Do not cite a small p-value alone while
omitting coverage, residual profile imbalance, or null outcomes.

Direct MIRT changes the ability estimator and its optimization, not the primary
population, holdouts, blueprint, pair mask, anchor fractions, or resampling
counts. It is penalized joint logistic estimation, not marginal maximum
likelihood or Bayesian MIRT. Optimizer objective plateaus do not prove a global
optimum or complete removal of latent capability differences.
