# Reproducing the additional diagnostics

These three commands reproduce the paper's post-hoc discrimination/reliability,
within-family specificity, and capability-profile overlap diagnostics. They use
the primary spectral audit's frozen 50% anchors, owner splits, model
representatives, and one-percentage-point near-tie definition.

Install the package and pinned dependencies using the root README first. These
modules use NumPy, pandas, SciPy, scikit-learn, threadpoolctl, and the primary data
loader; MMLU-Pro also requires a parquet reader such as PyArrow. Exact paper
reproduction requires the pinned environment. Strict equality is used for ties,
so changes in floating-point libraries can alter a few eligible pairs even when
the numerical scores appear identical after rounding. Each runner checks its
reconstructed cross-family counts against the frozen primary outputs and fails
if they differ; it does not silently relax the tie rule.

## Required primary files

First reconstruct the primary spectral experiment for the benchmark(s) of
interest. `--primary-root` points to the **parent** of these directories:

```text
outputs/
  mmlu_pro_family_ranking_impact/
  bbh_family_ranking_impact/
  mmlu_family_ranking_impact/
  hellaswag_family_ranking_impact/
  winogrande_family_ranking_impact/
```

Every selected benchmark must have:

```text
owner_family_representatives.csv
crossfit_anchor_items.csv
matched_random_anchor_controls.csv
ranking_impact_metrics.csv
```

Discrimination and capability-profile diagnostics also reconstruct the final
spectral representation and require:

```text
family_models.csv
dimension_selection.csv
crossfit_model_coordinates.csv
```

The raw response matrices and metadata loaded by the primary experiment are
also required. Setting `--primary-root` changes the location of the reconstructed
primary tables, not the response-data root. Set `FAMILY_DIF_PROJECT_ROOT` **before
starting Python** when response data live in another prepared project root.
Default paths otherwise resolve relative to this repository. The root data
preparation instructions explain the required response matrices and metadata.
Aggregate result tables shipped in `results/` are reference outputs; they do not
contain all model responses and cannot substitute for data preparation or the
primary reconstruction.

Missing primary files produce an error listing their relative names. A failure
that reconstructed coordinates or pair counts differ indicates that the
selected primary output, seed, input data, or numerical environment is not the
paper configuration. Use the primary spectral outputs, not the direct-logistic
MIRT comparison, as input to these diagnostics.

All commands accept `--benchmarks mmlu_pro,bbh,mmlu,hellaswag,winogrande`; by
default they run all five. A new or empty output directory is required to avoid
overwriting a completed run. Running one benchmark is supported for verification.

## Discrimination and reliability control

```bash
python -m family_dif_benchmark_audit.diagnostics.discrimination \
  --primary-root outputs \
  --output-dir outputs/discrimination_diagnostic
```

For every frozen item-group-by-easiness cell, the audit-half standard deviation
of the family-blind predicted-logit offset is divided into up to four nonempty,
near-equal strata. Each random subtest selects exactly the low-DIF count in each
refined stratum and retains the original cell-restoring weight. The opposite
owner half supplies the ranking outcomes and held-out corrected item-total
correlation, independent-item score SEM, and weighted Cronbach alpha. These
held-out diagnostics are evaluations, not matching inputs.

Both original and discrimination-matched controls use 1,000 replicates. Original
controls use seed `20260825 + fold_index`; refined controls use
`20260917 + fold_index`. The final spectral representation uses
`20260826 + 1000 * fold_index + 103`, preserving the primary reconstruction.
The CLI requires 1,000 replicates and four maximum discrimination strata for
this paper diagnostic.

Main outputs:

- `summary.csv`: pooled reversal results and fold-mean balance diagnostics.
- `low_dif_fold_diagnostics.csv`: observed metrics in each target half.
- `original_matched_random_controls.csv` and
  `discrimination_matched_random_controls.csv`: all control replicates.
- `item_discrimination_diagnostics.csv`: audit matching variables and held-out
  evaluation quantities for every item and fold.
- `report.md`, `appendix_table.tex`, `protocol.json`: readable results and settings.

Matching one predicted-logit spread functional and obtaining close SEM/alpha
does not equate every aspect of item discrimination or capability demand.

## Within-family specificity diagnostic

```bash
python -m family_dif_benchmark_audit.diagnostics.within_family \
  --primary-root outputs \
  --output-dir outputs/within_family_diagnostic
```

This diagnostic reuses frozen anchors without fitting MIRT again. Within- and
cross-family pairs must come from different owners, have a full-score difference
of at most 0.01, and have no exact tie under either score. Every random subtest
is shared by the two pair types. The original controls are reconstructed with
seed `20260825 + fold_index`, and all 1,000 cross-family replicate counts are
checked against the primary output.

The specificity statistic is:

```text
(cross-family reversal rate - within-family reversal rate) for low-DIF
  - median(cross-family rate - within-family rate for each paired random control)
```

The median is taken **after** pairing the two control rates. It is not generally
equal to the difference of their separate medians. Fold counts, rather than
unweighted fold rates, are pooled before calculating reversal rates. One-sided
empirical probabilities use `(1 + count(null >= observed)) / 1001`.

Outputs are `summary.csv`, `observed_fold_metrics.csv`,
`matched_random_controls.csv`, `report.md`, `appendix_table.tex`, `protocol.json`,
and `SHA256SUMS`. A positive within-family excess is compatible with a generic
recomposition sensitivity component; the paired specificity contrast addresses
whether the cross-family component is larger.

## Capability-profile overlap and matched replacements

```bash
python -m family_dif_benchmark_audit.diagnostics.capability_profile \
  --primary-root outputs \
  --output-dir outputs/capability_profile_diagnostic
```

The audit-only item representation uses **all** selected dimensions:
`embedding_i = loading_i / (scale_i * sqrt(number_of_audit_models))`.
Euclidean distance between these embeddings equals RMS difference between their
unclipped spectral predicted-logit profiles over the audit models. This is not
the same as raw loading distance. Matching cost adds squared profile distance
and squared fitted item-intercept difference, within each frozen blueprint cell.

All five scenarios are always run and reported:

| Scenario | Rule |
|---|---|
| `random_pairing` | Random one-to-one low-DIF/nonselected pairing within each cell |
| `optimal_unrestricted` | Minimum total matching cost, without a distance limit |
| `caliper_1.00` | Both profile RMS and absolute intercept difference at most 1 logit |
| `caliper_0.50` | Both distances at most 0.5 logits |
| `caliper_0.25` | Both distances at most 0.25 logits |

Caliper assignment maximizes feasible cardinality before minimizing cost. No
item is reused. Unmatched low-DIF anchors remain fixed. Each of 1,000 controls
chooses one endpoint per matched pair, preserving within-cell selected counts
and total weight. Complementary endpoint configurations provide the paired
contrast reference; a separate all-swapped form replaces every feasible pair.
Base seed is `20260924 + 100 * benchmark_index + fold_index`, with benchmark
order `mmlu_pro,bbh,mmlu,hellaswag,winogrande`; selecting fewer benchmarks does
not change a benchmark's seed. Random pairing adds 10,000 to that seed.

The summary reports every scenario, including null results, swappable weight
coverage, pair distances, cosine similarities, anchor overlap, DIF contrasts,
reliability diagnostics, reversal rates, empirical tails, and the paired
low-DIF/all-swapped contrast. Each benchmark subdirectory also contains the
audit representations, explicit pairs, random switches, observed/control/
all-swapped metrics, matching-quality tables, and reconstruction checks.

Run the independent read-only output check after the run:

```bash
python -m family_dif_benchmark_audit.diagnostics.capability_profile \
  --primary-root outputs \
  --output-dir outputs/capability_profile_diagnostic \
  --verify-only
```

This verifies input, output, and code hashes; all scenario/caliper conditions;
the profile-distance identity; independently enumerated pair counts for four
replicates in every fold/scenario; and pooled empirical tails. The benchmark
selection is read from the saved protocol. Code hashes refer to relative
package paths; data hashes use filenames, without local machine paths.

Unrestricted optimal matching does not guarantee close matches. Tight calipers
can leave almost all low-DIF mass unchanged. Such a low-coverage result is a
limited perturbation, not evidence for capability-equivalent tests or evidence
that an effect has disappeared. All matching-support diagnostics must accompany
the ranking results. These post-hoc diagnostics do not establish causal family
effects or that the low-DIF score is a superior measure of capability.

## Tests

```bash
python -m unittest discover -s tests -p test_release_diagnostics.py -v
```

The tests cover independent strict-pair enumeration, paired specificity tails,
direct leave-one-item-out correlations, weighted reliability, exact blueprint
mass/count preservation, optimal-assignment agreement with exhaustive search,
hard calipers, all scenario coverage, and the profile distance identity.
