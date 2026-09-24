# Audit your own benchmark

The `family-dif-audit` command evaluates whether cross-family model pairs with
small full-benchmark score differences change order under low-DIF reweighting
more often than under equally short, composition-matched random subtests.
It runs from your existing response matrix; it does not call model APIs.

This is a diagnostic for one specified composition change. A nonsignificant
result does not certify robustness or establish equivalence. A significant
result does not establish unfairness, causal family effects, or a superior
alternative score. Always report reversal rates, their denominators, and excess
relative to the controls together.

## Install and try the interface

From the repository root, in a Python 3.10+ environment:

```sh
export PYTHONDONTWRITEBYTECODE=1
python -m pip install -e .
python examples/make_synthetic_benchmark.py --output ../audit-demo.npz
family-dif-audit ../audit-demo.npz --quick --output ../audit-demo-output
```

The equivalent module invocation works without the console entry point:

```sh
python -m family_dif_benchmark_audit.maintainer ../audit-demo.npz --quick --output ../audit-demo-module
```

The example has 512 items, 96 models, three explicit families, and 32 owners.
It checks the interface and is not a scientific validation dataset.
`--quick` uses K candidates 1, 2, 4, 8, 16, 32 and 199 random controls. It
retains nested validation and all three purification rounds. Its minimum
resolvable randomization p-value is 0.005; it is exploratory and does not
guarantee a particular runtime. Runtime depends strongly on matrix size and K.

## Supply your data

Preferred format: one `.npz` archive, written with `numpy.savez_compressed`,
containing these arrays (Unicode strings, not pickled object arrays):

| Array | Shape | Meaning |
|---|---|---|
| `responses` | items × models | Complete binary 0/1 correctness matrix |
| `model_id` | models | Unique checkpoint identifiers, column-aligned |
| `family` | models | Explicit observational family labels |
| `owner` | models | Dependency groups whose models must stay in one half |
| `item_group` | items | Native task/domain labels, row-aligned |

Use family labels suitable for your evaluation question and disclose their
construction. Owner should capture the dependency unit relevant to your
collection (the paper used uploaders); do not assign a different owner to every
checkpoint from one owner just to increase the nominal sample size. For a
single-task benchmark, the same `item_group` value may be used for every item.
Do not create groups from low-DIF results. Missing responses, duplicate model
IDs, blank metadata, and nonbinary scores are rejected rather than imputed.

An alternative is `responses.npy`, `models.csv`, and `items.csv`:

```text
# models.csv (one row per response column; omit this explanatory line)
model_id,family,owner
model_0000,family_A,owner_0000
model_0001,family_B,owner_0001

# items.csv (one row per response row; omit this explanatory line)
item_group
task_A
task_B
```

CSV row order must match the matrix exactly; the loader does not sort or join
the metadata. String labels are treated literally and case-sensitively.

```sh
family-dif-audit ../responses.npy --models ../models.csv --items ../items.csv \
  --output ../benchmark-audit
```

## Primary audit and settings

```sh
family-dif-audit ../benchmark.npz --output ../benchmark-audit
```

Defaults follow the paper's primary spectral diagnostic: owner-family cap 5,
minimum 20 capped models per family, two owner-disjoint halves, the full K grid
1–256 in powers of two, inner 25% owner validation, within-group 50% calibration
items, coordinate ridge 1, DIF penalty 1, three purification rounds, four
easiness strata per item group, 50% anchors, 1 percentage point near-tie band,
and 1,000 matched-random controls. Matrix-infeasible K candidates are omitted.
The selected dimension is not silently clipped if purification cannot support
it. Such a failure is reported; declare a smaller grid or supply more data.

Full-score means help balance the outer split and choose the median-score
checkpoint within each owner-family. Once that population split is frozen,
only the audit half selects K, fits residual DIF, and constructs the weights
used on the opposite target half. Both directions are evaluated. The same
owner never crosses halves, and pairs sharing an owner are excluded.

Counts are checked in both halves. Each must contain every family, at least
two models per family, and enough models for the inner split (at least eight
inner-fitting and four inner-validation models). These are computational
feasibility checks, not a power guarantee; many independent owners and enough
eligible near-tie comparisons are still needed. Twenty capped models per
family is the paper's setting, not a universally validated minimum.

Useful explicit overrides:

```sh
family-dif-audit ../benchmark.npz --output ../audit-custom \
  --gap-pp 1 --resamples 1000 --dimensions 1,2,4,8,16,32,64,128,256 \
  --owner-cap 5 --seed 31415
```

`--gap-pp 1` means one percentage point, not a score difference of 1.0.
`--owner-cap 0` disables capping. `--seed` sets the population seed and derives
the inner/SVD/control seeds; without it the separate paper seed defaults are
used. All settings and software versions are recorded. Treat overrides and
multiple alternative analyses as declared sensitivity analyses, not as a
search for a favorable p-value. Exact paper reproduction also requires the
paper data, population, and recorded benchmark-specific configuration.

The maintainer command covers the primary diagnostic. It does not run the
paper's direct logistic MIRT, bootstrap, population perturbations, content
audit, or added matching diagnostics; those have separate reproduction entry
points described in the main README.

## Read the outputs

`report.md` is the human-readable result and `summary.json` is the API result.
Inspect both actual selected K values, warnings, pair counts, low-DIF reversal
rate, random median, excess in percentage points, and one-sided randomization
p-value. The p-value uses `(1 + # controls at least as large as observed) / (R + 1)`.

Only within-half, different-owner, different-family pairs with a nonzero full
gap no larger than the requested threshold can contribute. Exact ties under
either scoring rule are excluded separately for that rule. Numerators and
denominators are pooled across folds before rates are formed; fold percentages
are not averaged. The original full-score near-tie mask is reused for every
control. No between-half pairs are compared because their scoring weights
come from different audit halves.

If there are no eligible observed pairs, the status is
`insufficient_eligible_pairs`; the rate, excess, and p-value are null. If any
pooled random replicate has zero eligible pairs, inference is withheld with
`undefined_random_controls`. Neither status means the benchmark passed.
`no_significant_excess_detected` means no evidence at the nominal 0.05 level
for this specific diagnostic, not proof that all near ties are reliable.

Detailed files are also emitted:

- `population.csv`: included model indices, pseudonymized owners/families, and halves.
- `scored_models.csv`: representative indices and the two scores.
- `dimension_cv.csv`: inner-validation losses, one-SE thresholds, and selected K.
- `anchors.csv`: item indices, cell/group codes, selected flags, and frozen weights.
- `purification.csv`: selections and stability across rounds.
- `observed_by_fold.csv`: observed reversal counts and denominators.
- `random_controls_by_fold.csv` and `random_controls_pooled.csv`: every control.

The DIF ranges in `anchors.csv` are from the final refit for diagnostics;
selection and cell labels remain those frozen in the final purification round.
Do not reconstruct the selected set by re-sorting final-refit ranges.

Output files use indices or aliases rather than original model/owner/family
strings. Provenance stores input basenames and SHA-256 digests, never absolute
paths. Basenames are still your responsibility: use neutral filenames if you
intend to publish the report. The input metadata are not anonymized on disk.
The command refuses to overwrite nonempty output directories.

## Python API

```python
from family_dif_benchmark_audit.maintainer import (
    AuditConfig, load_benchmark, run_audit, write_audit,
)

data = load_benchmark("../benchmark.npz")
result = run_audit(data, AuditConfig())
print(result["summary"])
write_audit(result, "../benchmark-audit-api")
```

For in-memory arrays use `BenchmarkData(responses, model_id, family, owner,
item_group)`. DataFrames returned by `run_audit` can be inspected before
writing. A maintainer report should state the population and labels, audit
configuration, eligible pair counts, observed and random reversal rates,
excess, p-value, and any failure or coverage warning. Avoid reducing this
conditional diagnostic to an unconditional pass/fail badge.
