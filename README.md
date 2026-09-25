# Family-DIF benchmark ranking audit

Anonymous research artifact for **Are Near-Tied LLM Rankings Robust to
Family-DIF-Guided Benchmark Recomposition?**

This repository implements an owner-disjoint audit of near-tied model rankings.
One owner half selects items with low residual family differential item
functioning (DIF), conditional on a family-label-free spectral approximation to
multidimensional item-response theory (MIRT). Frozen weights then score models
in the other half. Matched-random subtests preserve the same item-group and
coarse-easiness composition and provide a baseline for item-subsampling effects.

The artifact includes the five primary benchmark implementations and all
reported supplementary analyses: dimension selection, anchor-fraction and
score-gap sensitivity, composition and owner resampling, population
perturbations, item-signature replication and item-group attribution, direct
logistic MIRT, discrimination matching with reliability checks, within-family specificity,
capability-profile overlap, and blinded content-audit statistics. It also
provides a separate interface for maintainers auditing their own benchmark.

This diagnostic concerns a specified change in item composition. Residual DIF
does not establish unfairness or a causal family mechanism, and the low-DIF
score is not proposed as a more correct replacement score.

## Choose a starting point

| Your goal | Start here | Need original benchmark data? |
|---|---|---|
| Check released tables, code tests, figures, and content statistics | [Offline verification](#install-and-verify-the-artifact) | No |
| Recompute the paper from item-level responses | [Reproduction guide](docs/REPRODUCE.md) | Yes; verified preparation is provided |
| Audit a benchmark you maintain | [Maintainer quickstart](#audit-your-own-benchmark) | Your own response matrix and metadata |
| Locate a paper claim or supplementary experiment | [Paper-to-artifact map](docs/PAPER_ARTIFACT_MAP.md) | Depends on the analysis |

## Install and verify the artifact

Run the following commands from the extracted repository directory. The pinned
spectral environment was checked with Python 3.12.14; use Python 3.12 for this
workflow. The package declares Python 3.10+ support, but that declaration does
not mean every pinned package has a wheel for every Python release.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python -m pip install -r requirements-pinned.txt
python -m pip install -e . --no-deps
python -m pytest -q -p no:cacheprovider
python tools/render_paper_figures.py --output-dir ../audit-data/outputs/paper_figures
python tools/audit_release.py
```

On Windows Command Prompt, activate with `.venv\Scripts\activate` and use
`set PYTHONDONTWRITEBYTECODE=1` instead of `export`. `make verify` runs the tests
and read-only release audit on systems with Make; figure rendering is separate.
See [figure provenance](docs/FIGURES.md) for the illustrative overview and the
data-derived research figures.

These commands check the packaged code and released numerical claims. They do
not rerun the full five-benchmark estimation. The number of tests may change as
coverage improves; success means all executed tests pass. PyTorch tests are
skipped when PyTorch is absent and require the separate direct-MIRT environment.
Numerical figure
checks are recorded in `../audit-data/outputs/paper_figures/verification.json`.
The release audit prints its verification result and compares the public files
with the shipped checksums under `provenance/`; it does not rewrite them.

The pinned requirements record direct dependency versions, not a complete
transitive lockfile. Exact-tie pair counting is sensitive to floating-point
software changes; retain the pinned numerical environment when reproducing
the paper. The direct logistic-MIRT comparison uses its own environment,
described in [the reproduction guide](docs/REPRODUCE.md#direct-logistic-mirt-comparison).

## Recompute the paper

Use a data directory outside the anonymous checkout. The preparation command
downloads approximately 88 MB of upstream files, verifies frozen checksums,
and builds the numeric input matrices. The global matrix alone is about
1.7 GB; intermediate analysis outputs require additional disk and memory.

```sh
python tools/prepare_inputs.py --data-root ../audit-data --download
python tools/reproduce.py --data-root ../audit-data --stage all
```

`all` runs the spectral workflow, additional diagnostics, offline content-label
replay, and rendering of figures from the frozen public result tables. It does
not invoke a remote annotation service or run the direct logistic-MIRT
comparison in the spectral environment. The latter has separate instructions.

For existing copies of the two upstream input files:

```sh
python tools/prepare_inputs.py --data-root ../audit-data \
  --score-zip ../downloads/leaderboard_score.zip \
  --mmlu-pro-parquet ../downloads/mmlu_pro_test.parquet
```

The [data guide](docs/DATA.md) explains alignment, provenance, trust checks, and
redistribution boundaries. The [reproduction guide](docs/REPRODUCE.md) gives
stage-by-stage commands, expected output locations, direct-MIRT instructions,
and numerical checks. A complete rerun is a research workload; runtime depends
on matrix sizes, selected dimensions, numerical libraries, and hardware.

## Audit your own benchmark

Prepare a complete binary item-by-model response matrix and explicit model IDs,
families, owners, and item groups. The maintainer interface accepts a single
NPZ archive or an NPY matrix with aligned metadata CSV files.

Try the interface without downloading the paper data:

```sh
python examples/make_synthetic_benchmark.py --output ../audit-demo.npz
family-dif-audit ../audit-demo.npz --quick --output ../audit-demo-output
```

Then run the primary settings on your data:

```sh
family-dif-audit ../benchmark.npz --output ../benchmark-audit
```

`--quick` reduces the dimension grid to K <= 32 and random controls to 199; it
is an exploratory interface check, not the paper configuration or a runtime
guarantee. The default uses the full candidate grid and 1,000 random controls.
Read `report.md` and `summary.json` in the chosen output directory. They report
selected K, eligible pair counts, reversal rates, matched-random median,
excess in percentage points, and the empirical p-value.

No eligible pairs produces an explicit non-estimable result. A nonsignificant
result does not certify general robustness. The [maintainer guide](docs/MAINTAINER_GUIDE.md)
documents input examples, sample-size checks, exact tie handling, anonymized
output aliases, settings, and the Python API.

## Reproduce content statistics without model calls

The release includes anonymous numeric labels for 500 items in 250 matched
pairs, each labeled by two annotators. Recompute the reliability, paired
enrichment, and exploratory family-direction statistics locally:

```sh
family-dif-content-replay replay \
  --labels results/content_audit/labels.csv \
  --reference-dir results/content_audit \
  --output-dir ../audit-data/outputs/content_audit_replay
```

The replay compares six reconstructed tables with the released references and
records `replay_verification.json`. It requires no question text or model API.
Fresh annotation is optional and uses an explicit provider-neutral backend;
it requires access rights for the question data and may incur provider charges.
New judgments need not be identical to the frozen annotations. See
[the content-audit guide](docs/CONTENT_AUDIT.md).

## Repository organization

```text
configs/       frozen settings, upstream checksums, annotation rubric and schema
docs/          reproduction, data, diagnostics, content, maintainer, and figure guides
examples/      synthetic maintainer-interface example
figures/       paper figures and numerical verification
results/       frozen, anonymous reference tables and numeric content labels
  primary/                 five benchmarks, K selection, bootstrap and random controls
  gap_sensitivity/         cumulative score-gap analyses
  population_robustness/   ten population specifications
  item_stability/          owner-half replication and item-group attribution
  direct_mirt/             direct logistic-MIRT comparison
  discrimination/          discrimination/reliability-matched controls
  within_family/           paired family-specificity comparison
  capability_profile/     matching support and all replacement scenarios
  content_audit/           anonymous labels and all reported statistical summaries
src/family_dif_benchmark_audit/
  core/          spectral, residual-DIF, and direct logistic-MIRT estimators
  ranking/       primary scoring, pooling, and exact-common-model comparisons
  robustness/    population and cumulative score-gap sensitivity
  diagnostics/   additional matched-control and family-specificity analyses
  interpretation/ item signatures, blinded annotation, and offline replay
  maintainer.py  CLI and API for a new benchmark
tests/         numerical, leakage, matching, input-contract, and release checks
tools/         preparation, staged reproduction, figures, and release audit
provenance/    public manifests, checksums, and verification records
```

`results/` contains frozen references. Fresh private outputs belong in the
external data directory and are not automatically copied into the release.
Raw benchmark text, original model identifiers, response matrices, annotation
rationales, provider logs, credentials, and private paths are not distributed.
Anonymous numeric annotation records and aggregate random-control/bootstrap
replicates are included where they support statistical replay.

## Interpretation and attribution

The primary finding is sensitivity of some near-tied cross-family rankings to
family-DIF-guided reweighting beyond matched-random variation. Five static,
mostly closed-form benchmarks from one response collection do not establish
universal behavior across evaluation tasks. MMLU and MMLU-Pro share lineage.
The [results guide](docs/RESULTS.md) and [artifact map](docs/PAPER_ARTIFACT_MAP.md)
identify the numerical evidence and its limits.

Authorship is listed as **Anonymous** for review.

## License

The authors' original code, documentation, figures, and released result files
in this repository are available under the [MIT License](LICENSE), to the
extent the authors hold rights in those materials.

Third-party dependencies, benchmark datasets, question text, and other
externally obtained materials retain their upstream licenses and terms.
This grant does not relicense RouterEval, MMLU-Pro, or other third-party
benchmark content. Raw benchmark inputs are obtained separately as described
in the [data guide](docs/DATA.md).
