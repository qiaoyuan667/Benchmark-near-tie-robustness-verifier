# Reproduction guide

Run shell commands from the repository root after activating the environment.
There are three different checks: recomputing statistics from released numeric
records, rerunning estimation from item-level responses, and drawing figures
from result tables. Success at one level is not a claim that the other levels
were rerun.

## 1. Install the spectral environment

The pinned spectral environment was checked with Python 3.12.14:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python -m pip install -r requirements-pinned.txt
python -m pip install -e . --no-deps
python -m pytest -q -p no:cacheprovider
```

For package entry-point help, use `family-dif-ranking --help`,
`family-dif-audit --help`, or the corresponding command below. The spectral
requirements and direct-MIRT requirements must be installed in separate
environments: differences in floating-point operations can change strict ties.

## 2. Check the public artifact without downloading data

```sh
python tools/render_paper_figures.py --output-dir ../audit-data/outputs/paper_figures
family-dif-content-replay replay \
  --labels results/content_audit/labels.csv \
  --reference-dir results/content_audit \
  --output-dir ../audit-data/outputs/content_audit_replay
python tools/audit_release.py
```

The figure renderer validates frozen claims before drawing the gap,
population-robustness, and content-audit figures. It reads public `results/`
tables, not raw responses. The content replay recomputes six statistical tables
from 1,000 anonymous annotation records with the frozen procedures and seed,
including 10,000 permutations for the exploratory family-direction analysis.
Inspect `../audit-data/outputs/paper_figures/verification.json`, the replay's
`replay_verification.json`, and the release audit's printed JSON result.
The public checksums under `provenance/` are verified without rewriting them.

The overview schematic is illustrative: its scores are not experimental
results. Its generator and the paper-style figure workflow are documented in
[FIGURES.md](FIGURES.md). No remote annotation calls occur in these checks.

## 3. Download and prepare the numeric inputs

```sh
python tools/prepare_inputs.py --data-root ../audit-data --download
```

This downloads the documented RouterEval score archive and MMLU-Pro parquet,
verifies frozen upstream checksums before deserializing score files, and builds
the matrix and metadata paths expected by the paper runners. It writes
`../audit-data/input_verification.json`. Existing copies can be supplied with
`--score-zip` and `--mmlu-pro-parquet`; see [DATA.md](DATA.md).

The approximately 88 MB download expands into a global matrix of about 1.7 GB,
plus separate benchmark matrices and metadata. Allow additional disk and memory
for fits, control replicates, and bootstrap outputs. Numeric experiments use
item-group metadata and binary correctness; fresh annotation additionally
requires authorized question text.

For manual commands, set this before starting each Python process:

```sh
export FAMILY_DIF_PROJECT_ROOT="$PWD/../audit-data"
```

Despite its name, this environment variable selects the prepared data and
generated-output root. The staged runner sets it from `--data-root` itself.
Keep the repository checkout and prepared data root separate.

## 4. Run the spectral paper workflow by stage

```sh
python tools/reproduce.py --data-root ../audit-data --stage primary
python tools/reproduce.py --data-root ../audit-data --stage summary
python tools/reproduce.py --data-root ../audit-data --stage gap
python tools/reproduce.py --data-root ../audit-data --stage population
python tools/reproduce.py --data-root ../audit-data --stage stability
python tools/reproduce.py --data-root ../audit-data --stage diagnostics
python tools/reproduce.py --data-root ../audit-data --stage content
python tools/reproduce.py --data-root ../audit-data --stage figures
```

The equivalent sequential workflow is:

```sh
python tools/reproduce.py --data-root ../audit-data --stage all
```

To inspect the planned commands without running them, add `--dry-run`.
The runner records each attempted stage under `outputs/reproduction_log/` and
refuses to silently repeat one. Inspect its log before explicitly using
`--rerun`; that option does not override stricter nonempty-directory checks
inside individual diagnostic runners.

Stages use the prepared data and outputs of their predecessors. `primary`
must finish all five benchmarks before `summary`; `gap` needs primary results
and their summary. `population`, `stability`, and the extra diagnostics reuse
the primary outputs. `content` is offline replay of released frozen labels,
not fresh annotation. `figures` renders the frozen public result tables into
the data root's `outputs/paper_figures/`; it does not by itself validate freshly
recomputed tables. Direct logistic MIRT is run separately below.

### Primary fitting and exact-common comparisons

The explicit primary commands are:

```sh
family-dif-ranking --benchmark mmlu_pro
family-dif-ranking --benchmark bbh
family-dif-ranking --benchmark mmlu
family-dif-ranking --benchmark hellaswag
family-dif-ranking --benchmark winogrande
family-dif-summary
```

For each benchmark the run selects K once per outer owner half, then fits
anchor fractions 0.3, 0.5, and 0.7 using three purification rounds. The 0.5
primary form receives 1,000 matched-random controls and 2,000 composition
bootstrap replicates. Family shifts also receive owner-bootstrap summaries.
The low-DIF fraction sensitivities report the quantities actually computed by
the primary runner; matched-random controls are not automatically recomputed
for every nonprimary fraction.

Per-benchmark files are created under:

```text
../audit-data/outputs/<benchmark>_family_ranking_impact/
  protocol.json
  family_models.csv
  owner_family_representatives.csv
  dimension_cv.csv
  dimension_selection.csv
  anchor_purification.csv
  crossfit_anchor_items.csv
  crossfit_model_coordinates.csv
  ranking_impact_metrics.csv
  matched_random_anchor_controls.csv
  matched_random_anchor_control_summary.csv
  composition_bootstrap.csv
  family_rank_shifts.csv
```

The summary command also runs the three frozen exact-common-model benchmark
pairs and produces the complete 15-row family comparison. Its output root is
`outputs/v3_five_benchmark_mirt_primary/`, including `benchmark_summary.csv`,
`family_shift_summary.csv`, and
`pairwise_exact_common_family_shift_summary.csv`.

Expected primary results, rounded for presentation:

| Benchmark | Selected K, discovery / validation | Eligible near-tie pairs | Low-DIF reversal % | Random median % | Excess pp | p |
|---|---:|---:|---:|---:|---:|---:|
| MMLU-Pro | 64 / 32 | 308 | 47.1 | 18.5 | 28.6 | .001 |
| BBH | 128 / 128 | 2,226 | 42.1 | 25.2 | 16.9 | .001 |
| MMLU | 128 / 128 | 3,235 | 40.7 | 16.3 | 24.4 | .001 |
| HellaSwag | 16 / 64 | 4,533 | 30.9 | 11.4 | 19.5 | .001 |
| WinoGrande | 8 / 8 | 5,574 | 33.8 | 34.7 | -0.9 | .689 |

The requested grid is 1, 2, 4, 8, 16, 32, 64, 128, 256. The inner fitting
matrix permits a maximum of 128 for MMLU-Pro and 256 for the other benchmarks.
Use full-precision reference tables in `results/primary/`, rather than the
rounded table above, when checking numerical agreement.

### Gap, population, and item-signature analyses

Manual commands, after primary fitting and summary:

```sh
family-dif-gap-sensitivity
family-dif-owner-robustness
family-dif-item-stability
```

| Analysis | Generated directory below the data root | Main reference |
|---|---|---|
| Cumulative gap thresholds 0.25, 0.5, 1, 2, 3, 5 pp | `outputs/v3_ranking_gap_sensitivity/` | `results/gap_sensitivity/gap_sensitivity_summary.csv` |
| Ten population specifications | `outputs/v3_family_owner_robustness/` | `results/population_robustness/robustness_summary.csv` |
| Owner-half item-signature replication | `outputs/v3_item_dif_signature_stability/` | `results/item_stability/stability_summary.csv` |
| Item-group effects and score-shift contributions | Same item-stability directory | `results/item_stability/source_validation_summary.csv` and `source_shift_driver_validation_summary.csv` |

Population variants retain the primary selected K and refit directions, DIF,
anchors, and controls. Nonbaseline variants use 200 random controls; the
baseline retains the 1,000-control primary result. Item stability uses 500
within-cell permutations, with minimum empirical p-value 1/501. Fields using
`source` in file and column names denote the manuscript's metadata-defined
item groups. Threshold flags retained in implementation outputs are operational
checks, not universally validated standards of metric quality.

### Additional matching and specificity diagnostics

```sh
family-dif-discrimination --output-dir "$FAMILY_DIF_PROJECT_ROOT/outputs/discrimination_matched"
family-dif-within-family --output-dir "$FAMILY_DIF_PROJECT_ROOT/outputs/within_family"
family-dif-capability-profile --output-dir "$FAMILY_DIF_PROJECT_ROOT/outputs/capability_profile"
family-dif-capability-profile --output-dir "$FAMILY_DIF_PROJECT_ROOT/outputs/capability_profile" --verify-only
```

These commands default to all five benchmarks and load raw data from
`FAMILY_DIF_PROJECT_ROOT`. The explicit output paths match the staged runner:
`outputs/discrimination_matched/`, `outputs/within_family/`, and
`outputs/capability_profile/`.
They require complete regenerated primary records, not only public aggregates.
Use `--benchmarks mmlu_pro` for a single-benchmark check and `--output-dir` for
a fresh diagnostic destination. The `--primary-root` argument, if used, names
the parent of the five `<benchmark>_family_ranking_impact` directories; it does
not relocate the raw responses.

The discrimination and capability analyses reconstruct the exact primary
representation. Checks fail if the model coordinates or strict pair counts
disagree with the frozen input run. Do not loosen equality checks to force a
match. The [diagnostics guide](DIAGNOSTICS.md) explains the seeds, quantities,
matching rules, paired specificity test, and independent verification.

Expected discrimination-matched excesses are approximately 24.0, 14.2, 21.3,
18.0, and -0.8 pp in benchmark order. Within-family specificity increments are
approximately 18.0, 6.7, 13.9, 9.3, and 0.0 pp. Capability-profile results must
include matching distances and swappable weight coverage for every scenario;
small coverage under tight calipers is not evidence of equivalent capability
demands or evidence that an effect vanished.

## Direct logistic-MIRT comparison

Keep the spectral environment intact and create a separate environment for the
versions in `requirements-direct-mirt.txt`:

```sh
deactivate
python3.12 -m venv .venv-direct-mirt
source .venv-direct-mirt/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python -m pip install -r requirements-direct-mirt.txt
python -m pip install -e . --no-deps
export FAMILY_DIF_PROJECT_ROOT="$PWD/../audit-data"

family-dif-direct-mirt --benchmark mmlu_pro
family-dif-direct-mirt --benchmark bbh
family-dif-direct-mirt --benchmark mmlu
family-dif-direct-mirt --benchmark hellaswag
family-dif-direct-mirt --benchmark winogrande
python tools/summarize_direct_mirt.py --data-root ../audit-data
```

The runner uses direct penalized joint logistic MIRT, CPU float32 fitting,
two threads, Adam, up to 6,000 steps, and two initializations for joint fits.
The conditional coordinate fit uses the fixed trained item parameters. The
objective-plateau stopping rule and all penalties are recorded in
`direct_mirt_settings.json`; failure to meet the stopping rule raises an error.
This is not a guarantee of a global optimum or marginal-likelihood estimation.

Runs are written under `outputs/direct_mirt/<benchmark>_family_ranking_impact/`.
The comparison summary and paired CV table are written under
`outputs/direct_mirt/`. Each benchmark's `verification.json` checks input hashes,
population, representatives, and shared protocol fields against the spectral
reference. Keep the existing spectral outputs available for this check.
`--max-steps` and `--threads` are exposed for explicit alternative runs; changing
them is not the frozen paper setting. Completed output directories are refused.

Expected direct-MIRT K values are 32/32, 128/128, 64/64, 16/16, and 16/16.
The corresponding excesses are approximately 23.1, 7.5, 9.2, 12.0, and 2.7 pp;
the first four have p approximately .001 and WinoGrande has p approximately .087.
Compare against `results/direct_mirt/`. This comparison reruns primary scoring,
not every population, matching, or content diagnostic under direct MIRT.

Return to the spectral environment for its diagnostics or plotting:

```sh
deactivate
source .venv/bin/activate
```

## 5. Content-audit replay and optional fresh annotation

The `content` stage replays the frozen anonymous numeric labels and compares
all six reference tables. This reproduces reported annotation statistics
without transmitting any benchmark question to a service.

Fresh packet preparation and annotation are separate, optional steps. They
require item-signature results, authorized question text aligned to the original
item indices, and a supplied backend executable. They may incur provider
charges. Follow [CONTENT_AUDIT.md](CONTENT_AUDIT.md) for the exact `prepare`,
`annotate`, and `analyze` commands and backend contract. New judgments are not
guaranteed to match the frozen labels, even with the same prompt and model name.

## 6. Validate outputs and render figures

Use the [artifact map](PAPER_ARTIFACT_MAP.md) to pair each generated table with
its released reference. Check K, model and owner counts, eligible denominators,
anchor counts, rates, and empirical tails before comparing rounded prose.
The primary, gap, and additional-diagnostic implementations contain internal
consistency checks. Capability-profile outputs additionally support
`--verify-only`. A complete numerical rerun should record its environment and
its comparisons; the packaged aggregates alone do not document a fresh rerun.

To redraw the validated public figures in a separate destination:

```sh
python tools/render_paper_figures.py \
  --results-dir results \
  --output-dir ../audit-data/outputs/paper_figures
```

The renderer expects the release-style directory layout and validates the
paper's frozen claims. Passing another `--results-dir` is appropriate only
after assembling the corresponding recomputed aggregate files in that layout;
the renderer does not read every private fitting output automatically. See
[FIGURES.md](FIGURES.md) for all figure inputs and the illustrative diagram.

To regenerate the separate overview illustration, install its optional plotting
pin and select an external destination:

```sh
python -m pip install -r requirements-figures.txt
python tools/generate_overview.py --output-dir ../audit-data/outputs/overview
```

## Troubleshooting and responsible interpretation

- **Missing primary files:** complete `primary` and `summary` first. Public
  aggregate tables are not substitutes for raw matrices or crossfit anchors.
- **Input checksum mismatch:** stop and determine whether the upstream file
  changed. Do not bypass the check before loading a pickle.
- **Different K or pair counts:** check data order, seeds, population settings,
  Python/dependency versions, and numerical libraries. Exact ties are excluded;
  tiny floating-point changes can alter a denominator.
- **Direct-MIRT convergence failure:** preserve the failed logs and report the
  failure. An increased iteration budget is a declared alternative setting.
- **Rerunning stages:** use a new data/output directory or inspect the stage's
  resume behavior. Some research runners write into existing directories;
  do not treat them as an automatic checkpoint/resume system.
- **No significant excess:** report the observed rate and available pair count.
  Lack of evidence for this specific effect is not a robustness certificate.

Full runs create original model identifiers and potentially sensitive
record-level outputs in the external data directory. Do not upload that
directory with the anonymous artifact. The release audit applies to the
curated public checkout, not to arbitrary raw data or GitHub account identity.
