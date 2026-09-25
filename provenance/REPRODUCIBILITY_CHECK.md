# Reviewer-workflow verification

Verification date: 2026-09-25. This check started from a fresh public Git clone,
not an environment that already contained the research project's imports or
private input paths. Fixes were then tested in that separate checkout.
The original research outputs and released numerical references were not edited.

Machine-readable evidence: [installed commands](command_smoke_verification.json),
[fresh primary comparison](fresh_primary_comparison.json),
[complete spectral comparison](full_spectral_comparison.json), and
[environment versions](verification_environments.json). These records contain
no original model-owner identifiers or local machine paths.

## Installation and executable interfaces

- Fresh Python 3.12.14 spectral environment: installed the pinned requirements
  and editable package using the documented commands; `pip check` passed.
- A separate direct-MIRT environment installed successfully. The spectral
  suite passed 115 tests, with the optional PyTorch module skipped. All three
  direct-MIRT tests passed in its own environment.
- The anonymous ZIP was extracted without Git history, installed into a new
  virtual environment using the README commands, and passed tests and the file
  audit. With the Git executable additionally removed from `PATH`, the suite
  passed 113 tests and explicitly skipped the two Git fixtures plus the optional
  PyTorch module; the ZIP file audit still passed.
- All 12 installed entry points and all 13 helper-tool help interfaces were
  exercised. The installed-command smoke workflow passed all 35 checks with
  plotting dependencies present. This includes actual default and quick
  maintainer audits, module and Python API calls, API/CLI agreement, content
  replay, empirical figure rendering, and the staged command plan.
- The NPY/CSV and NPZ maintainer inputs produced equivalent statistical output.
  An explicit-settings maintainer invocation also ran. A separate unpinned
  `pip install -e .` environment ran the synthetic quickstart successfully;
  it is not the reference environment for strict numerical paper reproduction.
- Both documented plotting installation routes retained the spectral numerical
  pins. The overview schematic and three empirical figures rendered. The
  schematic's no-argument invocation writes to generated outputs, not to the
  shipped figure directory.
- All public Python files passed syntax inspection and a check for undefined
  names. A clean Linux CI environment independently passed unit tests, the
  installed-command smoke workflow, plotting installation, and artifact audit.
  CI does not rerun the full raw-data paper experiments.

## Data and numerical reruns

The two upstream inputs were freshly downloaded at immutable revisions and
checked against their frozen hashes. Preparation reconstructed the global
response matrix and MMLU-Pro arrays/model ordering. The supplied-local-input
preparation route also passed.

Completed numerical checks include:

- Fresh full primary fits for all five benchmarks, including
  dimension selection, three anchor fractions, the 1,000-control primary
  comparison, and 2,000 composition-bootstrap replicates. All 50 corresponding
  released CSV tables matched at `rtol=atol=1e-12`.
- Fresh full WinoGrande direct-MIRT fitting, not just a synthetic optimizer
  test. All eight released per-benchmark CSV tables matched at the same
  tolerance. Input/population/shared-protocol checks against the spectral
  reference passed.
- All three five-benchmark primary aggregate tables were regenerated from those
  fresh fits, including the 15-row exact-common-model comparison, and matched
  at the same tolerance: 53 primary tables checked in total.
- All three gap-sensitivity tables matched, including the 60,000-row
  matched-random fold-level table.
- All four item-stability and item-group-attribution reference tables matched
  after rerunning the corrected numeric-only pipeline with the reference
  15-thread OpenBLAS configuration.
- The population-robustness workflow completed all ten specifications for all
  five benchmarks. Its 50-row summary, 225-row selection audit, and 10-row
  criterion summary all matched the released tables at the same tolerance.
- The discrimination/reliability-matched control completed all five benchmarks.
  All four reference tables matched, including both 10,000-row random-control
  tables and all held-out reliability diagnostics.
- The within-family specificity diagnostic completed all five benchmarks.
  Its three reference tables matched, including 10,000 paired-control records.
- The capability-profile diagnostic completed all five benchmarks and all five
  matching scenarios. All 21 reference tables matched. The separate
  `--verify-only` command passed 200 independently rescored configurations,
  matching/caliper/weight checks, all pooled empirical tails, and input/code/
  output checksum checks.
- All six content-audit tables were recomputed from the 1,000 frozen annotation
  records, including the 10,000-permutation exploratory analysis, and passed
  the replay's reference checks.
- The direct-versus-spectral summary command regenerated its two aggregate
  tables from released per-benchmark numeric records. This validates the
  aggregation command, not additional direct-MIRT fits.

Downstream integration checks reused copied primary fitting records and freshly
prepared raw responses. They are distinct from the fresh primary reruns above;
reusing those records is not presented as another independent primary fit.
All 25 reused downstream-input tables (family models, representatives, anchor
items, model coordinates, and model scores for five benchmarks) were compared
with the fresh primary outputs and matched at `rtol=atol=1e-12`.
The gap, population, discrimination, and within-family integration runs used
two OpenBLAS/OMP threads. Capability-profile fitting and independent verification
use the implementation's four-thread context. No statistical setting, seed,
matching rule, replicate count, or comparison tolerance was relaxed.

The combined spectral comparison contains **91 matching tables, zero missing
tables, and zero mismatches**. Integer columns are checked exactly; floating
columns use `rtol=atol=1e-12`. The eight fresh WinoGrande direct-MIRT tables and
six content-replay tables are additional checks outside that 91-table total.

## Defects found and corrected

1. Reader-side verification incorrectly required an anonymous *local* Git
   identity. Normal verification now checks published history; publisher checks
   still require anonymous author and committer configuration.
2. An in-repository virtual environment was being scanned as release content.
   Known environment/build/output directories are now excluded consistently;
   unexpected raw data outside those directories is still rejected.
3. Installing the plotting pin alone could upgrade NumPy through a transitive
   dependency and break the fixed SciPy environment. Plotting installation now
   uses the spectral constraints, with `pip check` verified afterward.
4. Fresh annotation transport changed the backend's working directory, and its
   default schema path depended on the external data root. The backend now uses
   the caller's working directory and the bundled schema is resolved correctly.
5. The numeric item-stability stage unnecessarily required undistributed
   question text for example exports. Text is now explicitly optional and does
   not affect fitting, permutations, source attribution, or item selection.
6. The direct-MIRT port omitted the original BLAS/OpenMP startup defaults.
   Those defaults were restored; runtime thread limits also apply to supported
   BLAS pools. Apple Accelerate is not exposed by `threadpoolctl`.
7. Upstream download URLs now pin immutable revisions instead of mutable branch
   tips. Both pinned downloads were checked against the existing frozen hashes.
8. Two identity-fixture tests assumed the Git executable was installed even for
   ZIP readers. They now explicitly skip if Git is absent; they still execute
   in Git-enabled environments and CI. Numeric tests are not skipped for this.

Regression tests cover the corrected failure modes. The new
`tools/compare_reproduction.py` reports missing and mismatching reference tables
instead of treating command completion alone as numerical reproduction.

## Verification boundaries

- The full-data runs above used macOS/ARM64 and separate pinned numerical
  environments. Linux CI checks executability and unit-level numerical
  contracts, not full scientific fits. Windows and all declared Python versions
  have not been exhaustively tested.
- The other four direct-MIRT benchmarks were not refitted in this verification
  pass. Their released statistics and comparison summaries are checked, and the
  shared direct fitting implementation is exercised by WinoGrande and unit tests.
- Fresh annotation was exercised with a local mock backend through preparation,
  batching, schema validation, analysis, and label export. No live provider call
  or new human semantic validation was performed. Users must supply an authorized
  question source and their own backend for fresh remote annotation.
- Legacy question-mapping helpers require separately supplied text/mapping
  inputs; help-interface checks do not establish access to those external files.
  They are not required by the documented numeric reproduction pipeline.
- Strict ties and rank correlations can be sensitive to numerical-library,
  BLAS, and thread-count changes. A two-thread item-stability run reproduced the
  main indicators but differed in HellaSwag's auxiliary raw-magnitude Spearman
  correlation by about `5.4e-7`. This difference was reported by the strict
  comparator, not hidden by changing its tolerance. Repeating the full
  item-stability stage at the reference 15-thread setting reproduced all four
  tables to the original `1e-12` tolerance.

For a new run, follow `docs/REPRODUCE.md`, retain the environment information,
and save a fresh comparison report. Passing these checks is evidence for the
tested workflows, not a guarantee that every platform or future dependency
combination will behave identically.
