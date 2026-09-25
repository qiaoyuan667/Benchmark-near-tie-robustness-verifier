# Packaging validation

These checks validate the release migration, not a new scientific analysis.
This is the initial packaging record; subsequent clean-checkout numerical
reruns are documented in [Reviewer-workflow verification](REPRODUCIBILITY_CHECK.md).

- Reconstructed the global 50,265-by-8,577 response matrix from pinned upstream
  files; its SHA-256 matches the original matrix. Item and model metadata match.
- Reconstructed MMLU-Pro responses and model ordering: both exactly match the
  original 12,032-by-1,823 arrays.
- Ran the portable WinoGrande primary pipeline on reconstructed inputs: ranking,
  matched-random, composition-bootstrap and population outputs match the frozen
  run to 1e-12. All 18 candidate-wise CV rows match the final uniform-256 export.
- Replayed all three new diagnostic runners on WinoGrande using the pinned
  spectral environment; summaries match original outputs to 1e-12, with primary
  integer reversal/eligibility counts exactly reproduced.
- Independently verified 40 capability-profile scoring configurations, all five
  matching scenarios, caliper feasibility, pooled tails, and relevant hashes.
- Recomputed all six content-audit tables from public anonymous labels, including
  the 10,000-permutation exploratory family-direction analysis.
- Compared the direct-MIRT core with the original implementation on free and
  conditional synthetic fits: parameters and objective histories match exactly.
- Rebuilt the direct-versus-spectral CV comparison from the final per-benchmark
  tables. An earlier comparison used a pre-extension WinoGrande spectral table
  and its inner join omitted the two K=256 rows. The public comparison includes
  all 88 rows and the final spectral losses. Selected dimensions and primary
  ranking outcomes are unchanged. The two WinoGrande one-SE threshold metadata
  entries are synchronized with those final CV losses rather than the earlier
  grid; this does not change either selected dimension.

Unit tests additionally check owner disjointness, held-out-data invariance,
blueprint mass, strict reversal semantics, random-control pooling, missing-pair
behavior, anonymized output, and numerical aggregates. Tests requiring PyTorch
are optional in the spectral environment and are run in the direct-MIRT
environment when validating that estimator. At initial packaging, full
five-benchmark primary and direct-MIRT fitting were not rerun merely to prepare
the release; the later verification record gives the expanded coverage.

Floating-point ties can change across numerical-library versions. The pinned
spectral and direct-MIRT environments are separate. Do not relax a failed strict
tie or population-consistency assertion to obtain apparent reproduction.
