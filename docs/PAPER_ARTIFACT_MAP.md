# Paper-to-artifact map

This map covers the paper's analysis implementation and promised reproducible
materials. Topic names are used instead of table numbers so that the map
remains usable when the manuscript's layout changes. Paths in the reference
column are in the anonymous repository. Generated paths described in
[REPRODUCE.md](REPRODUCE.md) are under the separate prepared data root.

## Primary analysis and supplementary experiments

| Paper result or method | Implementation / command | Released reference or protocol |
|---|---|---|
| Input population, family-name matching, whole-owner split, cap 5, median-score owner-family representative | `core/family_dif.py`; `ranking/mmlu_pro.py`; `family-dif-ranking` | `configs/analysis_protocol.json`; `results/primary/per_benchmark/*/protocol.json` |
| Family-label-free spectral representation and residual family-DIF fitting | `core/spectral_mirt.py`; `ranking/mirt_primary.py` | `results/primary/per_benchmark/*/protocol.json` |
| Common K grid, nested owner/item validation, one-SE rule, feasible ceilings | `core/spectral_mirt.py:choose_spectral_dimension` | `results/primary/per_benchmark/*/dimension_cv.csv` and `dimension_selection.csv` |
| Three-round purification, low-DIF selection, exact blueprint weight restoration | `core/spectral_mirt.py:purify_spectral_mirt_anchors`; `ranking/mmlu_pro.py` | `results/primary/per_benchmark/*/anchor_purification.csv`; protocol settings |
| Primary five-benchmark strict near-tie reversal table and global Kendall correlations | `family-dif-ranking`; `family-dif-summary` | `results/primary/benchmark_summary.csv`; per-benchmark `ranking_impact_metrics.csv` |
| Equally short matched-random controls, count pooling, excess and empirical p | `ranking/mmlu_pro.py:matched_random_anchor_controls`; primary/summary runners | Per-benchmark `matched_random_anchor_controls.csv` and `matched_random_anchor_control_summary.csv` |
| Anchor fractions 0.3, 0.5, 0.7 | Primary runner, once K is selected | Per-benchmark `ranking_impact_metrics.csv`; controls are computed for the 0.5 primary fraction |
| Composition block bootstrap and fixed original near-tie mask | `ranking/mmlu_pro.py:source_block_bootstrap`; primary runner | Per-benchmark `composition_bootstrap.csv` and `source_block_bootstrap_summary.csv` |
| Family score/rank shifts and owner-bootstrap summaries | Primary runner; `family-dif-summary` | `results/primary/family_shift_summary.csv`; per-benchmark `family_rank_shifts.csv` and `owner_bootstrap_family_shifts.csv` |
| Three exact-common-model benchmark-pair comparisons, all five families | `ranking/exact_common.py`; `family-dif-summary` | `results/primary/pairwise_exact_common_family_shift_summary.csv` (15 rows) |
| Cumulative full-score gap sensitivity, 0.25–5 pp | `family-dif-gap-sensitivity` | `results/gap_sensitivity/gap_sensitivity_summary.csv`, observed and random fold curves |
| Ten population specifications: baseline, caps, score-blind checkpoint, conservative lineage, leave-one-family-out | `family-dif-owner-robustness` | `results/population_robustness/robustness_summary.csv`, `selection_audit.csv`, and `protocol.json` |
| Owner-half residual item-signature replication and within-cell permutations | `family-dif-item-stability` | `results/item_stability/stability_summary.csv` and `protocol.json` |
| Discovery-selected item-group effect signs and exact group score-shift contributions | `interpretation/item_stability_v3.py`, same command | `results/item_stability/source_validation_summary.csv`, `source_shift_driver_validation_summary.csv`, and `source_shift_decomposition_audit.csv` |
| Direct penalized joint logistic-MIRT comparison, with reselected K and anchors | `family-dif-direct-mirt` in the separate direct-MIRT environment; `tools/summarize_direct_mirt.py` | `results/direct_mirt/summary.csv`, `paired_cv_comparison.csv`, per-benchmark CV/selection/settings/primary tables |
| Predicted-logit-spread-matched control and held-out reliability diagnostics | `family-dif-discrimination` | `results/discrimination/summary.csv`, fold diagnostics, both control replicate tables, `protocol.json` |
| Within-family excess and paired cross-minus-within specificity | `family-dif-within-family` | `results/within_family/summary.csv`, observed fold metrics, paired control replicates, `protocol.json` |
| Capability-profile overlap: random pairing, unrestricted optimal assignment, and three calipers | `family-dif-capability-profile`; `--verify-only` | `results/capability_profile/summary.csv`; per-benchmark matching quality, observed/control/all-swapped aggregates and verification; `protocol.json` |

Module paths above are relative to `src/family_dif_benchmark_audit/`.
The diagnostic details and independent checks are described in
[DIAGNOSTICS.md](DIAGNOSTICS.md). `source` in retained code or column names
means the paper's metadata-defined item group. It does not denote a model
family or an inferred latent skill.

The capability-profile rows include all scenarios and matching-support
quantities. Unrestricted assignment does not guarantee close capability
profiles, while strict calipers may leave almost all anchor weight unchanged.
These diagnostics therefore do not establish capability-equivalent subtests.

## Blinded content audit

| Commitment | Implementation / command | Released material |
|---|---|---|
| Frozen annotation task and all binary/ordinal axes | `family-dif-content-audit prepare` | `configs/blinded_content_rubric.md`; `configs/blinded_content_annotation_schema.json` |
| Blinded pair sampling and private unblinding procedure | `interpretation/blinded_content_v3.py:prepare` | Code plus `results/content_audit/protocol.json` and pre-annotation audit |
| Portable interface for optional fresh model judgments | `family-dif-content-audit annotate --backend-command ...` | Backend contract in `docs/CONTENT_AUDIT.md`; no service credentials or calls bundled |
| Frozen anonymous numeric judgments sufficient for statistical replay | `family-dif-content-replay replay` | `results/content_audit/labels.csv`: 1,000 records, 500 items, 250 pairs |
| Binary inter-annotator reliability | Content analysis and offline replay | `results/content_audit/binary_annotation_reliability.csv` |
| Ordinal inter-annotator reliability | Same | `results/content_audit/ordinal_annotation_reliability.csv` |
| Paired binary prevalence differences, exact paired tests, BH adjustment | Same | `results/content_audit/paired_binary_enrichment.csv` |
| Paired ordinal comparisons | Same | `results/content_audit/paired_ordinal_enrichment.csv` |
| Exploratory family-direction associations and benchmark-stratified permutation tests | Same | `results/content_audit/exploratory_family_direction_association.csv` |
| Family-direction prevalence breakdown | Same | `results/content_audit/exploratory_family_direction_prevalence.csv` |

The six statistical tables are recomputed and checked by offline replay.
The numeric labels have fresh item/pair/cell IDs; original item mappings,
question text, answers, rationales, and execution logs are not distributed.
Statistical replay needs no remote calls. Fresh annotation requires authorized
question data and a supplied provider backend, may incur charges, and is not
guaranteed to reproduce the frozen judgments bit for bit.

## Figures, practical interface, and release integrity

| Commitment | Public path | Verification scope |
|---|---|---|
| Illustrative benchmark-recomposition overview | `tools/generate_overview.py`; `docs/FIGURES.md` | Reconstructs the schematic; displayed scores are illustrative |
| Gap-excess figure | `tools/render_paper_figures.py`; `figures/gap_excess_verified.pdf` | Checks primary/gap table consistency before rendering |
| Population-robustness figure | Same renderer; `figures/population_robustness_verified.pdf` | Checks complete benchmark/specification coverage and frozen numerical claims |
| Content-audit figure | Same renderer; `figures/content_audit_verified.pdf` | Draws paired prevalence differences from released numeric tables |
| Primary protocol and hyperparameter disclosure | `configs/analysis_protocol.json`; per-analysis `protocol.json` and direct-MIRT settings | Frozen numerical settings and seeds; rationales do not imply hyperparameter optimality |
| Raw-input reconstruction | `tools/prepare_inputs.py`; `configs/upstream_inputs.json`; `configs/routereval_item_layout.json` | Verified source bytes and expected global matrix; local `input_verification.json` |
| Staged reproduction entry point | `tools/reproduce.py`; `docs/REPRODUCE.md` | Orders spectral stages and makes their input/output dependencies explicit |
| New-benchmark maintainer CLI/API | `family-dif-audit`; `maintainer.py`; `docs/MAINTAINER_GUIDE.md` | Explicit labels, owner-disjoint fitting, exact blueprint controls, counts and non-estimable states |
| Synthetic interface example | `examples/make_synthetic_benchmark.py` | Demonstrates the input contract; not a scientific validation experiment |
| Numerical and contract tests | `tests/` | Includes holdout invariance, blueprint mass, strict ties, matching, replay, and release checks |
| Public artifact audit and checksums | `tools/audit_release.py`; `provenance/` | Checks the curated public tree and records public-file digests |

The figure guide distinguishes the illustration from data-derived figures.
The staged `figures` command draws frozen public result tables; it is not a
claim to have refitted models or regenerated the underlying empirical results.

## What is not claimed by this artifact

Passing unit tests, rendering reference figures, and replaying numeric
annotations do not themselves constitute a fresh end-to-end run of every
large experiment. Full response-level recomputation requires the verified
external data tree and the commands in the reproduction guide. The direct-MIRT
comparison has its own fitting environment and stopping diagnostics.

The public release excludes raw question/model records and privately generated
intermediates. Those records can be reconstructed locally where upstream terms
permit. It includes anonymous numeric labels and aggregate replicate tables
where needed for statistical replay, so exclusion of raw text does not prevent
recomputing the published annotation statistics.

The code still contains some supporting scalar-IRT and discovery utilities
used by shared modules. Their presence does not change the paper's primary
estimator, which is the spectral pipeline. Unperformed penalty sweeps,
new ground-truth simulations, or new disjoint-gap analyses should not be
inferred from a general promise of reproducibility; this map lists implemented
paper analyses explicitly.
