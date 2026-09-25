# Documentation claims and evidence

This review checks public documentation against the released implementation,
numeric tables, and recorded packaging validation. It does not claim a new
five-benchmark response-level experiment or validation of unobserved inputs.

| Claim | Evidence and scope |
| --- | --- |
| MIT licensing | Root `LICENSE` and `pyproject.toml` cover the authors' original contributions; README excludes third-party benchmark content and dependencies from this grant. |
| Primary and supplementary coverage | `docs/PAPER_ARTIFACT_MAP.md` maps each implemented analysis to its entry point and released tables. `tools/reproduce.py --stage all` excludes direct-MIRT fitting and fresh remote annotation, as documented. |
| Primary, direct-MIRT, gap and diagnostic numerical results | Existing `tests/test_artifact_integrity.py` reconstructs rates, medians and empirical tails from the released counts; `tests/test_workflow_tools.py` checks CV selection and the paired estimator comparison. |
| Paired family specificity | `diagnostics/within_family.py` subtracts the median of paired random cross-minus-within differences from the observed difference. This is generally different from subtracting two separately computed excesses. |
| Item-group replication | `results/item_stability/source_validation_summary.csv` reports 58/60 agreeing effect signs; `source_shift_driver_validation_summary.csv` reports 43/60 agreeing contribution signs. These counts do not assert equality of magnitudes. |
| Discrimination control | `diagnostics/discrimination.py` matches strata of audit-half predicted-logit spread. Held-out CITC, score SEM, and Cronbach alpha are balance diagnostics rather than direct matching inputs. |
| Maintainer output | `maintainer.py` pools counts across two owner-disjoint target halves. Its exported fold counts support inspection; its headline test does not establish independently significant replication in both halves. |
| Test coverage | The spectral environment executes its available tests and skips the optional PyTorch module. Direct-MIRT tests use the separate estimator environment. Passing tests alone does not establish full raw-data recomputation. |
| Annotation replay | `interpretation/content_replay.py` checks six statistical tables from released numeric labels. Semantic correctness and fresh hosted-model judgments are outside that replay guarantee. |
| Reproduction paths | Diagnostic and fresh-annotation examples explicitly use the prepared external data root; public configuration and schema paths remain relative to the repository. |

The detailed checks actually performed during initial packaging are recorded
in `PACKAGING_VALIDATION.md`. No experimental result table or estimator was
changed by the licensing and documentation update.
