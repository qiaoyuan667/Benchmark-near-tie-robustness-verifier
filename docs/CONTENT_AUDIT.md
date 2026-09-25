# Reproducing the blinded content audit

There are two distinct reproduction tasks. Statistical replay reproduces the
paper's results from the frozen numeric annotations and makes no model calls.
Fresh annotation constructs the blinded packets again and asks the two named
models for new judgments. Fresh model judgments are not guaranteed to be
bitwise identical because inference and hosted model implementations can vary.

## Offline statistical replay

From the repository root, with this package installed:

```bash
python -m family_dif_benchmark_audit.interpretation.content_replay replay
```

The command reads `results/content_audit/labels.csv`, recomputes six tables,
and compares every table against the frozen public aggregates. Outputs go to
`outputs/content_audit_replay/`; `replay_verification.json` reports the checks.
It uses the original paired binary tests, ordinal tests, reliability functions,
Benjamini-Hochberg adjustments, and 10,000 benchmark-stratified permutations
with the original random seed and record order.

Expected results are 1,000 annotation records (500 items from 250 matched
pairs, each labeled by both annotators), median non-degenerate binary Cohen's
kappa approximately 0.715454, median ordinal Spearman correlation approximately
0.800084, no replicated confirmatory content axis, and 158 high-DIF items with
stable advantaged-family labels. None of the exploratory family-direction
axes survives correction in both annotators.

The label file contains only the two annotator/model names, benchmark and
experimental group, newly assigned item/pair/cell identifiers, binary and
ordinal labels, and categorical advantaged-family labels. It contains no
question text, rationale, original item index, original item-group name,
model-owner identity, fitted item-effect vector, or private file path. The
label table supports statistical replay; it does not independently validate
the semantic correctness of the annotations or reproduce selection of the
original questions without the authorized benchmark inputs.

## Fresh preparation and annotation

First run the primary analysis and item-signature stability stage using the
authorized response matrices and question sources described in `DATA.md`.
Then prepare the blinded packet:

```bash
export FAMILY_DIF_PROJECT_ROOT="$PWD/../audit-data"
family-dif-content-audit prepare \
  --signature-input "$FAMILY_DIF_PROJECT_ROOT/outputs/v3_item_dif_signature_stability/item_family_signatures.csv" \
  --question-file "$FAMILY_DIF_PROJECT_ROOT/inputs/aligned_questions.csv" \
  --output-dir "$FAMILY_DIF_PROJECT_ROOT/outputs/v3_blinded_content_annotation" \
  --schema configs/blinded_content_annotation_schema.json
```

Preparation selects 50 exact-cell matched pairs per benchmark and removes
benchmark, experimental-group, family, and DIF labels from the packet shown
to the annotators. The private unblinding key remains in the local output
directory. The complete rubric is in `configs/blinded_content_rubric.md`;
the structured response schema is also included in `configs/`.

The optional aligned question file is CSV or JSONL with the columns
`benchmark`, `dataset_item_index`, and `question_text`. Benchmark names and
indices must match the signature file exactly; each text should include the
question and answer choices presented to the annotator, without leaking the
correct answer or hidden experimental labels. The loader verifies coverage
and unique item identifiers. If this option is omitted, the legacy local
question sources (`external_cache/mmlu_pro_test.parquet` and the aligned
`outputs/routereval_raw_global/all_benchmarks_dataset.json`) are used.

Annotation uses a provider-neutral executable supplied by the reproducer:

```bash
family-dif-content-audit annotate \
  --output-dir "$FAMILY_DIF_PROJECT_ROOT/outputs/v3_blinded_content_annotation" \
  --schema configs/blinded_content_annotation_schema.json \
  --backend-command 'python my_annotation_backend.py' \
  --workers 4 --batch-size 20
```

The default annotators remain `gpt-5.5` and `gpt-5.4`, with low reasoning effort.
The scientific prompt, schema, sampling seeds, batch size, second-annotator
item permutation, and subsequent statistical analysis are preserved. The
released transport adapter is a portability interface; it is not a claim that
all provider transports yield identical annotations. No remote model calls are
made by installation, tests, or offline replay. Fresh annotation may incur
provider charges and requires the appropriate right to transmit the questions.

### Backend interface

For each batch, the executable receives one JSON object on standard input:

```json
{
  "model": "gpt-5.5",
  "reasoning_effort": "low",
  "prompt": "The complete frozen rubric followed by the blinded item batch",
  "response_schema": {"type": "object", "properties": {"annotations": {"type": "array"}}}
}
```

The example above abbreviates the schema; the real request carries the entire
released schema. The backend should send the specified prompt and schema to
its chosen provider and write exactly one JSON response to standard output:

```json
{"annotations": [{"blinded_id": "item_0000", "...": "all required schema fields"}]}
```

Logs belong on standard error. Return a nonzero exit code on failure. The
command is parsed into arguments and executed without a shell. The runner
checks complete, unique item coverage and valid label values, stores the
validated response, and resumes from valid cached batch outputs. Use a fresh
output directory if changing the annotator models, prompt, or schema.

Finally, analyze the resulting local records:

```bash
family-dif-content-audit analyze \
  --signature-input "$FAMILY_DIF_PROJECT_ROOT/outputs/v3_item_dif_signature_stability/item_family_signatures.csv" \
  --output-dir "$FAMILY_DIF_PROJECT_ROOT/outputs/v3_blinded_content_annotation"
```

All fresh packets, responses, rationales, provider logs, and unblinding keys
are local execution material and are excluded from the anonymous public tree.

## Exporting anonymous labels from an authorized rerun

The exporter uses an explicit public-column allowlist, remaps item/pair/cell
identifiers, and retains row order for Monte Carlo replay:

```bash
python -m family_dif_benchmark_audit.interpretation.content_replay export \
  --annotations "$FAMILY_DIF_PROJECT_ROOT/outputs/v3_blinded_content_annotation/annotations_with_unblinded_key.csv" \
  --signatures "$FAMILY_DIF_PROJECT_ROOT/outputs/v3_item_dif_signature_stability/item_family_signatures.csv" \
  --output "$FAMILY_DIF_PROJECT_ROOT/outputs/public_content_labels.csv"
```

Do not replace the released frozen labels with a fresh annotation run when
checking the published results. Fresh annotation is a replication, whose
agreement with the original results must be measured and reported.
