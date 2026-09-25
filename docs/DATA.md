# Data preparation, alignment, and redistribution

The anonymous repository contains code, aggregate reference results, and
anonymous numeric content-audit labels. It does not contain raw benchmark
questions, the full response matrices, or original model identifiers. The
preparation tool retrieves upstream inputs into a separate local directory.

## 1. Obtain the paper inputs

Install the pinned spectral environment as described in the root README, then
run from the repository directory:

```sh
python tools/prepare_inputs.py --data-root ../audit-data --download
```

The two upstream inputs are:

- [RouterEval](https://huggingface.co/datasets/linggm/RouterEval):
  `leaderboard_score.zip`, containing the old leaderboard, new leaderboard,
  and MMLU-Pro score exports.
- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro): the test parquet
  used to obtain the MMLU-Pro item-group metadata.

These links identify third-party data sources, not the artifact's authors.
Their existing authorship, licenses, and terms continue to apply. The MMLU-Pro
upstream parquet includes question content even though the numerical audit
only needs its item-group metadata; keep the local source file out of the
anonymous public package.

The input preparation tool verifies the complete archive and parquet hashes,
then verifies each extracted score file before loading it. It extracts only
the expected archive members into fixed filenames. It constructs an item
layout from the public, text-free `configs/routereval_item_layout.json`, runs
the matrix builders, and verifies the expected global matrix hash. It also
checks MMLU-Pro response-array and ordered-model-label hashes independently
of the NPZ container representation, including when reusing prepared inputs.

The authoritative checksum contract is `configs/upstream_inputs.json`:

| Input | SHA-256 |
|---|---|
| RouterEval score ZIP | `21e0429f549898afdf229dac9d7f77c49b5ff8061340d8b86ed6031a38a54077` |
| MMLU-Pro test parquet | `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8` |
| Prepared global response matrix | `22d224b88d41395bf9e58abe4ca779d3e1c4473b29bcc6426243cc616ffa219e` |

Individual score-file hashes and revision-pinned download URLs are also in
that configuration. Downloads use immutable upstream commit IDs, not `main`.
A checksum mismatch is an explicit failure, not permission to skip verification.
Record any intentionally changed input as a new analysis. Upstream access or
availability can still change independently of this repository.

If the inputs are already available locally:

```sh
python tools/prepare_inputs.py --data-root ../audit-data \
  --score-zip ../downloads/leaderboard_score.zip \
  --mmlu-pro-parquet ../downloads/mmlu_pro_test.parquet
```

The tool does not redownload existing files with `--download`; it verifies
them. A stale or partial input therefore needs to be inspected before retrying.
Python pickle loading can execute code: only use trusted upstream exports with
the frozen provenance checks. A checksum is an identity check against that
contract, not a general-purpose malware scan.

## 2. Prepared directory contract

Paths below are relative to the chosen data root, not the public checkout:

```text
audit-data/
  input_verification.json
  external_sources/
    leaderboard_score.zip
    leaderboard_score/
      leaderboard_old.pkl
      leaderboard_new.pkl
      leaderboard_mmlu_pro.pkl
    item_layout.csv
  external_cache/
    mmlu_pro_test.parquet
  outputs/
    routereval_raw_global/
      all_response_matrix_q_by_m.npy
      routereval_raw_50265_items.csv
      routereval_raw_8577_models.csv
    routereval_raw_matrices/
      mmlu_pro_response_matrix.npz
```

The global matrix has 50,265 rows and 8,577 model columns. It includes missing
response indicators; the benchmark loaders retain complete models for each
benchmark and then check binary 0/1 data. They do not impute missing answers.
The global binary file is approximately 1.7 GB in its stored representation.
The preparation tool may create additional mapping, metadata, and benchmark
matrix files as well as those shown above.

The global item CSV supplies row order, benchmark membership, and native item
groups. The model CSV supplies the dense ordered model-column indices and
original model IDs. The dedicated MMLU-Pro NPZ contains
`responses_q_by_m` and `model_ids`; rows align with the MMLU-Pro test parquet.
These are the paper-input formats, not the simpler maintainer-interface NPZ
schema. See [MAINTAINER_GUIDE.md](MAINTAINER_GUIDE.md) for the latter.

Never independently sort matrix rows, model metadata, or item metadata. Every
response entry is interpreted by its aligned row and column. Changing ordering
can change deterministic tie-breaking and random selections even when the
underlying set of records appears unchanged.

`input_verification.json` records the verified source status and hashes of the
prepared inputs using relative paths. The per-benchmark frozen protocols under
`results/primary/per_benchmark/` retain the input hashes used for the published
run. Regenerated container bytes can depend on serialization versions; compare
matrix values, metadata order, and the recorded software before treating a
container-hash difference as a scientific difference. Do not bypass a changed
upstream pickle or global-matrix checksum to reach the fitting stage.

## 3. Point the analysis at the prepared data

The staged workflow accepts the data root directly:

```sh
python tools/reproduce.py --data-root ../audit-data --stage primary
```

Manual module or console commands use this environment variable, set before
launching Python:

```sh
export FAMILY_DIF_PROJECT_ROOT="$PWD/../audit-data"
family-dif-ranking --benchmark mmlu_pro
```

The variable selects the response-data and generated-output root. It does not
move the installed code. A `--primary-root` argument on diagnostic commands
selects existing primary output directories; it does not replace the raw data
root. Instructions and stage dependencies are in [REPRODUCE.md](REPRODUCE.md).

## 4. Anonymous labels support complete statistical replay

`results/content_audit/labels.csv` contains 1,000 annotation records: 500 items
in 250 matched pairs, with two annotations per item. Published columns include
new item/pair/cell identifiers, the two annotator/model names, benchmark,
experimental group, categorical advantaged-family labels, and the binary and
ordinal responses needed by the statistical tests.

The table excludes original item indices, original item-group names, question
text, answer choices, rationales, fitted item-effect vectors, model-owner
identities, and unblinding-to-original-item mappings. It allows recomputation
of all six released annotation-analysis tables without access to raw text or
a model service. It does not independently establish that the content labels
are semantically correct or reconstruct the original sampling from questions.

```sh
family-dif-content-replay replay \
  --labels results/content_audit/labels.csv \
  --reference-dir results/content_audit \
  --output-dir ../audit-data/outputs/content_audit_replay
```

This is statistical replay of frozen judgments. The question data and backend
requirements for fresh judgments are separate.

## 5. Optional fresh content annotation

After primary fitting and item-signature reconstruction, supply an authorized
CSV or JSONL question source with columns:

```text
benchmark,dataset_item_index,question_text
```

Benchmark labels and item indices must match the reconstructed
`item_family_signatures.csv`. Question text should contain the question and
answer options presented to annotators, without the answer key, DIF statistics,
family labels, or experimental condition. The packet preparation step checks
identifier uniqueness and coverage. Keep the private aligned input and
unblinding key outside the public repository.

The numeric preparation command does not obtain every benchmark's question
text. Fresh annotation requires separate access to those sources, the right to
transmit them to the chosen provider, a backend executable, and any associated
provider budget. The provider-neutral transport contract and exact commands
are in [CONTENT_AUDIT.md](CONTENT_AUDIT.md). Hosted model outputs may change, so
a fresh annotation run is a new replication rather than bitwise replay.

## 6. Public versus local files

| Included in the anonymous artifact | Kept in the external data/output directory |
|---|---|
| Analysis implementation and frozen protocols | Original item-by-model response matrices |
| Aggregate benchmark, fold, control, and bootstrap statistics | Original model identifiers and owner mappings |
| Anonymous numeric annotation labels | Questions, choices, answer keys, rationales |
| Annotation rubric and structured schema | Raw packets, unblinding keys, provider outputs/logs |
| Figure generators and numeric figure checks | Per-model scores and item-level fitting intermediates |
| Public input hashes and public-file manifests | Credentials, caches, environment files, local paths |

Aggregate random-control rows and anonymous content labels are intentionally
public; they are not original model or question records. Raw primary runs will
recreate identifying model metadata locally because the analysis needs it.
The release audit scans the curated public tree; do not assume it anonymizes
an arbitrary data directory, commit history, Git remote, or hosting account.

Files in this repository do not change third-party redistribution rights.
The [MIT License](../LICENSE) covers the authors' original contributions as
specified in the [README](../README.md#license); it does not relicense the
external benchmark inputs or dependencies.
Local access to a dataset does not imply permission to republish its text or
raw records. Use the neutral public artifact and keep the prepared input tree
separate when preparing an anonymous submission.
