"""Build the unfiltered 50,265 x 8,577 RouterEval response matrix.

The implementation reproduces the block layout and model-name deduplication in
``Benchmark_Analyses/codes/data_processing.ipynb``. It deliberately runs before
the low-standard-deviation and 2%/98% item filters.

Output orientation is questions x models. Values are float32: -1 missing,
0 incorrect, 1 correct, with fractional scores preserved for benchmarks that
use them. The matrix is written through a memory map to keep RAM usage modest.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_FILES = (
    "leaderboard_old.pkl",
    "leaderboard_new.pkl",
    "leaderboard_mmlu_pro.pkl",
)
EXPECTED_QUESTIONS = 50265
EXPECTED_MODELS = 8577


def normalize_model_name(name: Any) -> str:
    """Match the original preprocessing notebook exactly."""

    normalized = str(name).replace("open-llm-leaderboard-old/", "")
    if normalized.startswith("details_"):
        normalized = normalized[len("details_") :]
    if normalized.endswith("-details"):
        normalized = normalized[: -len("-details")]
    return normalized


def load_pickle(path: Path) -> dict[str, Any]:
    """Load a trusted upstream RouterEval pickle.

    Python pickle can execute code while loading; callers must verify provenance
    and must never pass an untrusted or user-supplied file.
    """
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or "model" not in value or "data" not in value:
        raise ValueError(f"Unexpected RouterEval pickle structure: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_item_runs(prompts_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    current_name: str | None = None
    current_start = 0

    with prompts_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"item_id", "dataset", "dataset_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{prompts_path} is missing columns: {sorted(missing)}")

        for expected_id, row in enumerate(reader):
            item_id = int(row["item_id"])
            if item_id != expected_id:
                raise ValueError(
                    f"Prompts are not dense/in order at row {expected_id}: {item_id}"
                )
            dataset_name = row["dataset_name"]
            item_rows.append(
                {
                    "global_item_id": item_id,
                    "dataset": row["dataset"],
                    "dataset_name": dataset_name,
                }
            )
            if current_name is None:
                current_name = dataset_name
                current_start = item_id
            elif dataset_name != current_name:
                runs.append(
                    {
                        "dataset_name": current_name,
                        "start": current_start,
                        "stop": item_id,
                        "n_items": item_id - current_start,
                    }
                )
                current_name = dataset_name
                current_start = item_id

    if len(item_rows) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS:,} question rows, found {len(item_rows):,}"
        )
    assert current_name is not None
    runs.append(
        {
            "dataset_name": current_name,
            "start": current_start,
            "stop": len(item_rows),
            "n_items": len(item_rows) - current_start,
        }
    )
    if len({run["dataset_name"] for run in runs}) != len(runs):
        raise ValueError("A dataset_name appears in multiple non-contiguous prompt runs")
    return runs, item_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_matrix(matrix: np.ndarray, chunk_rows: int = 256) -> dict[str, Any]:
    missing = 0
    zeros = 0
    ones = 0
    other = 0
    nonfinite = 0
    observed_by_model = np.zeros(matrix.shape[1], dtype=np.int64)
    row_observed_min = matrix.shape[1]
    row_observed_max = 0

    for start in range(0, matrix.shape[0], chunk_rows):
        block = np.asarray(matrix[start : start + chunk_rows])
        finite = np.isfinite(block)
        is_missing = block == -1
        is_zero = block == 0
        is_one = block == 1
        is_observed = finite & ~is_missing

        missing += int(is_missing.sum())
        zeros += int(is_zero.sum())
        ones += int(is_one.sum())
        other += int((finite & ~(is_missing | is_zero | is_one)).sum())
        nonfinite += int((~finite).sum())
        observed_by_model += is_observed.sum(axis=0)
        row_counts = is_observed.sum(axis=1)
        row_observed_min = min(row_observed_min, int(row_counts.min()))
        row_observed_max = max(row_observed_max, int(row_counts.max()))

    return {
        "shape_questions_by_models": [int(x) for x in matrix.shape],
        "dtype": str(matrix.dtype),
        "total_cells": int(matrix.size),
        "missing_minus_one_cells": missing,
        "zero_cells": zeros,
        "one_cells": ones,
        "fractional_or_other_finite_cells": other,
        "nonfinite_cells": nonfinite,
        "observed_cells": zeros + ones + other,
        "observed_responses_per_question_min": row_observed_min,
        "observed_responses_per_question_max": row_observed_max,
        "observed_questions_per_model_min": int(observed_by_model.min()),
        "observed_questions_per_model_max": int(observed_by_model.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {name: args.score_dir / name for name in SOURCE_FILES}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    runs, item_rows = read_item_runs(args.prompts)
    run_by_name = {run["dataset_name"]: run for run in runs}

    source_models: dict[str, list[str]] = {}
    global_models: OrderedDict[str, None] = OrderedDict()
    source_dataset_names: dict[str, list[str]] = {}

    # First pass: recover the exact global normalized model order.
    for source_name in SOURCE_FILES:
        source = load_pickle(source_paths[source_name])
        models = [normalize_model_name(value) for value in source["model"]]
        if len(set(models)) != len(models):
            raise ValueError(f"Normalized model names are duplicated within {source_name}")
        source_models[source_name] = models
        source_dataset_names[source_name] = list(source["data"].keys())
        for model in models:
            global_models.setdefault(model, None)
        del source
        gc.collect()

    if len(global_models) != EXPECTED_MODELS:
        raise ValueError(
            f"Expected {EXPECTED_MODELS:,} normalized models, found {len(global_models):,}"
        )
    global_model_ids = list(global_models.keys())
    global_model_index = {model: idx for idx, model in enumerate(global_model_ids)}

    all_source_datasets = {
        dataset
        for datasets in source_dataset_names.values()
        for dataset in datasets
    }
    if all_source_datasets != set(run_by_name):
        raise ValueError(
            "Source/prompt dataset mismatch: "
            f"only in source={sorted(all_source_datasets - set(run_by_name))}, "
            f"only in prompts={sorted(set(run_by_name) - all_source_datasets)}"
        )

    final_matrix = args.out_dir / "routereval_raw_50265x8577_q_by_m.npy"
    partial_matrix = args.out_dir / "routereval_raw_50265x8577_q_by_m.partial.npy"
    if final_matrix.exists() or partial_matrix.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing matrix: {final_matrix} or {partial_matrix}"
        )

    matrix = np.lib.format.open_memmap(
        partial_matrix,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_QUESTIONS, EXPECTED_MODELS),
    )
    matrix[:] = -1.0
    matrix.flush()

    dataset_manifest: list[dict[str, Any]] = []
    filled_rows = np.zeros(EXPECTED_QUESTIONS, dtype=bool)

    # Second pass: write each benchmark block into the merged global columns.
    for source_name in SOURCE_FILES:
        source = load_pickle(source_paths[source_name])
        models = source_models[source_name]
        model_columns = np.asarray(
            [global_model_index[model] for model in models], dtype=np.int32
        )

        for dataset_name, payload in source["data"].items():
            run = run_by_name[dataset_name]
            start = int(run["start"])
            stop = int(run["stop"])
            block = np.asarray(payload["correctness"], dtype=np.float32)
            expected_shape = (stop - start, len(models))
            if block.shape != expected_shape:
                raise ValueError(
                    f"{dataset_name}: expected {expected_shape}, got {block.shape}"
                )
            if not np.all(np.isfinite(block)):
                raise ValueError(f"{dataset_name}: non-finite source response values")
            if filled_rows[start:stop].any():
                raise ValueError(f"Rows for {dataset_name} were already filled")

            matrix[start:stop, model_columns] = block
            filled_rows[start:stop] = True
            dataset_manifest.append(
                {
                    "dataset_name": dataset_name,
                    "source_pickle": source_name,
                    "row_start": start,
                    "row_stop_exclusive": stop,
                    "n_questions": stop - start,
                    "n_source_models": len(models),
                    "global_model_index_min": int(model_columns.min()),
                    "global_model_index_max": int(model_columns.max()),
                }
            )

        matrix.flush()
        del source
        gc.collect()

    if not filled_rows.all():
        missing_rows = np.flatnonzero(~filled_rows)
        raise ValueError(f"Unfilled matrix rows: {missing_rows[:20].tolist()}")

    stats = summarize_matrix(matrix)
    matrix.flush()
    del matrix
    gc.collect()
    partial_matrix.replace(final_matrix)

    model_rows = [
        {"global_model_index": idx, "model_id": model}
        for idx, model in enumerate(global_model_ids)
    ]
    write_csv(
        args.out_dir / "routereval_raw_8577_models.csv",
        ["global_model_index", "model_id"],
        model_rows,
    )
    write_csv(
        args.out_dir / "routereval_raw_50265_items.csv",
        ["global_item_id", "dataset", "dataset_name"],
        item_rows,
    )

    manifest = {
        "source": "RouterEval official leaderboard_score.zip",
        "description": "Unfiltered matrix after cross-source model-name deduplication",
        "matrix_file": final_matrix.name,
        "orientation": "questions_by_models",
        "encoding": {"-1": "missing", "0": "incorrect", "1": "correct"},
        "model_file": "routereval_raw_8577_models.csv",
        "item_file": "routereval_raw_50265_items.csv",
        "full_prompt_metadata": str(args.prompts.resolve()),
        "source_model_counts": {
            source_name: len(source_models[source_name])
            for source_name in SOURCE_FILES
        },
        "source_model_count_before_cross_source_deduplication": int(
            sum(len(source_models[source_name]) for source_name in SOURCE_FILES)
        ),
        "normalized_unique_model_count": len(global_model_ids),
        "matrix_stats": stats,
        "source_sha256": {
            source_name: sha256(path) for source_name, path in source_paths.items()
        },
        "prompts_sha256": sha256(args.prompts),
        "matrix_sha256": sha256(final_matrix),
        "dataset_blocks": sorted(dataset_manifest, key=lambda row: row["row_start"]),
    }
    manifest_path = args.out_dir / "routereval_raw_50265x8577_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "dataset_blocks"}, indent=2))


if __name__ == "__main__":
    main()
