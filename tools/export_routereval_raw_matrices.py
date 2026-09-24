"""Export unfiltered RouterEval response blocks for Winogrande and MMLU-Pro.

This reproduces the model-name normalization used in
``Benchmark_Analyses/codes/data_processing.ipynb`` but deliberately does not
apply either item filter (low response standard deviation or extreme item
accuracy).  The input directory must contain the three official RouterEval
leaderboard score pickles:

* leaderboard_old.pkl
* leaderboard_new.pkl
* leaderboard_mmlu_pro.pkl

Each output NPZ contains the response matrix in both question-by-model and
model-by-question orientation, plus model and item identifiers.  Response
values use int8 encoding: -1 missing, 0 incorrect, 1 correct.
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


TARGETS = {
    "winogrande": {
        "pickle": "leaderboard_old.pkl",
        "dataset_name": "harness_winogrande_5",
        "expected_questions": 1267,
        "expected_models": 5000,
    },
    "mmlu_pro": {
        "pickle": "leaderboard_mmlu_pro.pkl",
        "dataset_name": "mmlu_pro",
        "expected_questions": 12032,
        "expected_models": 1823,
    },
}


def normalize_model_name(name: Any) -> str:
    """Match the normalization in the original preprocessing notebook."""

    normalized = str(name).replace("open-llm-leaderboard-old/", "")
    if normalized.startswith("details_"):
        normalized = normalized[len("details_") :]
    if normalized.endswith("-details"):
        normalized = normalized[: -len("-details")]
    return normalized


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_binary_response_matrix(
    matrix: np.ndarray,
    expected_questions: int,
    expected_models: int,
    dataset: str,
) -> np.ndarray:
    if matrix.shape != (expected_questions, expected_models):
        raise ValueError(
            f"{dataset}: expected {(expected_questions, expected_models)}, "
            f"got {matrix.shape}"
        )
    values = np.unique(matrix)
    if not np.all(np.isin(values, [-1, 0, 1])):
        raise ValueError(f"{dataset}: unexpected response values: {values.tolist()}")
    return matrix.astype(np.int8, copy=False)


def load_items(prompts_path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    rows_by_dataset = {name: [] for name in TARGETS}
    with prompts_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"item_id", "dataset", "dataset_name"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"{prompts_path} is missing columns: {sorted(missing)}")

        expected_item_id = 0
        for row in reader:
            item_id = int(row["item_id"])
            if item_id != expected_item_id:
                raise ValueError(
                    f"Prompts are not dense/in order at row {expected_item_id}: {item_id}"
                )
            expected_item_id += 1
            for name, spec in TARGETS.items():
                if row["dataset_name"] == spec["dataset_name"]:
                    rows_by_dataset[name].append(row)
                    break

    if expected_item_id != 50265:
        raise ValueError(f"Expected 50,265 raw questions, found {expected_item_id:,}")
    return fieldnames, rows_by_dataset


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def matrix_stats(matrix: np.ndarray) -> dict[str, Any]:
    valid = matrix != -1
    correct = matrix == 1
    valid_per_question = valid.sum(axis=1)
    valid_per_model = valid.sum(axis=0)
    correct_per_question = correct.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy_per_question = correct_per_question / valid_per_question
    std_per_question = np.array(
        [row[row != -1].std() if np.any(row != -1) else np.nan for row in matrix]
    )

    unique_values, counts = np.unique(matrix, return_counts=True)
    return {
        "shape_questions_by_models": [int(x) for x in matrix.shape],
        "shape_models_by_questions": [int(matrix.shape[1]), int(matrix.shape[0])],
        "value_counts": {
            str(int(value)): int(count)
            for value, count in zip(unique_values, counts)
        },
        "missing_cells": int((matrix == -1).sum()),
        "valid_cells": int(valid.sum()),
        "correct_cells": int(correct.sum()),
        "incorrect_cells": int((matrix == 0).sum()),
        "overall_accuracy": float(correct.sum() / valid.sum()),
        "valid_responses_per_question_min": int(valid_per_question.min()),
        "valid_responses_per_question_max": int(valid_per_question.max()),
        "valid_questions_per_model_min": int(valid_per_model.min()),
        "valid_questions_per_model_max": int(valid_per_model.max()),
        "question_accuracy_min": float(np.nanmin(accuracy_per_question)),
        "question_accuracy_median": float(np.nanmedian(accuracy_per_question)),
        "question_accuracy_max": float(np.nanmax(accuracy_per_question)),
        "questions_low_std_below_0_01": int(np.sum(std_per_question < 0.01)),
        "questions_accuracy_le_0_02": int(np.sum(accuracy_per_question <= 0.02)),
        "questions_accuracy_ge_0_98": int(np.sum(accuracy_per_question >= 0.98)),
        "questions_kept_by_old_two_stage_filter": int(
            np.sum(
                (std_per_question >= 0.01)
                & (accuracy_per_question > 0.02)
                & (accuracy_per_question < 0.98)
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    item_fieldnames, items_by_dataset = load_items(args.prompts)

    pickle_paths = {
        name: args.score_dir / name
        for name in (
            "leaderboard_old.pkl",
            "leaderboard_new.pkl",
            "leaderboard_mmlu_pro.pkl",
        )
    }
    for path in pickle_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    model_lists: dict[str, list[str]] = {}
    target_matrices: dict[str, np.ndarray] = {}
    target_models: dict[str, list[str]] = {}

    # Load sequentially so the large source pickles are never resident together.
    for source_name in (
        "leaderboard_old.pkl",
        "leaderboard_new.pkl",
        "leaderboard_mmlu_pro.pkl",
    ):
        source = load_pickle(pickle_paths[source_name])
        normalized_models = [normalize_model_name(x) for x in source["model"]]
        model_lists[source_name] = normalized_models

        for target_name, spec in TARGETS.items():
            if source_name != spec["pickle"]:
                continue
            raw = np.asarray(source["data"][spec["dataset_name"]]["correctness"])
            target_matrices[target_name] = validate_binary_response_matrix(
                raw,
                expected_questions=int(spec["expected_questions"]),
                expected_models=int(spec["expected_models"]),
                dataset=target_name,
            ).copy()
            target_models[target_name] = normalized_models

        del source
        gc.collect()

    global_model_order: OrderedDict[str, None] = OrderedDict()
    for source_name in (
        "leaderboard_old.pkl",
        "leaderboard_new.pkl",
        "leaderboard_mmlu_pro.pkl",
    ):
        for model in model_lists[source_name]:
            global_model_order.setdefault(model, None)

    if len(global_model_order) != 8577:
        raise ValueError(
            f"Expected 8,577 normalized global models, found {len(global_model_order):,}"
        )
    global_model_index = {
        model: idx for idx, model in enumerate(global_model_order.keys())
    }

    report: dict[str, Any] = {
        "source": "RouterEval official leaderboard_score.zip",
        "raw_global_shape_questions_by_models": [50265, 8577],
        "raw_source_model_counts_before_cross_source_deduplication": {
            source_name: len(models) for source_name, models in model_lists.items()
        },
        "raw_source_model_count_total_before_deduplication": int(
            sum(len(models) for models in model_lists.values())
        ),
        "global_normalized_unique_models": len(global_model_order),
        "source_sha256": {
            source_name: sha256(path) for source_name, path in pickle_paths.items()
        },
        "datasets": {},
    }

    for target_name, spec in TARGETS.items():
        matrix = target_matrices[target_name]
        models = target_models[target_name]
        items = items_by_dataset[target_name]
        expected_q = int(spec["expected_questions"])
        expected_m = int(spec["expected_models"])

        if len(items) != expected_q:
            raise ValueError(
                f"{target_name}: expected {expected_q:,} prompt rows, found {len(items):,}"
            )
        if len(models) != expected_m or len(set(models)) != expected_m:
            raise ValueError(
                f"{target_name}: expected {expected_m:,} unique models, "
                f"found {len(models):,} rows / {len(set(models)):,} unique"
            )

        global_item_ids = np.asarray([int(row["item_id"]) for row in items], dtype=np.int32)
        dataset_item_indices = np.arange(expected_q, dtype=np.int32)

        npz_path = args.out_dir / f"{target_name}_raw_responses.npz"
        np.savez_compressed(
            npz_path,
            responses_q_by_m=matrix,
            responses_m_by_q=matrix.T,
            model_ids=np.asarray(models, dtype=str),
            global_item_ids=global_item_ids,
            dataset_item_indices=dataset_item_indices,
            dataset_name=np.asarray(str(spec["dataset_name"])),
        )

        model_rows = [
            {
                "dataset_model_index": idx,
                "model_id": model,
                "global_merged_model_index": global_model_index[model],
            }
            for idx, model in enumerate(models)
        ]
        write_csv(
            args.out_dir / f"{target_name}_models.csv",
            ["dataset_model_index", "model_id", "global_merged_model_index"],
            model_rows,
        )
        write_csv(
            args.out_dir / f"{target_name}_items.csv",
            item_fieldnames,
            items,
        )

        stats = matrix_stats(matrix)
        stats.update(
            {
                "dataset_name": str(spec["dataset_name"]),
                "models_unique": len(set(models)),
                "items_unique": len(set(global_item_ids.tolist())),
                "global_item_id_min": int(global_item_ids.min()),
                "global_item_id_max": int(global_item_ids.max()),
                "matrix_npz": npz_path.name,
                "models_csv": f"{target_name}_models.csv",
                "items_csv": f"{target_name}_items.csv",
            }
        )
        report["datasets"][target_name] = stats

    report_path = args.out_dir / "verification_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
