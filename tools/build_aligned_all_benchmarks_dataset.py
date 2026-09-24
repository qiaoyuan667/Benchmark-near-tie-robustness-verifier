#!/usr/bin/env python3
"""Reorder all_benchmarks_dataset.json to match the response-matrix rows.

The RouterEval matrix uses ``old_item_id`` / ``global_item_id`` as its row
index, while the existing all_benchmarks JSON uses ``new_id``.  The project's
authoritative fine-grained mapping is a complete bijection between the two.
This script applies that permutation, resets each output ``id`` to the matrix
row index, and preserves every other question field unchanged.

The 1.7 GB response matrix is opened with NumPy memory mapping solely to
validate its shape; its response cells are never loaded into RAM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parents[1]
FEATURE_DIR = Path(os.environ.get("FAMILY_DIF_SOURCE_DATA_ROOT", HERE / "external_sources"))

DEFAULT_MATRIX = (
    HERE / "outputs/routereval_raw_global/all_response_matrix_q_by_m.npy"
)
DEFAULT_ITEMS = (
    HERE / "outputs/routereval_raw_global/routereval_raw_50265_items.csv"
)
DEFAULT_MAPPING = FEATURE_DIR / "old_item_id_to_new_id_fine_grained_mapping.csv"
DEFAULT_SOURCE = FEATURE_DIR / "all_benchmarks_dataset.json"
DEFAULT_OUTPUT = (
    HERE / "outputs/routereval_raw_global/all_benchmarks_dataset.json"
)

EXPECTED_FIELDS = ["id", "source", "subject", "level", "problem", "solution"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_true(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_matrix_row_metadata(path: Path, expected_rows: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"global_item_id", "dataset", "dataset_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)

    if len(rows) != expected_rows:
        raise ValueError(
            f"Matrix metadata has {len(rows):,} rows; expected {expected_rows:,}"
        )
    for expected_id, row in enumerate(rows):
        if int(row["global_item_id"]) != expected_id:
            raise ValueError(
                f"Metadata row {expected_id} has global_item_id="
                f"{row['global_item_id']}"
            )
    return rows


def load_source(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or len(rows) != expected_rows:
        found = len(rows) if isinstance(rows, list) else type(rows).__name__
        raise ValueError(
            f"Source JSON must be a {expected_rows:,}-item list; found {found}"
        )
    for expected_id, row in enumerate(rows):
        if list(row) != EXPECTED_FIELDS:
            raise ValueError(
                f"Source item {expected_id} has fields {list(row)}; "
                f"expected {EXPECTED_FIELDS}"
            )
        if row["id"] != expected_id:
            raise ValueError(
                f"Source item at position {expected_id} has id={row['id']}"
            )
    return rows


def load_permutation(
    path: Path,
    metadata: list[dict[str, str]],
    source: list[dict[str, Any]],
) -> list[int]:
    n_rows = len(metadata)
    old_to_new: list[int | None] = [None] * n_rows
    seen_new: set[int] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "old_item_id",
            "old_row_index",
            "old_dataset",
            "old_dataset_name",
            "new_id",
            "new_source",
            "new_problem",
            "matched",
            "benchmark_matched",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        row_count = 0
        for mapping_row in reader:
            row_count += 1
            old_id = int(mapping_row["old_item_id"])
            new_id = int(mapping_row["new_id"])
            if not 0 <= old_id < n_rows or not 0 <= new_id < n_rows:
                raise ValueError(f"Out-of-range mapping {old_id} -> {new_id}")
            if old_to_new[old_id] is not None:
                raise ValueError(f"Duplicate old_item_id: {old_id}")
            if new_id in seen_new:
                raise ValueError(f"Duplicate new_id: {new_id}")
            if int(mapping_row["old_row_index"]) != old_id:
                raise ValueError(f"old_row_index mismatch for old_item_id {old_id}")
            if not is_true(mapping_row["matched"]):
                raise ValueError(f"Unmatched item at matrix row {old_id}")
            if not is_true(mapping_row["benchmark_matched"]):
                raise ValueError(f"Benchmark mismatch at matrix row {old_id}")

            item_meta = metadata[old_id]
            if mapping_row["old_dataset"] != item_meta["dataset"]:
                raise ValueError(f"Dataset mismatch at matrix row {old_id}")
            if mapping_row["old_dataset_name"] != item_meta["dataset_name"]:
                raise ValueError(f"Dataset-name mismatch at matrix row {old_id}")

            source_item = source[new_id]
            if mapping_row["new_source"] != str(source_item["source"]):
                raise ValueError(f"Source mismatch for all_benchmarks ID {new_id}")
            if mapping_row["new_problem"] != str(source_item["problem"]):
                raise ValueError(f"Problem mismatch for all_benchmarks ID {new_id}")

            old_to_new[old_id] = new_id
            seen_new.add(new_id)

    if row_count != n_rows:
        raise ValueError(f"Mapping has {row_count:,} rows; expected {n_rows:,}")
    if any(value is None for value in old_to_new):
        missing_old = [i for i, value in enumerate(old_to_new) if value is None]
        raise ValueError(f"Missing old_item_id values: {missing_old[:20]}")
    if seen_new != set(range(n_rows)):
        missing_new = sorted(set(range(n_rows)) - seen_new)
        raise ValueError(f"Missing new_id values: {missing_new[:20]}")

    return [int(value) for value in old_to_new]


def write_aligned_json(
    path: Path,
    source: list[dict[str, Any]],
    old_to_new: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("[\n")
            for matrix_row, source_id in enumerate(old_to_new):
                item = dict(source[source_id])
                item["id"] = matrix_row
                encoded = json.dumps(item, ensure_ascii=False, indent=2)
                handle.write("  " + encoded.replace("\n", "\n  "))
                handle.write(",\n" if matrix_row + 1 < len(old_to_new) else "\n")
            handle.write("]\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_readback(
    path: Path,
    source: list[dict[str, Any]],
    old_to_new: list[int],
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        aligned = json.load(handle)
    if len(aligned) != len(old_to_new):
        raise ValueError("Readback row count does not match the permutation")
    for matrix_row, (item, source_id) in enumerate(zip(aligned, old_to_new)):
        expected = dict(source[source_id])
        expected["id"] = matrix_row
        if item != expected:
            raise ValueError(f"Readback mismatch at matrix row {matrix_row}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    matrix = np.load(args.matrix, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D response matrix, got {matrix.shape}")
    n_rows = int(matrix.shape[0])

    metadata = load_matrix_row_metadata(args.items, n_rows)
    source = load_source(args.source, n_rows)
    old_to_new = load_permutation(args.mapping, metadata, source)
    write_aligned_json(args.output, source, old_to_new)
    verify_readback(args.output, source, old_to_new)

    dataset_counts = Counter(row["dataset"] for row in metadata)
    source_counts = Counter(source[new_id]["source"] for new_id in old_to_new)
    report = {
        "status": "valid",
        "output": str(args.output),
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "items": n_rows,
        "id_semantics": "output id equals zero-based matrix row",
        "other_fields": "copied unchanged from mapped all_benchmarks item",
        "dataset_counts": dict(dataset_counts),
        "fine_grained_source_count": len(source_counts),
        "source_sha256": sha256_file(args.source),
        "mapping_sha256": sha256_file(args.mapping),
        "output_sha256": sha256_file(args.output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
