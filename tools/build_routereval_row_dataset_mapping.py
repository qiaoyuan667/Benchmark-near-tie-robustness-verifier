#!/usr/bin/env python3
"""Build row-to-dataset metadata for the RouterEval response matrix.

The row order is read from ``routereval_raw_50265_items.csv`` and checked
against ``all_response_matrix_q_by_m.npy``.  The main output is deliberately
simple: a JSON object whose string key is a zero-based matrix row and whose
value is one of the 12 normalized dataset names.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/routereval_raw_global"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/routereval_raw_global/row_dataset_mapping_for_irt"
        ),
    )
    args = parser.parse_args()

    matrix_path = args.input_dir / "all_response_matrix_q_by_m.npy"
    items_path = args.input_dir / "routereval_raw_50265_items.csv"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}")

    with items_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != matrix.shape[0]:
        raise ValueError(
            f"Metadata has {len(rows)} rows but matrix has {matrix.shape[0]} rows"
        )
    for index, row in enumerate(rows):
        if int(row["global_item_id"]) != index:
            raise ValueError(
                f"Row {index} has global_item_id={row['global_item_id']}"
            )

    datasets = [row["dataset"] for row in rows]
    row_to_dataset = {str(index): dataset for index, dataset in enumerate(datasets)}

    simple_path = output_dir / "row_to_dataset.json"

    simple_path.write_text(
        json.dumps(row_to_dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(row_to_dataset):,} zero-based row mappings "
        f"for matrix shape {matrix.shape} to {simple_path}"
    )


if __name__ == "__main__":
    main()
