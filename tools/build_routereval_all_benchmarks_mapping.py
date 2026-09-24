"""Map raw RouterEval item rows to ``all_benchmarks_dataset.json`` IDs.

This script materializes the one-to-one mappings for the two unfiltered binary
benchmarks used by the project:

* RouterEval MMLU-Pro rows 38,233..50,264 -> all_benchmarks IDs 0..12,031
* RouterEval Winogrande rows 0..1,266 -> all_benchmarks IDs 48,998..50,264

The relationship is read from the project's authoritative old-item -> new-ID
mapping rather than inferred from the numeric ranges.  The script then audits
the relationship against RouterEval item metadata, prompt metadata, and the
current all_benchmarks JSON.  It refuses to write output if coverage,
uniqueness, source identity, or content checks fail.

The large response matrix is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parents[1]
FEATURE_DIR = Path(os.environ.get("FAMILY_DIF_SOURCE_DATA_ROOT", HERE / "external_sources"))

DEFAULT_ITEMS = (
    HERE / "outputs/routereval_raw_global/routereval_raw_50265_items.csv"
)
DEFAULT_PROMPTS = FEATURE_DIR / "prompts_details.csv"
DEFAULT_OLD_TO_NEW = (
    FEATURE_DIR / "old_item_id_to_new_id_fine_grained_mapping.csv"
)
DEFAULT_ALL_BENCHMARKS = FEATURE_DIR / "all_benchmarks_dataset.json"
DEFAULT_OUTPUT_DIR = HERE / "outputs/routereval_raw_global"

DATASETS: dict[str, dict[str, Any]] = {
    "mmlu_pro": {
        "dataset": "mmlu_pro",
        "dataset_name": "mmlu_pro",
        "all_source": "MMLU_Pro",
        "global_start": 38233,
        "global_stop": 50265,
        "all_start": 0,
        "all_stop": 12032,
    },
    "winogrande": {
        "dataset": "winogrande",
        "dataset_name": "harness_winogrande_5",
        "all_source": "winogrande",
        "global_start": 0,
        "global_stop": 1267,
        "all_start": 48998,
        "all_stop": 50265,
    },
}

OUTPUT_FIELDS = [
    "dataset",
    "routereval_global_item_id",
    "routereval_dataset_item_index",
    "routereval_dataset_name",
    "all_benchmarks_id",
    "all_benchmarks_source",
    "match_method",
    "mapped_problem_exact_all_benchmarks",
    "direct_prompt_content_status",
    "direct_prompt_similarity",
    "problem_sha256",
    "problem_preview",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_text(value: str) -> str:
    """Normalize presentation-only differences for direct content auditing."""

    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"(?:^|\n)Answer:\s*\Z", "", value).strip()
    return re.sub(r"\s+", " ", value).strip()


def problem_preview(value: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def load_router_items(path: Path) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"global_item_id", "dataset", "dataset_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for expected_id, row in enumerate(reader):
            item_id = int(row["global_item_id"])
            if item_id != expected_id:
                raise ValueError(
                    f"{path}: expected dense global_item_id={expected_id}, got {item_id}"
                )
            result[item_id] = row
    if len(result) != 50265:
        raise ValueError(f"Expected 50,265 RouterEval item rows, found {len(result):,}")
    return result


def load_target_prompts(path: Path) -> dict[int, dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    target_names = {str(spec["dataset_name"]) for spec in DATASETS.values()}
    result: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"item_id", "dataset", "dataset_name", "question", "options"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        row_count = 0
        for expected_id, row in enumerate(reader):
            item_id = int(row["item_id"])
            if item_id != expected_id:
                raise ValueError(
                    f"{path}: expected dense item_id={expected_id}, got {item_id}"
                )
            row_count += 1
            if row["dataset_name"] in target_names:
                result[item_id] = row
    if row_count != 50265:
        raise ValueError(f"Expected 50,265 prompt rows, found {row_count:,}")
    expected_targets = sum(
        int(spec["global_stop"]) - int(spec["global_start"])
        for spec in DATASETS.values()
    )
    if len(result) != expected_targets:
        raise ValueError(
            f"Expected {expected_targets:,} target prompt rows, found {len(result):,}"
        )
    return result


def load_authoritative_mapping(path: Path) -> dict[int, dict[str, str]]:
    wanted_ids = {
        item_id
        for spec in DATASETS.values()
        for item_id in range(int(spec["global_start"]), int(spec["global_stop"]))
    }
    result: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "old_item_id",
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
        for row in reader:
            old_id = int(row["old_item_id"])
            if old_id not in wanted_ids:
                continue
            if old_id in result:
                raise ValueError(f"Duplicate old_item_id in authoritative map: {old_id}")
            result[old_id] = row
    if set(result) != wanted_ids:
        missing_ids = sorted(wanted_ids - set(result))
        raise ValueError(f"Authoritative map is missing target IDs: {missing_ids[:20]}")
    return result


def extract_mmlu_target(row: dict[str, str]) -> tuple[str, str]:
    questions = json.loads(row["question"])
    option_groups = json.loads(row["options"])
    if not questions or not option_groups:
        raise ValueError(f"MMLU-Pro prompt {row['item_id']} has no target question/options")
    question = str(questions[-1])
    option_lines: list[str] = []
    for option in option_groups[-1]:
        text = re.sub(r"\nAnswer:\s*\Z", "", str(option["text"]))
        option_lines.append(f"{option['label']}. {text}")
    options = "\n".join(option_lines)
    return question + "\n" + options, options


def verify_mmlu_prompt(row: dict[str, str], problem: str) -> tuple[str, float]:
    target, options = extract_mmlu_target(row)
    target_normalized = normalized_text(target)
    problem_normalized = normalized_text(problem)
    if target_normalized == problem_normalized:
        return "normalized_exact_question_and_options", 1.0

    # Ten known source-format variants have additions/omissions in the stem,
    # while the complete labeled option block remains exactly the same.
    options_normalized = normalized_text(options)
    similarity = SequenceMatcher(
        None, target_normalized, problem_normalized, autojunk=False
    ).ratio()
    if problem_normalized.endswith(options_normalized) and similarity >= 0.75:
        return "source_variant_same_exact_options", float(similarity)
    raise ValueError(
        f"MMLU-Pro direct content check failed for RouterEval item {row['item_id']}"
    )


def parse_winogrande_problem(problem: str) -> tuple[str, str, str]:
    prefix = "Choose the right option for a given sentence:\n"
    canonical = unicodedata.normalize("NFKC", problem).replace("\r\n", "\n")
    if not canonical.startswith(prefix):
        raise ValueError("Unexpected all_benchmarks Winogrande problem template")
    sentence, option_1, option_2 = canonical[len(prefix) :].rsplit("\n", 2)
    if not option_1.startswith("Option 1: ") or not option_2.startswith("Option 2: "):
        raise ValueError("Unexpected all_benchmarks Winogrande option template")
    return (
        sentence,
        option_1[len("Option 1: ") :],
        option_2[len("Option 2: ") :],
    )


def verify_winogrande_prompt(row: dict[str, str], problem: str) -> tuple[str, float]:
    questions = json.loads(row["question"])
    if not questions:
        raise ValueError(f"Winogrande prompt {row['item_id']} has no question")
    target_prefix = normalized_text(str(questions[-1]).split("\n\n")[-1])
    sentence, option_1, option_2 = parse_winogrande_problem(problem)
    before_blank = sentence.split("_", 1)[0]
    positional_candidates = {
        normalized_text(before_blank + option_1),
        normalized_text(before_blank + option_2),
    }
    if target_prefix in positional_candidates:
        return "exact_blank_prefix_and_choice", 1.0

    # Two records in all_benchmarks reconstruct the underscore at a repeated
    # entity's other occurrence.  The prompt target is still an exact prefix of
    # the same sentence after inserting one of the two supplied options.
    filled_candidates = {
        normalized_text(sentence.replace("_", option_1)),
        normalized_text(sentence.replace("_", option_2)),
    }
    if any(candidate.startswith(target_prefix) for candidate in filled_candidates):
        return "placeholder_position_variant_same_sentence", 1.0
    raise ValueError(
        f"Winogrande direct content check failed for RouterEval item {row['item_id']}"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_report(audit: dict[str, Any]) -> str:
    lines = [
        "# RouterEval → all_benchmarks ID 映射报告",
        "",
        "本报告只处理未过滤的 MMLU-Pro 与 Winogrande。映射由项目已有的权威 "
        "`old_item_id → new_id` 表导出，并分别用 RouterEval 元数据、当前 "
        "`all_benchmarks_dataset.json` 和目标题文本复核。未加载 1.7 GB 响应矩阵。",
        "",
        "| 数据集 | RouterEval 行 | all_benchmarks ID | 数量 | 一一映射 | 映射表题目与 all_benchmarks 完全一致 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("mmlu_pro", "winogrande"):
        row = audit["datasets"][name]
        lines.append(
            f"| {name} | {row['routereval_global_range']} | "
            f"{row['all_benchmarks_id_range']} | {row['n_items']:,} | "
            f"{row['one_to_one']} | {row['mapped_problem_exact_count']:,}/{row['n_items']:,} |"
        )

    lines.extend(
        [
            "",
            "## 直接 prompt 内容复核",
            "",
            "这里把‘映射表里保存的题目等于 all_benchmarks’和‘RouterEval 实际 prompt 的目标题文本等于 "
            "all_benchmarks’分开报告，避免把两种检查混为一谈。",
            "",
        ]
    )
    for name in ("mmlu_pro", "winogrande"):
        counts = audit["datasets"][name]["direct_prompt_content_status_counts"]
        lines.append(f"- **{name}**：" + "；".join(f"{key} = {value:,}" for key, value in counts.items()) + "。")

    lines.extend(
        [
            "",
            "MMLU-Pro 的少量 `source_variant_same_exact_options` 是题干在两份源数据中的增删/格式差异，"
            "完整选项块一致；Winogrande 的 `placeholder_position_variant_same_sentence` 是同一实体在句中"
            "重复时下划线重建位置不同。两类记录均保留在 CSV 中供逐题审计，没有未解决或多对一记录。",
            "",
            "## 可直接用于矩阵连接的关系",
            "",
            "```text",
            "RouterEval MMLU-Pro matrix[38233:50265]  -> all_benchmarks IDs[0:12032]",
            "RouterEval Winogrande matrix[0:1267]    -> all_benchmarks IDs[48998:50265]",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--old-to-new", type=Path, default=DEFAULT_OLD_TO_NEW)
    parser.add_argument("--all-benchmarks", type=Path, default=DEFAULT_ALL_BENCHMARKS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    for path in (
        args.router_items,
        args.prompts,
        args.old_to_new,
        args.all_benchmarks,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    router_items = load_router_items(args.router_items)
    prompts = load_target_prompts(args.prompts)
    authoritative = load_authoritative_mapping(args.old_to_new)
    all_benchmarks = json.loads(args.all_benchmarks.read_text(encoding="utf-8"))
    if len(all_benchmarks) != 50265:
        raise ValueError(
            f"Expected 50,265 all_benchmarks items, found {len(all_benchmarks):,}"
        )
    if any(int(row.get("id", -1)) != idx for idx, row in enumerate(all_benchmarks)):
        raise ValueError("all_benchmarks IDs are not dense list positions")

    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {
        "mapping_semantics": (
            "RouterEval question-axis global row -> all_benchmarks_dataset.json ID/list position"
        ),
        "inputs": {
            "router_items": str(args.router_items.resolve()),
            "router_items_sha256": sha256_file(args.router_items),
            "prompts": str(args.prompts.resolve()),
            "prompts_sha256": sha256_file(args.prompts),
            "authoritative_old_to_new": str(args.old_to_new.resolve()),
            "authoritative_old_to_new_sha256": sha256_file(args.old_to_new),
            "all_benchmarks": str(args.all_benchmarks.resolve()),
            "all_benchmarks_sha256": sha256_file(args.all_benchmarks),
        },
        "datasets": {},
        "unresolved_count": 0,
    }

    combined_rows: list[dict[str, Any]] = []
    for name, spec in DATASETS.items():
        global_start = int(spec["global_start"])
        global_stop = int(spec["global_stop"])
        expected_all_ids = list(range(int(spec["all_start"]), int(spec["all_stop"])))
        output_rows: list[dict[str, Any]] = []

        for dataset_idx, global_id in enumerate(range(global_start, global_stop)):
            item_row = router_items[global_id]
            prompt_row = prompts[global_id]
            mapped = authoritative[global_id]

            if item_row["dataset"] != spec["dataset"]:
                raise ValueError(f"Router item {global_id}: wrong dataset")
            if item_row["dataset_name"] != spec["dataset_name"]:
                raise ValueError(f"Router item {global_id}: wrong dataset_name")
            if prompt_row["dataset"] != spec["dataset"]:
                raise ValueError(f"Prompt item {global_id}: wrong dataset")
            if prompt_row["dataset_name"] != spec["dataset_name"]:
                raise ValueError(f"Prompt item {global_id}: wrong dataset_name")
            if mapped["old_dataset"] != spec["dataset"]:
                raise ValueError(f"Mapped item {global_id}: wrong old_dataset")
            if mapped["old_dataset_name"] != spec["dataset_name"]:
                raise ValueError(f"Mapped item {global_id}: wrong old_dataset_name")
            if mapped["matched"].lower() != "true" or mapped[
                "benchmark_matched"
            ].lower() != "true":
                raise ValueError(f"Mapped item {global_id} is not marked matched")

            all_id = int(mapped["new_id"])
            benchmark_row = all_benchmarks[all_id]
            problem = str(benchmark_row["problem"])
            if str(benchmark_row["source"]) != spec["all_source"]:
                raise ValueError(f"Mapped item {global_id}: wrong all_benchmarks source")
            if mapped["new_source"] != spec["all_source"]:
                raise ValueError(f"Mapped item {global_id}: wrong mapped new_source")
            if mapped["new_problem"] != problem:
                raise ValueError(
                    f"Mapped item {global_id}: stored new_problem differs from all_benchmarks"
                )

            if name == "mmlu_pro":
                content_status, similarity = verify_mmlu_prompt(prompt_row, problem)
            else:
                content_status, similarity = verify_winogrande_prompt(
                    prompt_row, problem
                )

            output_rows.append(
                {
                    "dataset": name,
                    "routereval_global_item_id": global_id,
                    "routereval_dataset_item_index": dataset_idx,
                    "routereval_dataset_name": spec["dataset_name"],
                    "all_benchmarks_id": all_id,
                    "all_benchmarks_source": benchmark_row["source"],
                    "match_method": "authoritative_old_item_id_to_new_id",
                    "mapped_problem_exact_all_benchmarks": True,
                    "direct_prompt_content_status": content_status,
                    "direct_prompt_similarity": f"{similarity:.9f}",
                    "problem_sha256": sha256_text(problem),
                    "problem_preview": problem_preview(problem),
                }
            )

        actual_all_ids = [int(row["all_benchmarks_id"]) for row in output_rows]
        if actual_all_ids != expected_all_ids:
            raise ValueError(
                f"{name}: mapping is not the expected dense, order-preserving ID run"
            )
        if len(set(actual_all_ids)) != len(actual_all_ids):
            raise ValueError(f"{name}: all_benchmarks mapping is not one-to-one")

        statuses = Counter(
            str(row["direct_prompt_content_status"]) for row in output_rows
        )
        rows_by_dataset[name] = output_rows
        combined_rows.extend(output_rows)
        audit["datasets"][name] = {
            "n_items": len(output_rows),
            "routereval_global_range": f"{global_start}-{global_stop - 1}",
            "routereval_global_slice": f"[{global_start}:{global_stop}]",
            "all_benchmarks_id_range": (
                f"{expected_all_ids[0]}-{expected_all_ids[-1]}"
            ),
            "all_benchmarks_slice": (
                f"[{expected_all_ids[0]}:{expected_all_ids[-1] + 1}]"
            ),
            "routereval_ids_unique": len(
                {int(row["routereval_global_item_id"]) for row in output_rows}
            ),
            "all_benchmarks_ids_unique": len(set(actual_all_ids)),
            "one_to_one": True,
            "coverage_percent": 100.0,
            "mapped_problem_exact_count": sum(
                row["mapped_problem_exact_all_benchmarks"] for row in output_rows
            ),
            "direct_prompt_content_status_counts": dict(sorted(statuses.items())),
        }

    if len({int(row["routereval_global_item_id"]) for row in combined_rows}) != len(
        combined_rows
    ):
        raise ValueError("Combined mapping repeats a RouterEval global item ID")
    if len({int(row["all_benchmarks_id"]) for row in combined_rows}) != len(
        combined_rows
    ):
        raise ValueError("Combined mapping repeats an all_benchmarks ID")
    combined_rows.sort(key=lambda row: int(row["routereval_global_item_id"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in rows_by_dataset.items():
        write_csv(args.out_dir / f"routereval_{name}_to_all_benchmarks.csv", rows)
    write_csv(
        args.out_dir / "routereval_mmlu_pro_winogrande_to_all_benchmarks.csv",
        combined_rows,
    )

    np.savez_compressed(
        args.out_dir / "routereval_mmlu_pro_winogrande_to_all_benchmarks.npz",
        mmlu_pro_routereval_global_item_ids=np.asarray(
            [row["routereval_global_item_id"] for row in rows_by_dataset["mmlu_pro"]],
            dtype=np.int32,
        ),
        mmlu_pro_all_benchmarks_ids=np.asarray(
            [row["all_benchmarks_id"] for row in rows_by_dataset["mmlu_pro"]],
            dtype=np.int32,
        ),
        winogrande_routereval_global_item_ids=np.asarray(
            [row["routereval_global_item_id"] for row in rows_by_dataset["winogrande"]],
            dtype=np.int32,
        ),
        winogrande_all_benchmarks_ids=np.asarray(
            [row["all_benchmarks_id"] for row in rows_by_dataset["winogrande"]],
            dtype=np.int32,
        ),
    )

    audit_path = (
        args.out_dir / "routereval_mmlu_pro_winogrande_mapping_audit.json"
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path = (
        args.out_dir / "routereval_mmlu_pro_winogrande_mapping_report.md"
    )
    report_path.write_text(build_report(audit), encoding="utf-8")

    print(json.dumps(audit["datasets"], ensure_ascii=False, indent=2))
    print(f"Wrote {len(combined_rows):,} one-to-one mappings to {args.out_dir}")


if __name__ == "__main__":
    main()
