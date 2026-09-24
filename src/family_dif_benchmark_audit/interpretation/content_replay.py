"""Recompute the frozen content-audit statistics from anonymous numeric labels.

This module performs no model calls and needs no benchmark questions. The
optional export command converts authorized private records to the explicit
public-column allowlist; the source files and mappings are never copied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import blinded_content_v3 as audit

LABEL_COLUMNS = [
    "annotator", "model", "blinded_id", "benchmark", "pair_id", "group",
    "common_blueprint_cell", "discovery_advantaged_family",
    "validation_advantaged_family", *audit.BINARY_AXES, *audit.ORDINAL_AXES,
]
AGGREGATE_FILES = (
    "binary_annotation_reliability.csv", "ordinal_annotation_reliability.csv",
    "paired_binary_enrichment.csv", "paired_ordinal_enrichment.csv",
    "exploratory_family_direction_association.csv",
    "exploratory_family_direction_prevalence.csv",
)


def validate_labels(frame: pd.DataFrame, require_paper_size: bool = True) -> None:
    if set(frame.columns) != set(LABEL_COLUMNS):
        raise ValueError("labels must contain exactly the documented public columns")
    if frame.isna().any().any():
        raise ValueError("published labels contain missing values")
    if set(frame.annotator) != set(audit.ANNOTATOR_MODELS):
        raise ValueError("expected the two frozen annotators")
    if frame.duplicated(["annotator", "blinded_id"]).any():
        raise ValueError("duplicate annotator/item record")
    for axis in audit.BINARY_AXES:
        if not frame[axis].isin([0, 1]).all():
            raise ValueError("invalid binary label: " + axis)
    for axis in audit.ORDINAL_AXES:
        if not frame[axis].isin([0, 1, 2, 3]).all():
            raise ValueError("invalid ordinal label: " + axis)
    for half in ("discovery", "validation"):
        if not frame[half + "_advantaged_family"].isin(audit.FAMILIES).all():
            raise ValueError("invalid family label")
    groups = {"stable_high_dif", "matched_control"}
    if set(frame.group) != groups:
        raise ValueError("invalid content-audit groups")
    for annotator, part in frame.groupby("annotator", sort=False):
        if not part.model.eq(audit.ANNOTATOR_MODELS[annotator]).all():
            raise ValueError("annotator model differs from frozen protocol")
        pairs = part.groupby(["benchmark", "pair_id"], sort=False)
        if not pairs.size().eq(2).all() or not pairs.group.nunique().eq(2).all():
            raise ValueError("each matched pair must contain one high-DIF and one control item")
        if not pairs.common_blueprint_cell.nunique().eq(1).all():
            raise ValueError("matched pair crosses blueprint cells")
        if require_paper_size:
            if len(part) != 500 or not part.groupby("benchmark").size().eq(100).all() or part.benchmark.nunique() != 5:
                raise ValueError("expected 500 items and 50 matched pairs per benchmark per annotator")
    metadata = ["benchmark", "pair_id", "group", "common_blueprint_cell",
                "discovery_advantaged_family", "validation_advantaged_family"]
    if not frame.groupby("blinded_id").size().eq(2).all():
        raise ValueError("both annotators must label the same items")
    if (frame.groupby("blinded_id")[metadata].nunique() != 1).any().any():
        raise ValueError("item metadata differs between annotators")


def export_labels(annotations_path: Path, signatures_path: Path, output: Path) -> None:
    """Export numeric annotations with fresh item/pair/cell IDs only."""
    annotations = pd.read_csv(annotations_path)
    signatures = pd.read_csv(signatures_path)
    effects = [f"{half}_{family}_effect" for half in ("discovery", "validation") for family in audit.FAMILIES]
    # Preserve the original row order because Monte Carlo permutations depend on it.
    joined = annotations.merge(signatures[["benchmark", "dataset_item_index", *effects]],
                               on=["benchmark", "dataset_item_index"], how="left", validate="many_to_one", sort=False)
    if joined[effects].isna().any().any():
        raise ValueError("private annotation records could not be linked to family labels")
    for half in ("discovery", "validation"):
        values = joined[[f"{half}_{family}_effect" for family in audit.FAMILIES]].to_numpy(float)
        joined[half + "_advantaged_family"] = np.asarray(audit.FAMILIES)[np.argmax(values, axis=1)]
    for column, prefix in (("blinded_id", "record"), ("pair_id", "pair"), ("common_blueprint_cell", "cell")):
        # Cells are benchmark-specific; original values and mappings stay local.
        keys = list(zip(joined.benchmark.astype(str), joined[column].astype(str)))
        mapping = {key: f"{prefix}_{i:04d}" for i, key in enumerate(dict.fromkeys(keys))}
        joined[column] = [mapping[key] for key in keys]
    public = joined[LABEL_COLUMNS]
    validate_labels(public)
    output.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(output, index=False)


def replay(labels: Path, output_dir: Path, reference_dir: Path | None = None,
           permutations: int = audit.FAMILY_DIRECTION_PERMUTATIONS) -> dict:
    frame = pd.read_csv(labels)
    validate_labels(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    if reference_dir is not None and output_dir.resolve() == reference_dir.resolve():
        raise ValueError("replay output must differ from the frozen reference directory")
    binary = pd.concat([audit.paired_binary_results(frame, name) for name in audit.ANNOTATOR_MODELS], ignore_index=True)
    ordinal = pd.concat([audit.ordinal_results(frame, name) for name in audit.ANNOTATOR_MODELS], ignore_index=True)
    kappa, ordinal_reliability = audit.reliability_results(frame)
    confirmatory_axes = []
    for axis in audit.BINARY_AXES:
        part = binary[binary.axis.eq(axis)].set_index("annotator")
        directions = np.sign(part.paired_difference.to_numpy(float))
        if (directions[0] != 0 and directions[0] == directions[1]
            and part.bh_q.le(audit.BH_ALPHA).all()
            and part.paired_difference.abs().ge(audit.MIN_POOLED_DIFFERENCE).all()
            and part.benchmarks_same_direction.ge(audit.MIN_BENCHMARK_DIRECTION_COUNT).all()):
            confirmatory_axes.append(axis)
    binary["replicated_confirmatory_axis"] = binary.axis.isin(confirmatory_axes)
    family, prevalence, family_audit = audit.family_direction_results(frame, None, permutations)
    tables = dict(zip(AGGREGATE_FILES, (kappa, ordinal_reliability, binary, ordinal, family, prevalence)))
    checked = []
    for name, table in tables.items():
        if reference_dir is not None:
            reference = pd.read_csv(reference_dir / name)
            if set(table.columns) != set(reference.columns):
                raise ValueError("reference schema differs for " + name)
            # Benchmark column order can depend on frame serialization; compare named columns.
            pd.testing.assert_frame_equal(table.reindex(columns=reference.columns), reference,
                                          check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
            checked.append(name)
        table.to_csv(output_dir / name, index=False)
    median_kappa = float(kappa.cohen_kappa.dropna().median())
    median_rho = float(ordinal_reliability.spearman_r.median())
    summary = {"status": "pass", "annotation_rows": len(frame), "unique_items": frame.blinded_id.nunique(),
               "matched_pairs": frame.pair_id.nunique(), "labels_sha256": audit.sha256(labels),
               "median_binary_kappa": median_kappa, "median_ordinal_rho": median_rho,
               "reliability_gate_passed": bool(median_kappa >= audit.RELIABILITY_KAPPA_GATE and median_rho >= audit.RELIABILITY_ORDINAL_RHO_GATE),
               "replicated_confirmatory_axes": confirmatory_axes,
               "exploratory_family_direction": family_audit, "verified_reference_tables": checked,
               "remote_model_calls": 0}
    (output_dir / "replay_verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "exploratory_family_direction_audit.json").write_text(json.dumps(family_audit, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("replay", help="Recompute frozen statistical analysis without model calls")
    run.add_argument("--labels", type=Path, default=Path("results/content_audit/labels.csv"))
    run.add_argument("--output-dir", type=Path, default=Path("outputs/content_audit_replay"))
    run.add_argument("--reference-dir", type=Path, default=Path("results/content_audit"))
    run.add_argument("--permutations", type=int, default=audit.FAMILY_DIRECTION_PERMUTATIONS)
    export = sub.add_parser("export", help="Allowlist authorized local annotations for anonymous release")
    export.add_argument("--annotations", type=Path, required=True)
    export.add_argument("--signatures", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_labels(args.annotations, args.signatures, args.output)
    else:
        if args.permutations < 1:
            parser.error("--permutations must be positive")
        print(json.dumps(replay(args.labels, args.output_dir, args.reference_dir, args.permutations), indent=2))


if __name__ == "__main__":
    main()
