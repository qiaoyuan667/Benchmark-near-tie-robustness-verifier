#!/usr/bin/env python3
"""Owner-disjoint within-family near-tie diagnostic for the frozen audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


from .._paths import PROJECT_ROOT
from . import require_primary_files
from ..ranking import mirt_primary
from ..ranking.mmlu_pro import (
    matched_random_anchor_weights,
)


BENCHMARKS: Dict[str, str] = {
    "mmlu_pro": "MMLU-Pro",
    "bbh": "BBH",
    "mmlu": "MMLU",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
}
FOLDS = (("discovery", "validation"), ("validation", "discovery"))
PRIMARY_FRACTION = 0.5
CLOSE_GAP = 0.01
REPLICATES = 1000
ORIGINAL_SEED = 20260825


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        default=",".join(BENCHMARKS),
        help="Comma-separated benchmark keys.",
    )
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--primary-root", type=Path, default=PROJECT_ROOT / "outputs",
                        help="Parent of the five reconstructed primary benchmark directories.")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/within_family_diagnostic")
    return parser.parse_args(argv)


def pairwise_metrics(
    full_scores: np.ndarray,
    recomposed_scores: np.ndarray,
    families: np.ndarray,
    owners: np.ndarray,
) -> Dict[str, Any]:
    """Strict close-pair reversals for cross- and within-family pairs."""
    full = np.asarray(full_scores, dtype=np.float64)
    recomposed = np.asarray(recomposed_scores, dtype=np.float64)
    family = np.asarray(families).astype(str)
    owner = np.asarray(owners).astype(str)
    left, right = np.triu_indices(len(full), k=1)
    full_difference = full[left] - full[right]
    recomposed_difference = recomposed[left] - recomposed[right]
    different_owner = owner[left] != owner[right]
    close = np.abs(full_difference) <= CLOSE_GAP
    non_tied = (full_difference != 0.0) & (recomposed_difference != 0.0)
    reversed_pair = full_difference * recomposed_difference < 0.0
    base = different_owner & close & non_tied

    def summarize(mask: np.ndarray, prefix: str) -> Dict[str, Any]:
        denominator = base & mask
        pairs = int(denominator.sum())
        flips = int((denominator & reversed_pair).sum())
        return {
            prefix + "_eligible_pairs": pairs,
            prefix + "_flips": flips,
            prefix + "_flip_rate": float(flips / pairs) if pairs else float("nan"),
        }

    result: Dict[str, Any] = {}
    result.update(summarize(family[left] != family[right], "close_cross_family"))
    result.update(summarize(family[left] == family[right], "close_same_family"))
    return result


def load_fold(
    benchmark_key: str,
    fold_index: int,
    run_dir: Path,
    data: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load frozen anchors and the opposite owner-disjoint target half."""
    audit_half, target_half = FOLDS[fold_index]
    representatives = pd.read_csv(run_dir / "owner_family_representatives.csv")
    target = representatives[representatives["model_half"].eq(target_half)].copy()
    target = target.sort_values("dataset_model_index")
    responses = np.asarray(data["responses"], dtype=np.float64)
    target_responses = responses[
        :, target["dataset_model_index"].to_numpy(dtype=np.int32)
    ]

    frozen = pd.read_csv(run_dir / "crossfit_anchor_items.csv")
    frozen = frozen[
        np.isclose(frozen["anchor_fraction"], PRIMARY_FRACTION)
        & frozen["audit_half"].eq(audit_half)
    ].sort_values("dataset_item_index")
    expected_index = np.arange(responses.shape[0], dtype=np.int64)
    if not np.array_equal(
        frozen["dataset_item_index"].to_numpy(dtype=np.int64), expected_index
    ):
        raise AssertionError("frozen item table is not dense")
    weights = frozen["anchor_weight"].to_numpy(dtype=np.float64)
    selected = frozen["selected_anchor"].to_numpy(dtype=bool)
    cells = frozen["difficulty_cell"].to_numpy(dtype=str)
    if not np.array_equal(weights > 0.0, selected):
        raise AssertionError("frozen selected-anchor mask and weights disagree")
    if not np.isclose(weights.sum(), responses.shape[0], rtol=0.0, atol=1e-8):
        raise AssertionError("frozen anchor weights do not preserve item mass")

    return (
        target_responses,
        target["family"].to_numpy(dtype=str),
        target["owner"].to_numpy(dtype=str),
        weights,
        selected,
        cells,
    )


def evaluate_weights(
    target_responses: np.ndarray,
    families: np.ndarray,
    owners: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, Any]:
    full_scores = target_responses.mean(axis=0)
    recomposed_scores = np.asarray(weights, dtype=np.float64) @ target_responses
    recomposed_scores = recomposed_scores / float(np.asarray(weights).sum())
    return pairwise_metrics(full_scores, recomposed_scores, families, owners)


def pooled_rates(frame: pd.DataFrame, prefix: str) -> pd.Series:
    pooled = frame.groupby("replicate").agg(
        flips=(prefix + "_flips", "sum"),
        pairs=(prefix + "_eligible_pairs", "sum"),
    )
    if (pooled["pairs"] <= 0).any():
        raise AssertionError("a pooled replicate has no eligible " + prefix + " pairs")
    return pooled["flips"] / pooled["pairs"]


def empirical_upper_p(null: np.ndarray, observed: float) -> float:
    values = np.asarray(null, dtype=np.float64)
    return float((1 + np.sum(values >= observed)) / (len(values) + 1))


def summarize_benchmark(
    benchmark: str,
    observed: pd.DataFrame,
    controls: pd.DataFrame,
) -> Dict[str, Any]:
    cross_observed = float(
        observed["close_cross_family_flips"].sum()
        / observed["close_cross_family_eligible_pairs"].sum()
    )
    same_observed = float(
        observed["close_same_family_flips"].sum()
        / observed["close_same_family_eligible_pairs"].sum()
    )
    cross_null = pooled_rates(controls, "close_cross_family")
    same_null = pooled_rates(controls, "close_same_family")
    paired_null_gap = cross_null - same_null
    observed_gap = cross_observed - same_observed
    cross_median = float(cross_null.median())
    same_median = float(same_null.median())
    gap_median = float(paired_null_gap.median())
    return {
        "benchmark": benchmark,
        "cross_family_pairs": int(observed["close_cross_family_eligible_pairs"].sum()),
        "cross_family_low_dif_rate": cross_observed,
        "cross_family_random_median": cross_median,
        "cross_family_excess_points": 100.0 * (cross_observed - cross_median),
        "cross_family_p": empirical_upper_p(cross_null.to_numpy(), cross_observed),
        "same_family_pairs": int(observed["close_same_family_eligible_pairs"].sum()),
        "same_family_low_dif_rate": same_observed,
        "same_family_random_median": same_median,
        "same_family_excess_points": 100.0 * (same_observed - same_median),
        "same_family_p": empirical_upper_p(same_null.to_numpy(), same_observed),
        "observed_cross_minus_same_points": 100.0 * observed_gap,
        "random_cross_minus_same_median_points": 100.0 * gap_median,
        "specificity_excess_points": 100.0 * (observed_gap - gap_median),
        "specificity_p": empirical_upper_p(paired_null_gap.to_numpy(), observed_gap),
        "replicates": int(len(cross_null)),
    }


def run_benchmark(
    key: str, args: argparse.Namespace
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    label = BENCHMARKS[key]
    run_dir = args.primary_root / (
        key + "_family_ranking_impact"
    )
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    data = mirt_primary.load_data(key)
    frozen_controls = pd.read_csv(run_dir / "matched_random_anchor_controls.csv")
    frozen_observed = pd.read_csv(run_dir / "ranking_impact_metrics.csv")
    observed_rows = []
    control_rows = []

    for fold_index, (audit_half, target_half) in enumerate(FOLDS):
        print(f"{label}: audit={audit_half}, target={target_half}", flush=True)
        target_responses, families, owners, weights, selected, cells = load_fold(
            key, fold_index, run_dir, data
        )
        observed = evaluate_weights(target_responses, families, owners, weights)
        observed_rows.append(
            {
                "benchmark": label,
                "audit_half": audit_half,
                "target_half": target_half,
                **observed,
            }
        )

        frozen_fold_observed = frozen_observed[
            np.isclose(frozen_observed["anchor_fraction"], PRIMARY_FRACTION)
            & frozen_observed["audit_half"].eq(audit_half)
        ]
        if len(frozen_fold_observed) != 1:
            raise AssertionError("expected one frozen primary observed row")
        for column in (
            "close_cross_family_flips",
            "close_cross_family_eligible_pairs",
        ):
            if int(observed[column]) != int(frozen_fold_observed.iloc[0][column]):
                raise AssertionError("frozen observed result differs: " + column)

        rng = np.random.default_rng(ORIGINAL_SEED + fold_index)
        fold_controls = []
        for replicate in range(args.replicates):
            random_weights = matched_random_anchor_weights(cells, selected, rng)
            metrics = evaluate_weights(
                target_responses, families, owners, random_weights
            )
            row = {
                "benchmark": label,
                "audit_half": audit_half,
                "target_half": target_half,
                "replicate": replicate,
                **metrics,
            }
            fold_controls.append(row)
            control_rows.append(row)
            if (replicate + 1) % 250 == 0:
                print(
                    f"{label} audit={audit_half}: {replicate + 1}/{args.replicates}",
                    flush=True,
                )

        reproduced = pd.DataFrame(fold_controls).sort_values("replicate")
        frozen = frozen_controls[
            frozen_controls["audit_half"].eq(audit_half)
        ].sort_values("replicate")
        for column in (
            "close_cross_family_flips",
            "close_cross_family_eligible_pairs",
        ):
            if not np.array_equal(
                reproduced[column].to_numpy(dtype=np.int64),
                frozen[column].to_numpy(dtype=np.int64),
            ):
                raise AssertionError("frozen random controls differ: " + column)

    observed_frame = pd.DataFrame(observed_rows)
    control_frame = pd.DataFrame(control_rows)
    return (
        observed_frame,
        control_frame,
        summarize_benchmark(label, observed_frame, control_frame),
    )


def markdown_report(summary: pd.DataFrame) -> str:
    lines = [
        "# Within-family near-tie diagnostic",
        "",
        "This post-hoc diagnostic uses the frozen 50% low-DIF recomposition and",
        "the same 1,000 source-by-easiness-matched random subtests as the primary",
        "analysis. All pairs are different-owner near ties under the full benchmark.",
        "",
        "| Benchmark | Cross: low/random/excess | Within: low/random/excess | Specificity excess | p(spec.) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| %s | %.1f/%.1f/%+.1f | %.1f/%.1f/%+.1f | %+.1f pp | %.3f |"
            % (
                row.benchmark,
                100.0 * row.cross_family_low_dif_rate,
                100.0 * row.cross_family_random_median,
                row.cross_family_excess_points,
                100.0 * row.same_family_low_dif_rate,
                100.0 * row.same_family_random_median,
                row.same_family_excess_points,
                row.specificity_excess_points,
                row.specificity_p,
            )
        )
    lines.extend(
        [
            "",
            "Specificity excess is the observed cross-minus-within reversal gap",
            "minus the median paired random-subtest gap. Its one-sided empirical",
            "p-value compares the observed gap with the 1,000 paired random gaps.",
        ]
    )
    return "\n".join(lines) + "\n"


def latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Post-hoc family-specificity diagnostic for different-owner model pairs within one percentage point on the full benchmark. Cross and Within report low-DIF / matched-random reversal rates. $\Delta_{\mathrm{spec}}$ is the low-DIF cross-minus-within gap minus the median paired random-subtest gap; $p_{\mathrm{spec}}$ is its one-sided empirical value.}",
        r"\label{tab:within-family-diagnostic}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Benchmark & Cross (\%) & Within (\%) & $\Delta_{\mathrm{spec}}$ (pp) & $p_{\mathrm{spec}}$ \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "%s & %.1f/%.1f & %.1f/%.1f & %+.1f & %.3f \\\\"
            % (
                row.benchmark,
                100.0 * row.cross_family_low_dif_rate,
                100.0 * row.cross_family_random_median,
                100.0 * row.same_family_low_dif_rate,
                100.0 * row.same_family_random_median,
                row.specificity_excess_points,
                row.specificity_p,
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_checksums(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    keys = tuple(value.strip() for value in args.benchmarks.split(",") if value.strip())
    unknown = set(keys).difference(BENCHMARKS)
    if unknown or not keys or len(keys) != len(set(keys)):
        raise ValueError("supply distinct, known benchmarks: %s" % sorted(unknown))
    if args.replicates != REPLICATES:
        raise ValueError("formal diagnostic freezes 1,000 matched-random replicates")
    require_primary_files(args.primary_root, keys)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output is not empty; choose a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    all_observed = []
    all_controls = []
    summaries = []
    for key in keys:
        observed, controls, summary = run_benchmark(key, args)
        all_observed.append(observed)
        all_controls.append(controls)
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(
            args.output_dir / "summary.partial.csv", index=False
        )

    observed = pd.concat(all_observed, ignore_index=True)
    controls = pd.concat(all_controls, ignore_index=True)
    summary = pd.DataFrame(summaries)
    observed.to_csv(args.output_dir / "observed_fold_metrics.csv", index=False)
    controls.to_csv(args.output_dir / "matched_random_controls.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "report.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    (args.output_dir / "appendix_table.tex").write_text(
        latex_table(summary), encoding="utf-8"
    )
    protocol = {
        "experiment": "posthoc_within_family_near_tie_diagnostic",
        "benchmarks": [BENCHMARKS[key] for key in keys],
        "primary_anchor_fraction": PRIMARY_FRACTION,
        "close_pair_threshold": "1 percentage point on frozen full score",
        "pair_types": {
            "cross_family": "different owner, different family",
            "within_family": "different owner, same family",
        },
        "ties": "exact full-score or recomposed-score ties excluded",
        "control": "frozen source x easiness matched-random subtests",
        "replicates": args.replicates,
        "seed_base": ORIGINAL_SEED,
        "specificity_statistic": "(cross-within)_low-DIF - median[(cross-within)_matched-random]",
        "status": "post-hoc diagnostic",
        "owner_disjoint": True,
        "mirt_refit": False,
        "anchor_reselection": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    partial = args.output_dir / "summary.partial.csv"
    if partial.exists():
        partial.unlink()
    write_checksums(args.output_dir)
    print(summary.to_string(index=False), flush=True)
    print("written to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
