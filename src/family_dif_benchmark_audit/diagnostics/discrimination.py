#!/usr/bin/env python3
"""Discrimination-matched negative-control ablation for the extended-K audit.

This runner is deliberately isolated from the frozen release and the extended-K
primary outputs.  It reconstructs each frozen 50% spectral fit, verifies the
anchor mask and weights byte-for-value against the saved run, and then compares
the observed low-DIF test with two 1,000-replicate nulls:

1. the original source-by-easiness matched random subtests; and
2. random subtests additionally matched to the low-DIF anchors' distribution
   of family-blind spectral predicted-logit spread.

All discrimination quantities are learned from the audit owner half only.  The
opposite owner half is used only for ranking and reliability evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


from .._paths import PROJECT_ROOT
from . import require_primary_files
from ..ranking import mirt_primary
from ..ranking.mmlu_pro import (
    matched_random_anchor_weights,
    pairwise_ranking_metrics,
    weighted_accuracy,
)
from ..core.spectral_mirt import (
    fit_spectral_mirt_offsets,
    smoothed_item_parameters,
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
REPLICATES = 1000
ORIGINAL_SEED = 20260825
DISCRIMINATION_SEED = 20260917
SVD_SEED = 20260826


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", default=",".join(BENCHMARKS),
        help="Comma-separated benchmark keys.",
    )
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--max-discrimination-strata", type=int, default=4)
    parser.add_argument("--primary-root", type=Path, default=PROJECT_ROOT / "outputs",
                        help="Parent of the five reconstructed primary benchmark directories.")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/discrimination_diagnostic")
    return parser.parse_args(argv)


def corrected_item_total_correlation(responses: np.ndarray) -> np.ndarray:
    """Vectorized point-biserial correlation with the leave-one-item-out total."""
    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] < 3:
        raise ValueError("responses must be item by model with at least 3 models")
    n = float(y.shape[1])
    total = y.sum(axis=0)
    sum_total = float(total.sum())
    sum_total_sq = float(np.square(total).sum())
    sum_y = y.sum(axis=1)
    sum_y_total = y @ total
    sum_x = sum_total - sum_y
    sum_x_sq = sum_total_sq - 2.0 * sum_y_total + sum_y
    sum_xy = sum_y_total - sum_y
    centered_xy = sum_xy - sum_x * sum_y / n
    centered_xx = sum_x_sq - np.square(sum_x) / n
    centered_yy = sum_y - np.square(sum_y) / n
    denominator = np.sqrt(np.maximum(centered_xx * centered_yy, 0.0))
    result = np.zeros(y.shape[0], dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = centered_xy[valid] / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def discrimination_cells(
    blueprint_cells: np.ndarray,
    discrimination: np.ndarray,
    max_strata: int,
) -> np.ndarray:
    """Near-equal within-blueprint discrimination strata with no empty cells."""
    blueprint = np.asarray(blueprint_cells).astype(str)
    values = np.asarray(discrimination, dtype=np.float64)
    if len(blueprint) != len(values):
        raise ValueError("blueprint cells and discrimination differ")
    result = np.empty(len(blueprint), dtype=object)
    for cell in np.unique(blueprint):
        positions = np.flatnonzero(blueprint == cell)
        order = positions[np.lexsort((positions, values[positions]))]
        count = min(max_strata, len(order))
        for stratum, part in enumerate(np.array_split(order, count)):
            result[part] = "%s|discrimination_%d" % (cell, stratum)
    return result.astype(str)


def discrimination_matched_weights(
    blueprint_cells: np.ndarray,
    discrimination_cells_: np.ndarray,
    selected_template: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Match anchor counts by discrimination cell but preserve blueprint mass."""
    blueprint = np.asarray(blueprint_cells).astype(str)
    disc = np.asarray(discrimination_cells_).astype(str)
    selected = np.asarray(selected_template, dtype=bool)
    if not (len(blueprint) == len(disc) == len(selected)):
        raise ValueError("matching vectors differ")
    weights = np.zeros(len(selected), dtype=np.float64)
    for cell in np.unique(blueprint):
        cell_positions = np.flatnonzero(blueprint == cell)
        selected_in_cell = int(selected[cell_positions].sum())
        if selected_in_cell <= 0:
            raise ValueError("every blueprint cell must contain an anchor")
        cell_weight = float(len(cell_positions)) / float(selected_in_cell)
        for disc_cell in np.unique(disc[cell_positions]):
            positions = np.flatnonzero(disc == disc_cell)
            n_selected = int(selected[positions].sum())
            if n_selected == 0:
                continue
            chosen = rng.choice(positions, size=n_selected, replace=False)
            weights[chosen] = cell_weight
    for cell in np.unique(blueprint):
        mask = blueprint == cell
        if not np.isclose(weights[mask].sum(), mask.sum(), atol=1e-8, rtol=0.0):
            raise AssertionError("discrimination matching changed blueprint mass")
    return weights


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.dot(values, weights) / weights.sum())


def independent_item_score_sem(
    item_variance: np.ndarray, weights: np.ndarray
) -> float:
    """Independent-item score SEM on the normalized weighted scale."""
    variance = np.asarray(item_variance, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    weighted_variance = np.sum(np.square(w) * variance)
    return float(np.sqrt(max(weighted_variance, 0.0)) / w.sum())


def weighted_cronbach_alpha(
    weighted_total: np.ndarray,
    item_variance: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Cronbach alpha for weighted item contributions across target models."""
    w = np.asarray(weights, dtype=np.float64)
    selected = w > 0.0
    k = int(selected.sum())
    if k < 2 or len(weighted_total) < 3:
        return float("nan")
    total_variance = float(np.asarray(weighted_total).var(ddof=1))
    if total_variance <= 0.0:
        return float("nan")
    weighted_item_variance = float(
        np.sum(np.square(w[selected]) * np.asarray(item_variance)[selected])
    )
    return float(
        k / (k - 1.0) * (1.0 - weighted_item_variance / total_variance)
    )


def effective_item_count(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    return float(np.square(w.sum()) / np.square(w).sum())


def evaluate_weights(
    responses: np.ndarray,
    full_scores: np.ndarray,
    target_item_variance: np.ndarray,
    weights: np.ndarray,
    families: np.ndarray,
    owners: np.ndarray,
    discrimination: np.ndarray,
    citc: np.ndarray,
) -> Dict[str, Any]:
    weight_sum = float(np.asarray(weights).sum())
    weighted_total = np.asarray(weights, dtype=np.float64) @ responses
    subset = weighted_total / weight_sum
    metrics = pairwise_ranking_metrics(
        full_scores, subset, families, owners, 0.01
    )
    metrics.pop("close_pair_mask")
    return {
        **metrics,
        "mean_spectral_discrimination": weighted_mean(discrimination, weights),
        "mean_corrected_item_total_correlation": weighted_mean(citc, weights),
        "heldout_independent_item_score_sem": independent_item_score_sem(
            target_item_variance, weights
        ),
        "weighted_cronbach_alpha": weighted_cronbach_alpha(
            weighted_total, target_item_variance, weights
        ),
        "effective_item_count": effective_item_count(weights),
        "n_selected_items": int(np.sum(np.asarray(weights) > 0.0)),
    }


def pooled_metric(controls: pd.DataFrame, column: str) -> pd.Series:
    if column == "close_cross_family_flip_rate":
        group = controls.groupby("replicate").agg(
            flips=("close_cross_family_flips", "sum"),
            pairs=("close_cross_family_eligible_pairs", "sum"),
        )
        return group["flips"] / group["pairs"]
    return controls.groupby("replicate")[column].mean()


def summarize_benchmark(
    benchmark: str,
    observed: pd.DataFrame,
    original: pd.DataFrame,
    matched: pd.DataFrame,
) -> Dict[str, Any]:
    observed_rate = float(
        observed["close_cross_family_flips"].sum()
        / observed["close_cross_family_eligible_pairs"].sum()
    )
    original_rate = pooled_metric(original, "close_cross_family_flip_rate")
    matched_rate = pooled_metric(matched, "close_cross_family_flip_rate")
    p_value = float((1 + np.sum(matched_rate.to_numpy() >= observed_rate)) / (len(matched_rate) + 1))
    result: Dict[str, Any] = {
        "benchmark": benchmark,
        "low_dif_reversal_rate": observed_rate,
        "original_random_median": float(original_rate.median()),
        "discrimination_matched_random_median": float(matched_rate.median()),
        "original_excess_points": 100.0 * (observed_rate - float(original_rate.median())),
        "discrimination_matched_excess_points": 100.0 * (observed_rate - float(matched_rate.median())),
        "discrimination_matched_p": p_value,
        "replicates": int(len(matched_rate)),
    }
    diagnostics = (
        "mean_spectral_discrimination",
        "mean_corrected_item_total_correlation",
        "heldout_independent_item_score_sem",
        "weighted_cronbach_alpha",
        "effective_item_count",
    )
    for metric in diagnostics:
        observed_value = float(observed[metric].mean())
        original_values = pooled_metric(original, metric)
        matched_values = pooled_metric(matched, metric)
        result["low_dif_" + metric] = observed_value
        result["original_random_median_" + metric] = float(original_values.median())
        result["discrimination_matched_median_" + metric] = float(matched_values.median())
        result["discrimination_matched_ratio_" + metric] = (
            observed_value / float(matched_values.median())
            if float(matched_values.median()) != 0.0
            else float("nan")
        )
    return result


def reconstruct_fold(
    benchmark_key: str,
    fold_index: int,
    run_dir: Path,
    data: Dict[str, Any],
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    audit_half, target_half = FOLDS[fold_index]
    family_models = pd.read_csv(run_dir / "family_models.csv")
    representatives = pd.read_csv(run_dir / "owner_family_representatives.csv")
    dimensions = pd.read_csv(run_dir / "dimension_selection.csv")
    dimension = int(dimensions.loc[dimensions["audit_half"].eq(audit_half), "selected_dimension"].iloc[0])
    family_names = sorted(family_models["family"].unique())
    family_to_code = {name: index for index, name in enumerate(family_names)}
    audit = family_models[family_models["model_half"].eq(audit_half)].copy()
    responses = np.asarray(data["responses"], dtype=np.int8)
    audit_responses = responses[:, audit["dataset_model_index"].to_numpy(dtype=np.int32)]
    frozen = pd.read_csv(run_dir / "crossfit_anchor_items.csv")
    frozen = frozen[
        np.isclose(frozen["anchor_fraction"], PRIMARY_FRACTION)
        & frozen["audit_half"].eq(audit_half)
    ].sort_values("dataset_item_index")
    expected_index = np.arange(len(data["sources"]), dtype=np.int64)
    if not np.array_equal(frozen["dataset_item_index"].to_numpy(dtype=np.int64), expected_index):
        raise AssertionError("frozen item table is not dense")
    selected = frozen["selected_anchor"].to_numpy(dtype=bool)
    weights = frozen["anchor_weight"].to_numpy(dtype=np.float64)
    offsets, model_axes, item_loadings = fit_spectral_mirt_offsets(
        audit_responses,
        selected,
        dimension,
        SVD_SEED + fold_index * 1000 + 100 + 3,
    )
    saved_coordinates = pd.read_csv(run_dir / "crossfit_model_coordinates.csv")
    saved_coordinates = saved_coordinates[
        saved_coordinates["audit_half"].eq(audit_half)
    ].sort_values(["model_id", "axis"])
    reconstructed_coordinates = (
        pd.DataFrame(
            model_axes,
            index=audit["model_id"].astype(str),
            columns=np.arange(1, dimension + 1),
        )
        .rename_axis(index="model_id", columns="axis")
        .stack()
        .rename("coordinate")
        .reset_index()
        .sort_values(["model_id", "axis"])
    )
    if not np.allclose(
        reconstructed_coordinates["coordinate"].to_numpy(),
        saved_coordinates["coordinate"].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError("reconstructed spectral coordinates differ from frozen run")
    fit = SimpleNamespace(
        selected=selected,
        weights=weights,
        cells=frozen["difficulty_cell"].to_numpy(dtype=str),
        model_axes=model_axes,
        item_loadings=item_loadings,
        offsets=offsets,
    )
    target = representatives[representatives["model_half"].eq(target_half)].copy()
    target_responses = responses[:, target["dataset_model_index"].to_numpy(dtype=np.int32)]
    return fit, audit, target, audit_responses, target_responses


def run_benchmark(key: str, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label = BENCHMARKS[key]
    run_dir = args.primary_root / (key + "_family_ranking_impact")
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    data = mirt_primary.load_data(key)
    observed_rows = []
    original_rows = []
    matched_rows = []
    item_rows = []
    frozen_original = pd.read_csv(run_dir / "matched_random_anchor_controls.csv")

    for fold_index, (audit_half, target_half) in enumerate(FOLDS):
        print("%s: reconstructing audit=%s" % (label, audit_half), flush=True)
        fit, audit, target, audit_responses, target_responses = reconstruct_fold(
            key, fold_index, run_dir, data
        )
        probability, _ = smoothed_item_parameters(audit_responses)
        discrimination = fit.offsets.std(axis=1, ddof=0)
        audit_citc = corrected_item_total_correlation(audit_responses)
        disc_cells = discrimination_cells(
            fit.cells, discrimination, args.max_discrimination_strata
        )
        families = target["family"].to_numpy()
        owners = target["owner"].to_numpy()
        target_responses = np.asarray(target_responses, dtype=np.float64)
        full_scores = target_responses.mean(axis=0)
        target_item_variance = target_responses.var(axis=1, ddof=1)
        target_citc = corrected_item_total_correlation(target_responses)
        for index in range(len(discrimination)):
            item_rows.append(
                {
                    "benchmark": label,
                    "audit_half": audit_half,
                    "target_half": target_half,
                    "dataset_item_index": index,
                    "blueprint_cell": fit.cells[index],
                    "discrimination_cell": disc_cells[index],
                    "selected_low_dif": bool(fit.selected[index]),
                    "anchor_weight": float(fit.weights[index]),
                    "spectral_predicted_logit_sd": float(discrimination[index]),
                    "audit_corrected_item_total_correlation": float(audit_citc[index]),
                    "heldout_corrected_item_total_correlation": float(target_citc[index]),
                    "audit_success_probability": float(probability[index]),
                    "heldout_item_response_variance": float(target_item_variance[index]),
                }
            )
        observed = evaluate_weights(
            target_responses, full_scores, target_item_variance,
            fit.weights, families, owners,
            discrimination, target_citc,
        )
        observed_rows.append(
            {
                "benchmark": label,
                "audit_half": audit_half,
                "target_half": target_half,
                "control_type": "low_dif",
                **observed,
            }
        )

        original_rng = np.random.default_rng(ORIGINAL_SEED + fold_index)
        matched_rng = np.random.default_rng(DISCRIMINATION_SEED + fold_index)
        for replicate in range(args.replicates):
            original_weights = matched_random_anchor_weights(
                fit.cells, fit.selected, original_rng
            )
            matched_weights = discrimination_matched_weights(
                fit.cells, disc_cells, fit.selected, matched_rng
            )
            original_rows.append(
                {
                    "benchmark": label,
                    "audit_half": audit_half,
                    "target_half": target_half,
                    "replicate": replicate,
                    "control_type": "source_easiness_matched",
                    **evaluate_weights(
                        target_responses, full_scores, target_item_variance,
                        original_weights, families, owners,
                        discrimination, target_citc,
                    ),
                }
            )
            matched_rows.append(
                {
                    "benchmark": label,
                    "audit_half": audit_half,
                    "target_half": target_half,
                    "replicate": replicate,
                    "control_type": "source_easiness_discrimination_matched",
                    **evaluate_weights(
                        target_responses, full_scores, target_item_variance,
                        matched_weights, families, owners,
                        discrimination, target_citc,
                    ),
                }
            )
            if (replicate + 1) % 250 == 0:
                print(
                    "%s audit=%s controls=%d/%d"
                    % (label, audit_half, replicate + 1, args.replicates),
                    flush=True,
                )

        reproduced = pd.DataFrame(original_rows)
        reproduced = reproduced[
            reproduced["benchmark"].eq(label)
            & reproduced["audit_half"].eq(audit_half)
        ].sort_values("replicate")
        frozen = frozen_original[
            frozen_original["audit_half"].eq(audit_half)
        ].sort_values("replicate")
        for column in (
            "close_cross_family_flips",
            "close_cross_family_eligible_pairs",
        ):
            if not np.array_equal(
                reproduced[column].to_numpy(dtype=np.int64),
                frozen[column].to_numpy(dtype=np.int64),
            ):
                raise AssertionError("original random controls were not reproduced: " + column)

    return (
        pd.DataFrame(observed_rows),
        pd.DataFrame(original_rows),
        pd.DataFrame(matched_rows),
        pd.DataFrame(item_rows),
    )


def markdown_report(summary: pd.DataFrame) -> str:
    lines = [
        "# Discrimination-matched low-DIF ablation",
        "",
        "All values use the frozen extended-K owner halves, dimensions, 50% anchors,",
        "ranking representatives, close-pair definition, and source-by-easiness weights.",
        "Discrimination is the audit-half standard deviation of the family-blind",
        "spectral predicted-logit offset. The new null additionally matches the exact",
        "number of selected anchors in each within-blueprint discrimination quartile.",
        "",
        "| Benchmark | Low-DIF | Original random | Disc.-matched random | Disc.-matched excess (pp) | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| %s | %.1f%% | %.1f%% | %.1f%% | %+.1f | %.3f |"
            % (
                row.benchmark,
                100.0 * row.low_dif_reversal_rate,
                100.0 * row.original_random_median,
                100.0 * row.discrimination_matched_random_median,
                row.discrimination_matched_excess_points,
                row.discrimination_matched_p,
            )
        )
    lines.extend(
        [
            "",
            "## Matching and held-out reliability diagnostics",
            "",
            "| Benchmark | g: low / original / matched | CITC: low / original / matched | SEM: low / original / matched | alpha: low / original / matched |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            "| %s | %.3f / %.3f / %.3f | %.3f / %.3f / %.3f | %.4f / %.4f / %.4f | %.4f / %.4f / %.4f |"
            % (
                row.benchmark,
                row.low_dif_mean_spectral_discrimination,
                row.original_random_median_mean_spectral_discrimination,
                row.discrimination_matched_median_mean_spectral_discrimination,
                row.low_dif_mean_corrected_item_total_correlation,
                row.original_random_median_mean_corrected_item_total_correlation,
                row.discrimination_matched_median_mean_corrected_item_total_correlation,
                row.low_dif_heldout_independent_item_score_sem,
                row.original_random_median_heldout_independent_item_score_sem,
                row.discrimination_matched_median_heldout_independent_item_score_sem,
                row.low_dif_weighted_cronbach_alpha,
                row.original_random_median_weighted_cronbach_alpha,
                row.discrimination_matched_median_weighted_cronbach_alpha,
            )
        )
    lines.extend(
        [
            "",
            "The control is diagnostic rather than a claim that spectral predicted-logit",
            "spread exhausts every notion of item discrimination or test information.",
        ]
    )
    return "\n".join(lines) + "\n"


def latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Discrimination-matched negative control. $g_i$ is the audit-half standard deviation of the family-blind predicted-logit offset; CITC, score SEM, and weighted Cronbach's $\alpha$ are evaluated on the owner-disjoint target half. Original controls match source, easiness, and length; discrimination-matched controls additionally match the low-DIF count within audit-half $g_i$ quartiles. Entries for diagnostics are low-DIF / original-control median / discrimination-matched-control median.}",
        r"\label{tab:discrimination-matched-control}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Benchmark & $g_i$ & Held-out CITC & Held-out SEM & Held-out $\alpha$ \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "%s & %.3f / %.3f / %.3f & %.3f / %.3f / %.3f & %.4f / %.4f / %.4f & %.4f / %.4f / %.4f \\\\"
            % (
                row.benchmark,
                row.low_dif_mean_spectral_discrimination,
                row.original_random_median_mean_spectral_discrimination,
                row.discrimination_matched_median_mean_spectral_discrimination,
                row.low_dif_mean_corrected_item_total_correlation,
                row.original_random_median_mean_corrected_item_total_correlation,
                row.discrimination_matched_median_mean_corrected_item_total_correlation,
                row.low_dif_heldout_independent_item_score_sem,
                row.original_random_median_heldout_independent_item_score_sem,
                row.discrimination_matched_median_heldout_independent_item_score_sem,
                row.low_dif_weighted_cronbach_alpha,
                row.original_random_median_weighted_cronbach_alpha,
                row.discrimination_matched_median_weighted_cronbach_alpha,
            )
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{5}{l}{\textit{Ranking consequence under the discrimination-matched control}} \\",
            r"\midrule",
            r"Benchmark & Low-DIF reversal & Matched median & Excess (pp) & $p$ \\",
            r"\midrule",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            "%s & %.1f\\%% & %.1f\\%% & %+.1f & %.3f \\\\"
            % (
                row.benchmark,
                100.0 * row.low_dif_reversal_rate,
                100.0 * row.discrimination_matched_random_median,
                row.discrimination_matched_excess_points,
                row.discrimination_matched_p,
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    args = parse_args(argv)
    keys = tuple(value.strip() for value in args.benchmarks.split(",") if value.strip())
    unknown = set(keys).difference(BENCHMARKS)
    if unknown or not keys or len(keys) != len(set(keys)):
        raise ValueError("supply distinct, known benchmarks: %s" % sorted(unknown))
    if args.replicates != REPLICATES:
        raise ValueError("formal ablation freezes 1,000 control replicates")
    if args.max_discrimination_strata != 4:
        raise ValueError("formal ablation freezes four discrimination strata")
    require_primary_files(args.primary_root, keys, reconstruct=True)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output is not empty; choose a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    all_observed = []
    all_original = []
    all_matched = []
    all_items = []
    summaries = []
    for key in keys:
        observed, original, matched, items = run_benchmark(key, args)
        all_observed.append(observed)
        all_original.append(original)
        all_matched.append(matched)
        all_items.append(items)
        summaries.append(
            summarize_benchmark(BENCHMARKS[key], observed, original, matched)
        )
        pd.DataFrame(summaries).to_csv(args.output_dir / "summary.partial.csv", index=False)

    observed = pd.concat(all_observed, ignore_index=True)
    original = pd.concat(all_original, ignore_index=True)
    matched = pd.concat(all_matched, ignore_index=True)
    items = pd.concat(all_items, ignore_index=True)
    summary = pd.DataFrame(summaries)
    observed.to_csv(args.output_dir / "low_dif_fold_diagnostics.csv", index=False)
    original.to_csv(args.output_dir / "original_matched_random_controls.csv", index=False)
    matched.to_csv(args.output_dir / "discrimination_matched_random_controls.csv", index=False)
    items.to_csv(args.output_dir / "item_discrimination_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    (args.output_dir / "appendix_table.tex").write_text(
        latex_table(summary), encoding="utf-8"
    )
    protocol = {
        "experiment": "extended_k_discrimination_matched_negative_control",
        "benchmarks": [BENCHMARKS[key] for key in keys],
        "primary_anchor_fraction": PRIMARY_FRACTION,
        "discrimination": "audit-half SD across models of family-blind spectral predicted-logit offsets",
        "heldout_diagnostics": [
            "owner-disjoint target-half corrected item-total correlation",
            "owner-disjoint target-half independent-item score SEM",
            "owner-disjoint target-half weighted Cronbach alpha",
        ],
        "matching": "exact selected count within source x easiness x discrimination quartile; original source x easiness weights retained",
        "replicates": args.replicates,
        "original_control_seed_base": ORIGINAL_SEED,
        "discrimination_control_seed_base": DISCRIMINATION_SEED,
        "owner_disjoint": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    partial = args.output_dir / "summary.partial.csv"
    if partial.exists():
        partial.unlink()
    print(summary.to_string(index=False), flush=True)
    print("written to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
