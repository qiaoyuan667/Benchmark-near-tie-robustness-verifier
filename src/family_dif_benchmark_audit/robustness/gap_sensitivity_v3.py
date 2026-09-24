#!/usr/bin/env python3
"""V3 score-gap sensitivity for the spectral-MIRT-primary analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from ..ranking.routereval import (
    DEFAULT_ITEMS,
    DEFAULT_MATRIX,
    DEFAULT_MODELS,
    composition_bootstrap_units,
    load_benchmark_inputs,
)
from ..ranking.mmlu_pro import (
    matched_random_anchor_weights,
    source_aggregates,
    weighted_accuracy,
)
from ..discovery.qmirt_minimal import write_json


from .._paths import PROJECT_ROOT


HERE = PROJECT_ROOT
OUTPUT = HERE / "outputs/v3_ranking_gap_sensitivity"
GAPS = np.asarray([0.0025, 0.005, 0.01, 0.02, 0.03, 0.05], dtype=np.float64)
PRIMARY_FRACTION = 0.5
BOOTSTRAP_REPLICATES = 2000
CONTROL_REPLICATES = 1000
RUNS: Mapping[str, Dict[str, Any]] = {
    "MMLU-Pro": {
        "kind": "mmlu_pro",
        "path": HERE / "outputs/mmlu_pro_family_ranking_impact",
        "responses": HERE
        / "outputs/routereval_raw_matrices/mmlu_pro_response_matrix.npz",
    },
    "BBH": {
        "kind": "routereval",
        "benchmark": "bbh",
        "path": HERE / "outputs/bbh_family_ranking_impact",
    },
    "MMLU": {
        "kind": "routereval",
        "benchmark": "mmlu",
        "path": HERE / "outputs/mmlu_family_ranking_impact",
    },
    "HellaSwag": {
        "kind": "routereval",
        "benchmark": "hellaswag",
        "path": HERE / "outputs/hellaswag_family_ranking_impact",
    },
    "WinoGrande": {
        "kind": "routereval",
        "benchmark": "winogrande",
        "path": HERE / "outputs/winogrande_family_ranking_impact",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def load_responses(config: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    if config["kind"] == "mmlu_pro":
        with np.load(config["responses"], allow_pickle=False) as archive:
            responses = archive["responses_q_by_m"].astype(np.int8)
            model_ids = archive["model_ids"].astype(str)
        return responses, model_ids
    data = load_benchmark_inputs(
        DEFAULT_MATRIX,
        DEFAULT_ITEMS,
        DEFAULT_MODELS,
        str(config["benchmark"]),
    )
    return (
        np.asarray(data["responses"], dtype=np.int8),
        np.asarray(data["model_ids"]).astype(str),
    )


def pair_context(
    observed_full: np.ndarray,
    families: np.ndarray,
    owners: np.ndarray,
) -> Dict[str, np.ndarray]:
    left, right = np.triu_indices(len(observed_full), 1)
    eligible = owners[left] != owners[right]
    cross_family = eligible & (families[left] != families[right])
    observed_gap = np.abs(observed_full[left] - observed_full[right])
    return {
        "left": left,
        "right": right,
        "cross_family": cross_family,
        "observed_gap": observed_gap,
    }


def gap_curve_counts(
    full_scores: np.ndarray,
    alternative_scores: np.ndarray,
    context: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    left = context["left"]
    right = context["right"]
    full_order = np.sign(full_scores[left] - full_scores[right])
    alternative_order = np.sign(
        alternative_scores[left] - alternative_scores[right]
    )
    non_tied = (full_order != 0) & (alternative_order != 0)
    reversed_pair = full_order * alternative_order < 0
    rows: List[Dict[str, Any]] = []
    for gap in GAPS:
        denominator = (
            context["cross_family"]
            & (context["observed_gap"] <= gap + 1e-15)
            & non_tied
        )
        pairs = int(denominator.sum())
        flips = int((denominator & reversed_pair).sum())
        rows.append(
            {
                "gap": float(gap),
                "gap_points": float(100.0 * gap),
                "eligible_pairs": pairs,
                "flips": flips,
                "flip_rate": float(flips / pairs) if pairs else float("nan"),
            }
        )
    return rows


def bootstrap_curves(
    responses: np.ndarray,
    bootstrap_units: np.ndarray,
    anchor_weights: np.ndarray,
    families: np.ndarray,
    owners: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    (
        source_names,
        full_success,
        full_mass,
        anchor_success,
        anchor_mass,
    ) = source_aggregates(responses, bootstrap_units, anchor_weights)
    observed_full = responses.mean(axis=0, dtype=np.float64)
    context = pair_context(observed_full, families, owners)
    rng = np.random.default_rng(seed)
    source_weights = rng.multinomial(
        len(source_names),
        np.full(len(source_names), 1.0 / len(source_names)),
        size=replicates,
    ).astype(np.float64)
    full_scores = (source_weights @ full_success) / (
        source_weights @ full_mass
    )[:, None]
    anchor_scores = (source_weights @ anchor_success) / (
        source_weights @ anchor_mass
    )[:, None]
    rows: List[Dict[str, Any]] = []
    for replicate in range(replicates):
        for row in gap_curve_counts(
            full_scores[replicate], anchor_scores[replicate], context
        ):
            rows.append({"replicate": replicate, **row})
    return pd.DataFrame(rows)


def random_control_curves(
    responses: np.ndarray,
    cells: np.ndarray,
    selected: np.ndarray,
    families: np.ndarray,
    owners: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    observed_full = responses.mean(axis=0, dtype=np.float64)
    context = pair_context(observed_full, families, owners)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for replicate in range(replicates):
        random_weights = matched_random_anchor_weights(cells, selected, rng)
        random_scores = weighted_accuracy(responses, random_weights)
        for row in gap_curve_counts(observed_full, random_scores, context):
            rows.append({"replicate": replicate, **row})
    return pd.DataFrame(rows)


def fold_inputs(
    benchmark: str,
    config: Mapping[str, Any],
    responses: np.ndarray,
    model_ids: np.ndarray,
    audit_half: str,
) -> Dict[str, Any]:
    path = Path(config["path"])
    target_half = "validation" if audit_half == "discovery" else "discovery"
    anchors = pd.read_csv(path / "crossfit_anchor_items.csv")
    anchors = anchors[
        np.isclose(anchors["anchor_fraction"], PRIMARY_FRACTION)
        & anchors["audit_half"].eq(audit_half)
    ].sort_values("dataset_item_index")
    expected_items = np.arange(responses.shape[0])
    if not np.array_equal(
        anchors["dataset_item_index"].to_numpy(dtype=np.int64), expected_items
    ):
        raise ValueError("%s %s anchor rows are not dense" % (benchmark, audit_half))
    representatives = pd.read_csv(path / "owner_family_representatives.csv")
    representatives = representatives[
        representatives["model_half"].eq(target_half)
    ].copy()
    indices = representatives["dataset_model_index"].to_numpy(dtype=np.int32)
    observed_ids = model_ids[indices]
    expected_ids = representatives["model_id"].astype(str).to_numpy()
    if not np.array_equal(observed_ids, expected_ids):
        raise ValueError("%s representative model indices do not align" % benchmark)
    target_responses = responses[:, indices]
    weights = anchors["anchor_weight"].to_numpy(dtype=np.float64)
    full_scores = target_responses.mean(axis=0, dtype=np.float64)
    anchor_scores = weighted_accuracy(target_responses, weights)

    saved = pd.read_csv(path / "crossfit_model_scores.csv")
    saved = saved[
        np.isclose(saved["anchor_fraction"], PRIMARY_FRACTION)
        & saved["audit_half"].eq(audit_half)
    ].set_index("model_id")
    full_saved = saved.loc[expected_ids, "full_accuracy"].to_numpy(dtype=np.float64)
    anchor_saved = saved.loc[
        expected_ids, "invariant_anchor_accuracy"
    ].to_numpy(dtype=np.float64)
    if not np.allclose(full_scores, full_saved, rtol=0.0, atol=1e-12):
        raise ValueError("%s full scores do not reproduce" % benchmark)
    if not np.allclose(anchor_scores, anchor_saved, rtol=0.0, atol=1e-12):
        raise ValueError("%s anchor scores do not reproduce" % benchmark)

    sources = anchors["source"].astype(str).to_numpy()
    if benchmark == "MMLU-Pro":
        bootstrap_units = sources
        bootstrap_strategy = "item_source_block"
    else:
        bootstrap_units, bootstrap_strategy = composition_bootstrap_units(
            sources, 100
        )
    return {
        "target_half": target_half,
        "responses": target_responses,
        "families": representatives["family"].astype(str).to_numpy(),
        "owners": representatives["owner"].astype(str).to_numpy(),
        "cells": anchors["difficulty_cell"].astype(str).to_numpy(),
        "selected": anchors["selected_anchor"].astype(bool).to_numpy(),
        "weights": weights,
        "full_scores": full_scores,
        "anchor_scores": anchor_scores,
        "bootstrap_units": bootstrap_units,
        "bootstrap_strategy": bootstrap_strategy,
    }


def summarize_curves(
    observed: pd.DataFrame,
    bootstrap: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    observed_summary = observed.groupby(["benchmark", "gap_points"], as_index=False).agg(
        eligible_pairs=("eligible_pairs", "sum"),
        flips=("flips", "sum"),
    )
    observed_summary["observed_flip_rate"] = (
        observed_summary["flips"] / observed_summary["eligible_pairs"]
    )

    bootstrap_pooled = bootstrap.groupby(
        ["benchmark", "gap_points", "replicate"], as_index=False
    ).agg(flips=("flips", "sum"), eligible_pairs=("eligible_pairs", "sum"))
    bootstrap_pooled["flip_rate"] = (
        bootstrap_pooled["flips"] / bootstrap_pooled["eligible_pairs"]
    )
    bootstrap_summary = bootstrap_pooled.groupby(
        ["benchmark", "gap_points"]
    )["flip_rate"].agg(
        bootstrap_ci_low=lambda values: values.quantile(0.025),
        bootstrap_median="median",
        bootstrap_ci_high=lambda values: values.quantile(0.975),
    ).reset_index()

    control_pooled = controls.groupby(
        ["benchmark", "gap_points", "replicate"], as_index=False
    ).agg(flips=("flips", "sum"), eligible_pairs=("eligible_pairs", "sum"))
    control_pooled["flip_rate"] = (
        control_pooled["flips"] / control_pooled["eligible_pairs"]
    )
    control_summary = control_pooled.groupby(
        ["benchmark", "gap_points"]
    )["flip_rate"].agg(
        random_ci_low=lambda values: values.quantile(0.025),
        random_median="median",
        random_ci_high=lambda values: values.quantile(0.975),
    ).reset_index()

    summary = observed_summary.merge(
        bootstrap_summary, on=["benchmark", "gap_points"], validate="one_to_one"
    ).merge(
        control_summary, on=["benchmark", "gap_points"], validate="one_to_one"
    )
    control_lookup = control_pooled.groupby(["benchmark", "gap_points"])
    p_values: List[float] = []
    for row in summary.itertuples(index=False):
        values = control_lookup.get_group(
            (row.benchmark, row.gap_points)
        )["flip_rate"].to_numpy(dtype=np.float64)
        p_values.append(
            float((np.sum(values >= row.observed_flip_rate) + 1) / (len(values) + 1))
        )
    summary["one_sided_randomization_p"] = p_values
    summary["excess_flip_rate"] = (
        summary["observed_flip_rate"] - summary["random_median"]
    )
    return summary.sort_values(["benchmark", "gap_points"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    """Write a dependency-free vector figure suitable for the paper draft."""
    width, height = 1260, 760
    panel_width, panel_height = 360, 265
    left_margin, top_margin = 75, 95
    col_gap, row_gap = 45, 80
    plot_left, plot_top, plot_width, plot_height = 48, 30, 292, 190
    labels = ["0.25", "0.5", "1", "2", "3", "5"]
    observed_high_max = 100.0 * float(summary["bootstrap_ci_high"].max())
    random_high_max = 100.0 * float(summary["random_ci_high"].max())
    y_max = max(20.0, 10.0 * np.ceil(max(observed_high_max, random_high_max) / 10.0))

    def sx(index: int, panel_x: float) -> float:
        return panel_x + plot_left + index * plot_width / (len(labels) - 1)

    def sy(value: float, panel_y: float) -> float:
        return panel_y + plot_top + plot_height * (1.0 - value / y_max)

    def points(values: np.ndarray, panel_x: float, panel_y: float) -> str:
        return " ".join(
            "%.2f,%.2f" % (sx(index, panel_x), sy(float(value), panel_y))
            for index, value in enumerate(values)
        )

    elements: List[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
        % (width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124}.title{font-size:22px;font-weight:600}.panel-title{font-size:15px;font-weight:600}.tick{font-size:11px}.axis-label{font-size:12px}.legend{font-size:13px}</style>',
        '<text class="title" x="630" y="38" text-anchor="middle">Cumulative near-tie ranking sensitivity across score-gap thresholds</text>',
    ]
    order = list(RUNS)
    for panel_index, benchmark in enumerate(order):
        row, col = divmod(panel_index, 3)
        panel_x = left_margin + col * (panel_width + col_gap)
        panel_y = top_margin + row * (panel_height + row_gap)
        frame = summary[summary["benchmark"].eq(benchmark)].sort_values("gap_points")
        observed = 100.0 * frame["observed_flip_rate"].to_numpy()
        observed_low = 100.0 * frame["bootstrap_ci_low"].to_numpy()
        observed_high = 100.0 * frame["bootstrap_ci_high"].to_numpy()
        random = 100.0 * frame["random_median"].to_numpy()
        random_low = 100.0 * frame["random_ci_low"].to_numpy()
        random_high = 100.0 * frame["random_ci_high"].to_numpy()
        elements.append(
            '<text class="panel-title" x="%.2f" y="%.2f" text-anchor="middle">%s</text>'
            % (panel_x + plot_left + plot_width / 2, panel_y + 12, benchmark)
        )
        for tick in np.arange(0.0, y_max + 0.1, 10.0):
            y = sy(float(tick), panel_y)
            elements.append(
                '<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#e1e4e8" stroke-width="1"/>'
                % (panel_x + plot_left, y, panel_x + plot_left + plot_width, y)
            )
            elements.append(
                '<text class="tick" x="%.2f" y="%.2f" text-anchor="end">%.0f</text>'
                % (panel_x + plot_left - 7, y + 4, tick)
            )
        elements.append(
            '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="none" stroke="#9aa0a6" stroke-width="1"/>'
            % (panel_x + plot_left, panel_y + plot_top, plot_width, plot_height)
        )
        for index, label in enumerate(labels):
            elements.append(
                '<text class="tick" x="%.2f" y="%.2f" text-anchor="middle">%s</text>'
                % (sx(index, panel_x), panel_y + plot_top + plot_height + 18, label)
            )
        observed_band_points = " ".join(
            [
                "%.2f,%.2f" % (sx(index, panel_x), sy(float(value), panel_y))
                for index, value in enumerate(observed_high)
            ]
            + [
                "%.2f,%.2f" % (sx(index, panel_x), sy(float(value), panel_y))
                for index, value in zip(range(len(labels) - 1, -1, -1), observed_low[::-1])
            ]
        )
        random_band_points = " ".join(
            [
                "%.2f,%.2f" % (sx(index, panel_x), sy(float(value), panel_y))
                for index, value in enumerate(random_high)
            ]
            + [
                "%.2f,%.2f" % (sx(index, panel_x), sy(float(value), panel_y))
                for index, value in zip(range(len(labels) - 1, -1, -1), random_low[::-1])
            ]
        )
        elements.extend(
            [
                '<polygon points="%s" fill="#2f6f9f" fill-opacity="0.16"/>' % observed_band_points,
                '<polygon points="%s" fill="#777777" fill-opacity="0.13"/>' % random_band_points,
                '<polyline points="%s" fill="none" stroke="#2f6f9f" stroke-width="2.5"/>'
                % points(observed, panel_x, panel_y),
                '<polyline points="%s" fill="none" stroke="#666666" stroke-width="2" stroke-dasharray="6 4"/>'
                % points(random, panel_x, panel_y),
            ]
        )
        for index, value in enumerate(observed):
            elements.append(
                '<circle cx="%.2f" cy="%.2f" r="3.5" fill="#2f6f9f"/>'
                % (sx(index, panel_x), sy(float(value), panel_y))
            )
        for index, value in enumerate(random):
            x, y = sx(index, panel_x), sy(float(value), panel_y)
            elements.append(
                '<rect x="%.2f" y="%.2f" width="6" height="6" fill="#666666"/>'
                % (x - 3, y - 3)
            )
        elements.append(
            '<text class="axis-label" x="%.2f" y="%.2f" text-anchor="middle">Maximum full-score gap (pp)</text>'
            % (panel_x + plot_left + plot_width / 2, panel_y + plot_top + plot_height + 38)
        )
        if col == 0:
            elements.append(
                '<text class="axis-label" transform="translate(%.2f %.2f) rotate(-90)" text-anchor="middle">Cross-family rank flips (%%)</text>'
                % (panel_x + 9, panel_y + plot_top + plot_height / 2)
            )
    legend_x, legend_y = 905, 505
    elements.extend(
        [
            '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#2f6f9f" stroke-width="3"/>'
            % (legend_x, legend_y, legend_x + 45, legend_y),
            '<circle cx="%d" cy="%d" r="4" fill="#2f6f9f"/>'
            % (legend_x + 22, legend_y),
            '<text class="legend" x="%d" y="%d">Low-DIF anchor score (composition-resampling 95%% percentile range)</text>'
            % (legend_x + 58, legend_y + 5),
            '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#666666" stroke-width="2" stroke-dasharray="6 4"/>'
            % (legend_x, legend_y + 34, legend_x + 45, legend_y + 34),
            '<rect x="%d" y="%d" width="7" height="7" fill="#666666"/>'
            % (legend_x + 19, legend_y + 30),
            '<text class="legend" x="%d" y="%d">Matched-random subtest (95%% interval)</text>'
            % (legend_x + 58, legend_y + 39),
            "</svg>",
        ]
    )
    (output / "figure_gap_sensitivity.svg").write_text(
        "\n".join(elements) + "\n", encoding="utf-8"
    )


def build_report(summary: pd.DataFrame) -> str:
    lines = [
        "# Mandatory experiment A: score-gap sensitivity",
        "",
        "Each point is cumulative over model pairs whose full-score gap is less "
        "than or equal to the displayed threshold. Low-DIF and matched-random "
        "subtests use the same item count and native "
        "source-by-easiness blueprint. Composition-bootstrap intervals resample "
        "native source blocks, except HellaSwag and WinoGrande, which use the frozen "
        "100 contiguous item clusters. Pair sets are frozen from the observed "
        "full-benchmark score.",
        "",
        "| benchmark | gap | eligible pairs | low-DIF flips | composition-resampling 95% percentile range | random median | random 95% interval | excess | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| %s | %.2f pp | %d | %.1f%% | [%.1f%%, %.1f%%] | %.1f%% | [%.1f%%, %.1f%%] | %+.1f pp | %.4f |"
            % (
                row.benchmark,
                row.gap_points,
                row.eligible_pairs,
                100.0 * row.observed_flip_rate,
                100.0 * row.bootstrap_ci_low,
                100.0 * row.bootstrap_ci_high,
                100.0 * row.random_median,
                100.0 * row.random_ci_low,
                100.0 * row.random_ci_high,
                100.0 * row.excess_flip_rate,
                row.one_sided_randomization_p,
            )
        )
    lines.extend(
        [
            "",
            "The primary 1 pp row exactly reproduces the five-family main "
            "MIRT-primary analysis. The curve is descriptive across thresholds; "
            "the confirmatory threshold remains the frozen 1 pp band.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    observed_parts: List[pd.DataFrame] = []
    bootstrap_parts: List[pd.DataFrame] = []
    control_parts: List[pd.DataFrame] = []
    run_metadata: Dict[str, Any] = {}

    for benchmark, config in RUNS.items():
        print("load", benchmark, flush=True)
        responses, model_ids = load_responses(config)
        with (Path(config["path"]) / "protocol.json").open(encoding="utf-8") as handle:
            protocol = json.load(handle)
        control_seed = int(
            protocol.get("matched_random_anchor_control", {}).get(
                "seed", protocol.get("matched_random_anchor_seed", 20260825)
            )
        )
        bootstrap_seed = int(
            protocol.get("bootstrap", {}).get(
                "seed", protocol.get("bootstrap_seed", 20260824)
            )
        ) + 100
        fold_strategies: Dict[str, str] = {}
        for fold_index, audit_half in enumerate(("discovery", "validation")):
            print("  curves", audit_half, flush=True)
            fold = fold_inputs(
                benchmark, config, responses, model_ids, audit_half
            )
            fold_strategies[audit_half] = str(fold["bootstrap_strategy"])
            context = pair_context(
                fold["full_scores"], fold["families"], fold["owners"]
            )
            observed = pd.DataFrame(
                gap_curve_counts(
                    fold["full_scores"], fold["anchor_scores"], context
                )
            )
            observed["benchmark"] = benchmark
            observed["audit_half"] = audit_half
            observed["target_half"] = fold["target_half"]
            observed_parts.append(observed)

            controls = random_control_curves(
                fold["responses"],
                fold["cells"],
                fold["selected"],
                fold["families"],
                fold["owners"],
                CONTROL_REPLICATES,
                control_seed + (0 if fold["target_half"] == "validation" else 1),
            )
            controls["benchmark"] = benchmark
            controls["audit_half"] = audit_half
            controls["target_half"] = fold["target_half"]
            control_parts.append(controls)

            bootstrap = bootstrap_curves(
                fold["responses"],
                fold["bootstrap_units"],
                fold["weights"],
                fold["families"],
                fold["owners"],
                BOOTSTRAP_REPLICATES,
                bootstrap_seed,
            )
            bootstrap["benchmark"] = benchmark
            bootstrap["audit_half"] = audit_half
            bootstrap["target_half"] = fold["target_half"]
            bootstrap_parts.append(bootstrap)
        run_metadata[benchmark] = {
            "items": int(responses.shape[0]),
            "models_in_response_matrix": int(responses.shape[1]),
            "bootstrap_strategy_by_audit_half": fold_strategies,
            "control_seed": control_seed,
            "bootstrap_seed_for_primary_fraction": bootstrap_seed,
        }

    observed = pd.concat(observed_parts, ignore_index=True)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    controls = pd.concat(control_parts, ignore_index=True)
    summary = summarize_curves(observed, bootstrap, controls)

    frozen = pd.read_csv(
        HERE / "outputs/v3_five_benchmark_mirt_primary/benchmark_summary.csv"
    ).set_index("benchmark")
    primary = summary[np.isclose(summary["gap_points"], 1.0)].set_index("benchmark")
    for benchmark in RUNS:
        if not np.isclose(
            primary.loc[benchmark, "observed_flip_rate"],
            frozen.loc[benchmark, "close_cross_family_flip_rate"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError("%s 1pp observed rate failed to reproduce" % benchmark)
        if not np.isclose(
            primary.loc[benchmark, "random_median"],
            frozen.loc[benchmark, "random_control_close_median"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError("%s 1pp random control failed to reproduce" % benchmark)

    observed.to_csv(OUTPUT / "observed_gap_curves_by_fold.csv", index=False)
    bootstrap.to_csv(OUTPUT / "composition_bootstrap_gap_curves_by_fold.csv", index=False)
    controls.to_csv(OUTPUT / "matched_random_gap_curves_by_fold.csv", index=False)
    summary.to_csv(OUTPUT / "gap_sensitivity_summary.csv", index=False)
    write_json(
        OUTPUT / "protocol.json",
        {
            "experiment": "spectral_mirt_primary_score_gap_sensitivity",
            "gaps_accuracy_units": GAPS.tolist(),
            "gaps_percentage_points": (100.0 * GAPS).tolist(),
            "primary_gap_percentage_points": 1.0,
            "primary_anchor_fraction": PRIMARY_FRACTION,
            "composition_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "matched_random_control_replicates": CONTROL_REPLICATES,
            "pair_set": "cross-family, cross-owner, frozen from observed full-benchmark score",
            "runs": run_metadata,
        },
    )
    plot_summary(summary, OUTPUT)
    (OUTPUT / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("written to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
