#!/usr/bin/env python3
"""Validate final result claims and render paper-ready vector figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

BENCHMARKS = ["MMLU-Pro", "BBH", "MMLU", "HellaSwag", "WinoGrande"]
GAPS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]


def hex_color(value: str) -> colors.Color:
    value = value.lstrip("#")
    return colors.Color(
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


SERIES = {
    "MMLU-Pro": (hex_color("#0072B2"), "circle"),
    "BBH": (hex_color("#D55E00"), "square"),
    "MMLU": (hex_color("#009E73"), "triangle"),
    "HellaSwag": (hex_color("#CC79A7"), "diamond"),
    "WinoGrande": (hex_color("#666666"), "cross"),
}


def assert_close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"expected {expected}, observed {actual}")


def validate_claims() -> dict:
    main = pd.read_csv(
        OUTPUTS / "primary" / "benchmark_summary.csv"
    ).set_index("benchmark")
    gap = pd.read_csv(
        OUTPUTS / "gap_sensitivity" / "gap_sensitivity_summary.csv"
    )
    robustness = pd.read_csv(
        OUTPUTS / "population_robustness" / "robustness_summary.csv"
    )
    stability = pd.read_csv(
        OUTPUTS / "item_stability" / "stability_summary.csv"
    ).set_index("benchmark")
    source = pd.read_csv(
        OUTPUTS
        / "item_stability"
        / "source_validation_summary.csv"
    ).set_index("benchmark")
    direct_source = pd.read_csv(
        OUTPUTS
        / "item_stability"
        / "source_shift_driver_validation_summary.csv"
    ).set_index("benchmark")
    binary = pd.read_csv(
        OUTPUTS / "content_audit" / "paired_binary_enrichment.csv"
    )
    binary_reliability = pd.read_csv(
        OUTPUTS / "content_audit" / "binary_annotation_reliability.csv"
    )
    ordinal_reliability = pd.read_csv(
        OUTPUTS / "content_audit" / "ordinal_annotation_reliability.csv"
    )
    content_protocol = json.loads(
        (
            OUTPUTS / "content_audit" / "protocol.json"
        ).read_text(encoding="utf-8")
    )
    post_annotation_audit = json.loads(
        (
            OUTPUTS / "content_audit" / "post_annotation_audit.json"
        ).read_text(encoding="utf-8")
    )

    expected_main = {
        "MMLU-Pro": (0.923587, 0.470779, 0.185065, 0.285714, 0.000999),
        "BBH": (0.915300, 0.421384, 0.252471, 0.168913, 0.000999),
        "MMLU": (0.930019, 0.406801, 0.163215, 0.243586, 0.000999),
        "HellaSwag": (0.948326, 0.309287, 0.114090, 0.195197, 0.000999),
        "WinoGrande": (0.899644, 0.338177, 0.346926, -0.008749, 0.689311),
    }
    for benchmark, expected in expected_main.items():
        row = main.loc[benchmark]
        observed = (
            row["mean_kendall_tau_b"],
            row["close_cross_family_flip_rate"],
            row["random_control_close_median"],
            row["excess_close_flip_rate"],
            row["random_control_close_p"],
        )
        for actual, target in zip(observed, expected):
            assert_close(float(actual), float(target), tolerance=5e-6)

    assert_close(float(main["mean_kendall_tau_b"].min()), 0.899644, 5e-6)
    assert_close(float(main["mean_kendall_tau_b"].max()), 0.948326, 5e-6)
    assert_close(float(main["all_cross_family_flip_rate"].min()), 0.024578, 5e-6)
    assert_close(float(main["all_cross_family_flip_rate"].max()), 0.042449, 5e-6)

    primary_gap = gap[np.isclose(gap["gap_points"], 1.0)].set_index("benchmark")
    for benchmark in BENCHMARKS:
        assert_close(
            float(primary_gap.loc[benchmark, "observed_flip_rate"]),
            float(main.loc[benchmark, "close_cross_family_flip_rate"]),
            1e-12,
        )
        assert_close(
            float(primary_gap.loc[benchmark, "random_median"]),
            float(main.loc[benchmark, "random_control_close_median"]),
            1e-12,
        )

    variants_per_benchmark = robustness.groupby("benchmark")["variant"].nunique()
    if not (variants_per_benchmark == 10).all():
        raise AssertionError("robustness matrix is not 10 by 5")
    primary_positive = ["MMLU-Pro", "BBH", "MMLU", "HellaSwag"]
    robust_primary = robustness[robustness["benchmark"].isin(primary_positive)]
    if not robust_primary["significant_positive_excess"].all():
        raise AssertionError("a primary-positive robustness cell failed")
    wino_significant = int(
        robustness[robustness["benchmark"].eq("WinoGrande")][
            "significant_positive_excess"
        ].sum()
    )
    if wino_significant != 1:
        raise AssertionError("WinoGrande significant robustness count changed")

    if not stability["passes_all_stability_gates"].all():
        raise AssertionError("an item-stability gate failed")
    if not np.allclose(
        stability["family_within_cell_permutation_p"].to_numpy(dtype=float),
        1.0 / 501.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("item-stability permutation p-values changed")
    if int(source.loc["Pooled source-rich main", "validation_sign_replicated"]) != 58:
        raise AssertionError("source-effect sign replication count changed")
    if int(direct_source.loc["Pooled source-rich main", "validation_sign_replicated"]) != 43:
        raise AssertionError("direct source contribution count changed")

    if binary["replicated_confirmatory_axis"].any():
        raise AssertionError("content-audit confirmatory null changed")
    maximum_content_difference = float(binary["paired_difference"].abs().max())
    assert_close(maximum_content_difference, 0.096, 1e-12)
    assert_close(float(binary_reliability["cohen_kappa"].median()), 0.7154544707, 1e-10)
    assert_close(float(ordinal_reliability["spearman_r"].median()), 0.8000840608, 1e-10)
    if post_annotation_audit["exploratory_family_direction"]["stable_direction_items"] != 158:
        raise AssertionError("stable advantaged-family item count changed")
    if bool(content_protocol["semantic_gate_passed"]):
        raise AssertionError("content semantic gate unexpectedly passed")

    interval_contains_observed = (
        (gap["observed_flip_rate"] >= gap["bootstrap_ci_low"])
        & (gap["observed_flip_rate"] <= gap["bootstrap_ci_high"])
    )
    interval_failures = gap.loc[
        ~interval_contains_observed,
        [
            "benchmark",
            "gap_points",
            "observed_flip_rate",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ],
    ].copy()
    interval_failures[["observed_flip_rate", "bootstrap_ci_low", "bootstrap_ci_high"]] *= 100

    return {
        "status": "all final claims used by the paper passed",
        "benchmarks": BENCHMARKS,
        "main_tau_range": [
            float(main["mean_kendall_tau_b"].min()),
            float(main["mean_kendall_tau_b"].max()),
        ],
        "main_all_pair_reversal_range": [
            float(main["all_cross_family_flip_rate"].min()),
            float(main["all_cross_family_flip_rate"].max()),
        ],
        "robustness_specs_per_benchmark": 10,
        "wino_significant_robustness_specs": wino_significant,
        "source_sign_replications": 58,
        "direct_source_contribution_replications": 43,
        "content_maximum_absolute_paired_difference": maximum_content_difference,
        "content_median_binary_cohen_kappa": float(
            binary_reliability["cohen_kappa"].median()
        ),
        "content_median_ordinal_spearman": float(
            ordinal_reliability["spearman_r"].median()
        ),
        "content_stable_advantaged_family_items": 158,
        "content_semantic_gate_passed": False,
        "composition_percentile_intervals_not_containing_observed": interval_failures.to_dict(
            orient="records"
        ),
    }


def draw_marker(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    shape: str,
    color: colors.Color,
    filled: bool,
    radius: float = 3.2,
) -> None:
    pdf.setStrokeColor(color)
    pdf.setLineWidth(1.2)
    pdf.setFillColor(color if filled else colors.white)
    if shape == "circle":
        pdf.circle(x, y, radius, stroke=1, fill=1)
    elif shape == "square":
        pdf.rect(x - radius, y - radius, 2 * radius, 2 * radius, stroke=1, fill=1)
    elif shape == "triangle":
        path = pdf.beginPath()
        path.moveTo(x, y + radius + 0.8)
        path.lineTo(x - radius - 0.5, y - radius)
        path.lineTo(x + radius + 0.5, y - radius)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)
    elif shape == "diamond":
        path = pdf.beginPath()
        path.moveTo(x, y + radius + 0.8)
        path.lineTo(x - radius - 0.5, y)
        path.lineTo(x, y - radius - 0.8)
        path.lineTo(x + radius + 0.5, y)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)
    elif shape == "cross":
        pdf.setFillColor(colors.white)
        pdf.circle(x, y, radius, stroke=1, fill=1)
        pdf.line(x - radius + 0.7, y - radius + 0.7, x + radius - 0.7, y + radius - 0.7)
        pdf.line(x - radius + 0.7, y + radius - 0.7, x + radius - 0.7, y - radius + 0.7)
    else:
        raise ValueError(shape)


def render_gap_figure(gap: pd.DataFrame, destination: Path) -> None:
    width, height = 7.0 * 72, 4.25 * 72
    pdf = canvas.Canvas(str(destination), pagesize=(width, height), invariant=1)
    left, right, bottom, top = 58, 18, 47, 66
    plot_width = width - left - right
    plot_height = height - bottom - top
    y_min, y_max = -3.0, 33.0

    def x_scale(index: int) -> float:
        return left + index * plot_width / (len(GAPS) - 1)

    def y_scale(value: float) -> float:
        return bottom + (value - y_min) * plot_height / (y_max - y_min)

    pdf.setTitle("Excess near-tie reversals across score-gap thresholds")
    pdf.setAuthor("Anonymous")
    pdf.setCreator("Anonymous")
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(colors.black)
    pdf.drawCentredString(width / 2, height - 17, "Excess near-tie reversals across score-gap thresholds")

    legend_y = height - 39
    legend_xs = [58, 146, 215, 290, 390]
    for benchmark, legend_x in zip(BENCHMARKS, legend_xs):
        color, shape = SERIES[benchmark]
        pdf.setStrokeColor(color)
        pdf.setLineWidth(1.8)
        pdf.line(legend_x, legend_y, legend_x + 15, legend_y)
        draw_marker(pdf, legend_x + 7.5, legend_y, shape, color, True, radius=2.7)
        pdf.setFont("Helvetica", 7.8)
        pdf.setFillColor(colors.black)
        pdf.drawString(legend_x + 20, legend_y - 2.5, benchmark)

    for tick in [0, 5, 10, 15, 20, 25, 30]:
        y = y_scale(tick)
        pdf.setStrokeColor(hex_color("#D9D9D9"))
        pdf.setLineWidth(0.5)
        pdf.line(left, y, width - right, y)
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica", 8)
        pdf.drawRightString(left - 7, y - 2.8, str(tick))

    zero_y = y_scale(0)
    pdf.setStrokeColor(hex_color("#555555"))
    pdf.setLineWidth(0.9)
    pdf.line(left, zero_y, width - right, zero_y)

    primary_x = x_scale(GAPS.index(1.0))
    pdf.setStrokeColor(hex_color("#777777"))
    pdf.setDash(3, 3)
    pdf.setLineWidth(0.8)
    pdf.line(primary_x, bottom, primary_x, height - top)
    pdf.setDash()
    pdf.setFont("Helvetica", 7.5)
    pdf.setFillColor(hex_color("#555555"))
    pdf.drawCentredString(primary_x, height - top + 5, "Primary: 1 pp")

    primary_offsets = {
        "MMLU-Pro": (5, 5),
        "BBH": (5, 5),
        "MMLU": (5, -10),
        "HellaSwag": (5, 5),
        "WinoGrande": (5, 5),
    }
    for benchmark in BENCHMARKS:
        frame = gap[gap["benchmark"].eq(benchmark)].set_index("gap_points").loc[GAPS]
        values = 100.0 * frame["excess_flip_rate"].to_numpy(dtype=float)
        p_values = frame["one_sided_randomization_p"].to_numpy(dtype=float)
        color, shape = SERIES[benchmark]
        points = [(x_scale(i), y_scale(float(value))) for i, value in enumerate(values)]
        pdf.setStrokeColor(color)
        pdf.setLineWidth(1.8)
        for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
            pdf.line(x1, y1, x2, y2)
        for (x, y), p_value in zip(points, p_values):
            draw_marker(pdf, x, y, shape, color, bool(p_value <= 0.05), radius=3.1)
        primary_value = float(values[GAPS.index(1.0)])
        dx, dy = primary_offsets[benchmark]
        pdf.setFillColor(color)
        pdf.setFont("Helvetica-Bold", 7.4)
        pdf.drawString(primary_x + dx, y_scale(primary_value) + dy, f"{primary_value:.1f}")

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 8)
    for index, gap_value in enumerate(GAPS):
        label = f"{gap_value:g}"
        pdf.drawCentredString(x_scale(index), bottom - 14, label)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawCentredString(
        left + plot_width / 2,
        13,
        "Maximum full-score gap (percentage points)",
    )
    pdf.saveState()
    pdf.translate(15, bottom + plot_height / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Excess reversal (percentage points)")
    pdf.restoreState()
    pdf.setFont("Helvetica", 7.2)
    pdf.setFillColor(hex_color("#555555"))
    pdf.drawRightString(
        width - right - 4,
        height - top - 10,
        "Filled markers: matched-random p ≤ .05",
    )
    pdf.setStrokeColor(hex_color("#777777"))
    pdf.setLineWidth(0.6)
    pdf.rect(left, bottom, plot_width, plot_height, stroke=1, fill=0)
    pdf.showPage()
    pdf.save()


def interpolate_color(start: colors.Color, end: colors.Color, amount: float) -> colors.Color:
    amount = float(np.clip(amount, 0.0, 1.0))
    return colors.Color(
        start.red + amount * (end.red - start.red),
        start.green + amount * (end.green - start.green),
        start.blue + amount * (end.blue - start.blue),
    )


def heat_color(value: float) -> colors.Color:
    """Vivid, high-contrast diverging scale centered at zero."""
    neutral = hex_color("#F7F9FC")
    if value >= 0:
        amount = np.power(np.clip(value / 36.0, 0.0, 1.0), 0.68)
        return interpolate_color(neutral, hex_color("#0067C5"), amount)
    amount = np.power(np.clip(abs(value) / 6.0, 0.0, 1.0), 0.68)
    return interpolate_color(neutral, hex_color("#E5484D"), amount)


def contrasting_text_color(fill: colors.Color) -> colors.Color:
    """Choose black or white text from the rendered cell luminance."""
    luminance = 0.2126 * fill.red + 0.7152 * fill.green + 0.0722 * fill.blue
    return colors.white if luminance < 0.56 else hex_color("#111827")


def render_robustness_heatmap(robustness: pd.DataFrame, destination: Path) -> None:
    """Render a compact small-multiple robustness dot plot.

    A common horizontal scale makes the scientific comparison visible without
    relying on a saturated heatmap. Exact primary values are labelled; the
    remaining population perturbations are shown as uncluttered points.
    """
    width, height = 7.0 * 72, 3.55 * 72
    pdf = canvas.Canvas(str(destination), pagesize=(width, height), invariant=1)
    pdf.setTitle("Population robustness of excess near-tie reversals")
    pdf.setAuthor("Anonymous")
    pdf.setCreator("Anonymous")
    # Shift slightly beyond geometric centering to compensate perceptually for
    # the visually heavy WinoGrande panel and the right-aligned axis title.
    pdf.translate(-16.5, 0)
    variant_order = [
        "owner_cap_5_baseline",
        "owner_cap_1",
        "owner_cap_3",
        "lexicographic_checkpoint",
        "strict_lineage",
        "omit_gemma",
        "omit_llama",
        "omit_mistral",
        "omit_phi",
        "omit_qwen",
    ]
    variant_labels = {
        "owner_cap_5_baseline": "Baseline (owner cap 5)",
        "owner_cap_1": "Owner cap 1",
        "owner_cap_3": "Owner cap 3",
        "lexicographic_checkpoint": "Score-blind checkpoint",
        "strict_lineage": "Conservative lineage filter",
        "omit_gemma": "Omit Gemma",
        "omit_llama": "Omit Llama",
        "omit_mistral": "Omit Mistral/Mixtral",
        "omit_phi": "Omit Phi",
        "omit_qwen": "Omit Qwen",
    }
    lookup = robustness.set_index(["variant", "benchmark"])

    # The caption carries the figure title; inside the graphic we keep only
    # the information needed to read the marks.
    left, right = 128, 15
    panel_gap = 8
    panel_width = (width - left - right - panel_gap * 4) / 5
    plot_top, plot_bottom = height - 29, 47
    row_step = (plot_top - plot_bottom) / (len(variant_order) - 1)
    x_min, x_max = -3.0, 38.0
    positive_color = hex_color("#2667A8")
    boundary_color = hex_color("#C4515C")
    neutral_dark = hex_color("#252525")
    neutral_mid = hex_color("#737373")
    grid_color = hex_color("#DEDEDE")

    def panel_left(column: int) -> float:
        return left + column * (panel_width + panel_gap)

    def x_scale(column: int, value: float) -> float:
        return panel_left(column) + (value - x_min) * panel_width / (x_max - x_min)

    def y_scale(row: int) -> float:
        return plot_top - row * row_step

    pdf.setFillColor(neutral_mid)
    pdf.setFont("Helvetica", 7.0)
    pdf.drawRightString(left - 10, height - 16, "Population specification")

    # Row labels are a single quiet typographic column, not a boxed table.
    for row, variant in enumerate(variant_order):
        y = y_scale(row)
        pdf.setFillColor(neutral_dark)
        pdf.setFont("Helvetica-Bold" if row == 0 else "Helvetica", 7.3)
        pdf.drawRightString(left - 10, y - 2.5, variant_labels[variant])

    for column, benchmark in enumerate(BENCHMARKS):
        panel_color = boundary_color if benchmark == "WinoGrande" else positive_color
        x0 = panel_left(column)
        x1 = x0 + panel_width

        pdf.setFillColor(neutral_dark)
        pdf.setFont("Helvetica-Bold", 8.0)
        pdf.drawCentredString((x0 + x1) / 2, height - 16, benchmark)

        # A shared scale across panels makes WinoGrande's near-zero behavior
        # immediately comparable with the four primary-positive benchmarks.
        for tick in [0, 20]:
            x = x_scale(column, tick)
            pdf.setStrokeColor(neutral_mid if tick == 0 else grid_color)
            pdf.setLineWidth(0.75 if tick == 0 else 0.45)
            pdf.line(x, plot_bottom - 4, x, plot_top + 4)
            pdf.setFillColor(neutral_mid)
            pdf.setFont("Helvetica", 6.3)
            pdf.drawCentredString(x, plot_bottom - 15, str(tick))

        # A faint baseline divider guides the eye without surrounding the row.
        divider_y = (y_scale(0) + y_scale(1)) / 2
        pdf.setStrokeColor(grid_color)
        pdf.setLineWidth(0.45)
        pdf.line(x0, divider_y, x1, divider_y)

        for row, variant in enumerate(variant_order):
            record = lookup.loc[(variant, benchmark)]
            value = 100.0 * float(record["excess_close_flip_rate"])
            significant = bool(record["significant_positive_excess"])
            x = x_scale(column, value)
            y = y_scale(row)

            pdf.setStrokeColor(panel_color)
            pdf.setLineWidth(0.75)
            pdf.line(x_scale(column, 0.0), y, x, y)
            draw_marker(
                pdf,
                x,
                y,
                "diamond" if row == 0 else "circle",
                panel_color,
                significant,
                radius=3.0 if row == 0 else 2.45,
            )

            # Label only the pre-specified baseline; exact robustness values
            # remain available in the appendix table.
            if row == 0:
                label_x = min(x + 4.5, x1 - 1)
                anchor_right = label_x >= x1 - 10
                pdf.setFillColor(panel_color)
                pdf.setFont("Helvetica-Bold", 6.5)
                if anchor_right:
                    pdf.drawRightString(x - 4.5, y - 2.2, f"{value:+.1f}")
                else:
                    pdf.drawString(label_x, y - 2.2, f"{value:+.1f}")

    # Minimal legend: shape encodes the primary row; fill encodes significance.
    legend_y = 16
    legend_color = neutral_dark
    draw_marker(pdf, 117, legend_y + 1.2, "diamond", legend_color, True, radius=2.7)
    pdf.setFillColor(neutral_mid)
    pdf.setFont("Helvetica", 6.8)
    pdf.drawString(124, legend_y - 1, "Primary specification")
    draw_marker(pdf, 220, legend_y + 1.2, "circle", legend_color, True, radius=2.35)
    pdf.drawString(227, legend_y - 1, "p ≤ .05")
    draw_marker(pdf, 270, legend_y + 1.2, "circle", legend_color, False, radius=2.35)
    pdf.setFillColor(neutral_mid)
    pdf.drawString(277, legend_y - 1, "p > .05")
    pdf.setFillColor(neutral_dark)
    pdf.setFont("Helvetica", 7.1)
    pdf.drawRightString(
        width - right,
        legend_y - 1,
        "Excess reversal (percentage points)",
    )
    pdf.showPage()
    pdf.save()


def render_content_audit(binary: pd.DataFrame, destination: Path) -> None:
    width, height = 7.0 * 72, 4.85 * 72
    pdf = canvas.Canvas(str(destination), pagesize=(width, height), invariant=1)
    pdf.setTitle("Blinded content audit")
    pdf.setAuthor("Anonymous")
    pdf.setCreator("Anonymous")
    # Center the plotting rectangle itself. Long labels are wrapped below so
    # they fit in the remaining left margin without shifting the data panel.
    left, right, bottom, top = 100, 100, 39, 64
    plot_width = width - left - right
    plot_height = height - bottom - top
    x_min, x_max = -10.0, 10.0

    axis_order = [
        "quantitative_symbolic",
        "formal_rule_reasoning",
        "factual_domain_knowledge",
        "contextual_reading",
        "commonsense_narrative",
        "spatial_temporal",
        "linguistic_wordplay",
        "negation_exception",
        "code_structured_representation",
        "distractor_discrimination",
    ]
    axis_labels = {
        "quantitative_symbolic": "Quantitative / symbolic",
        "formal_rule_reasoning": "Formal-rule reasoning",
        "factual_domain_knowledge": "Factual / domain\nknowledge",
        "contextual_reading": "Contextual reading",
        "commonsense_narrative": "Commonsense /\nnarrative",
        "spatial_temporal": "Spatial / temporal",
        "linguistic_wordplay": "Linguistic / wordplay",
        "negation_exception": "Negation / exception",
        "code_structured_representation": "Code / structured\nrepresentation",
        "distractor_discrimination": "Distractor discrimination",
    }
    annotators = [
        ("annotator_a", hex_color("#0072B2"), "circle", 2.2),
        ("annotator_b", hex_color("#D55E00"), "square", -2.2),
    ]
    row_height = plot_height / len(axis_order)

    def x_scale(value: float) -> float:
        return left + (value - x_min) * plot_width / (x_max - x_min)

    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(colors.black)
    pdf.drawCentredString(width / 2, height - 17, "Blinded content audit: high-DIF minus matched control")
    pdf.setFont("Helvetica", 7.5)
    pdf.setFillColor(hex_color("#555555"))
    pdf.drawCentredString(width / 2, height - 31, "Confirmatory binary axes; differences are paired prevalence changes")

    for tick in [-8, -4, 0, 4, 8]:
        x = x_scale(tick)
        pdf.setStrokeColor(hex_color("#D9D9D9"))
        pdf.setLineWidth(0.5)
        pdf.line(x, bottom, x, height - top)
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica", 8)
        pdf.drawCentredString(x, bottom - 13, f"{tick:+d}")

    pdf.setStrokeColor(hex_color("#555555"))
    pdf.setLineWidth(0.9)
    pdf.line(x_scale(0), bottom, x_scale(0), height - top)
    pdf.setStrokeColor(hex_color("#B2182B"))
    pdf.setDash(3, 3)
    for threshold in [-8, 8]:
        pdf.line(x_scale(threshold), bottom, x_scale(threshold), height - top)
    pdf.setDash()
    pdf.setFont("Helvetica", 7.2)
    pdf.setFillColor(hex_color("#B2182B"))
    pdf.drawCentredString(x_scale(-8), height - top + 5, "-8 pp gate")
    pdf.drawCentredString(x_scale(8), height - top + 5, "+8 pp gate")

    lookup = binary.set_index(["annotator", "axis"])
    for row_index, axis in enumerate(axis_order):
        center_y = height - top - (row_index + 0.5) * row_height
        pdf.setFont("Helvetica", 7.8)
        pdf.setFillColor(colors.black)
        label_lines = axis_labels[axis].split("\n")
        label_leading = 8.0
        label_y = center_y - 2.7 + 0.5 * (len(label_lines) - 1) * label_leading
        for line_index, label_line in enumerate(label_lines):
            pdf.drawRightString(
                left - 8,
                label_y - line_index * label_leading,
                label_line,
            )
        pdf.setStrokeColor(hex_color("#EEEEEE"))
        pdf.setLineWidth(0.4)
        pdf.line(left, center_y - row_height / 2, width - right, center_y - row_height / 2)
        for annotator, color, shape, offset in annotators:
            difference = 100.0 * float(lookup.loc[(annotator, axis), "paired_difference"])
            draw_marker(pdf, x_scale(difference), center_y + offset, shape, color, True, radius=3.0)

    # Put the legend at the upper right of the reserved header band so the
    # x-axis title has its own line below the plot.
    legend_y = height - 43
    for legend_x, (annotator, color, shape, _) in zip([336, 415], annotators):
        draw_marker(pdf, legend_x, legend_y + 1, shape, color, True, radius=2.8)
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica", 7.8)
        pdf.drawString(legend_x + 8, legend_y - 1.5, "Annotator A" if annotator.endswith("a") else "Annotator B")
    pdf.setFont("Helvetica", 8.3)
    pdf.drawCentredString(left + plot_width / 2, 3, "Paired prevalence difference (percentage points)")
    pdf.setStrokeColor(hex_color("#777777"))
    pdf.setLineWidth(0.6)
    pdf.rect(left, bottom, plot_width, plot_height, stroke=1, fill=0)
    pdf.showPage()
    pdf.save()


def main() -> None:
    global OUTPUTS, FIGURE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=OUTPUTS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/paper_figures')
    args = parser.parse_args()
    OUTPUTS, FIGURE_DIR = args.results_dir, args.output_dir
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    verification = validate_claims()
    (FIGURE_DIR / "verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    gap = pd.read_csv(
        OUTPUTS / "gap_sensitivity" / "gap_sensitivity_summary.csv"
    )
    robustness = pd.read_csv(
        OUTPUTS / "population_robustness" / "robustness_summary.csv"
    )
    binary = pd.read_csv(
        OUTPUTS / "content_audit" / "paired_binary_enrichment.csv"
    )
    render_gap_figure(gap, FIGURE_DIR / "gap_excess_verified.pdf")
    render_robustness_heatmap(
        robustness, FIGURE_DIR / "population_robustness_verified.pdf"
    )
    render_content_audit(binary, FIGURE_DIR / "content_audit_verified.pdf")
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
