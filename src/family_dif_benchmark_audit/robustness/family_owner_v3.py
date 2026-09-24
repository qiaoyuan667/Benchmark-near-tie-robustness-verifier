#!/usr/bin/env python3
"""V3 family/owner robustness with spectral MIRT as the primary estimator."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from ..core.family_dif import (
    DEFAULT_FAMILIES,
    classify_family,
    owner_of,
    select_and_split_family_models,
)
from ..ranking.mmlu_pro import (
    matched_random_anchor_controls,
    pairwise_ranking_metrics,
    select_owner_family_representatives,
    weighted_accuracy,
)
from ..discovery.qmirt_minimal import write_json
from ..core.spectral_mirt import purify_spectral_mirt_anchors
from .gap_sensitivity_v3 import RUNS, load_responses


from .._paths import PROJECT_ROOT


HERE = PROJECT_ROOT
OUTPUT = HERE / "outputs/v3_family_owner_robustness"
PRIMARY_ANCHOR_FRACTION = 0.5
CLOSE_GAP = 0.01
FAMILY_SEED = 20260821
CONTROL_SEED = 20260825
CONTROL_REPLICATES = 200
ROBUSTNESS_MIN_FAMILY_MODELS = 15
STRICT_LINEAGE_PATTERN = re.compile(
    r"merge|slerp|franken|lora|qlora|adapter|distill|hybrid|fusion|fuse|"
    r"(^|[-_.])dpo($|[-_.])|(^|[-_.])ppo($|[-_.])|(^|[-_.])orpo($|[-_.])|"
    r"(^|[-_.])kto($|[-_.])|abliterat|uncensor",
    flags=re.IGNORECASE,
)
VARIANT_ORDER = (
    "owner_cap_1",
    "owner_cap_3",
    "owner_cap_5_baseline",
    "lexicographic_checkpoint",
    "strict_lineage",
    "omit_gemma",
    "omit_llama",
    "omit_mistral",
    "omit_phi",
    "omit_qwen",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--control-replicates", type=int, default=CONTROL_REPLICATES)
    return parser.parse_args()


def all_family_candidates(
    model_ids: np.ndarray,
    accuracy: np.ndarray,
    requested_families: Sequence[str] = DEFAULT_FAMILIES,
) -> pd.DataFrame:
    requested = set(requested_families)
    rows: List[Dict[str, Any]] = []
    for index, model_id in enumerate(np.asarray(model_ids).astype(str)):
        family, reason = classify_family(model_id)
        if family not in requested:
            continue
        rows.append(
            {
                "dataset_model_index": index,
                "model_id": model_id,
                "family": family,
                "owner": owner_of(model_id),
                "train_accuracy": float(accuracy[index]),
                "classification": reason,
            }
        )
    return pd.DataFrame(rows)


def select_filtered_family_models(
    model_ids: np.ndarray,
    accuracy: np.ndarray,
    keep: np.ndarray,
    requested_families: Sequence[str],
    owner_cap: int,
) -> pd.DataFrame:
    original_indices = np.flatnonzero(np.asarray(keep, dtype=bool))
    selected = select_and_split_family_models(
        np.asarray(model_ids)[original_indices],
        np.asarray(accuracy)[original_indices],
        requested_families,
        owner_cap,
        ROBUSTNESS_MIN_FAMILY_MODELS,
        FAMILY_SEED,
    )
    selected["dataset_model_index"] = original_indices[
        selected["dataset_model_index"].to_numpy(dtype=np.int32)
    ]
    return selected


def lexicographic_family_models(
    candidates: pd.DataFrame, owner_halves: Mapping[str, str]
) -> pd.DataFrame:
    ordered = candidates.sort_values(
        ["owner", "family", "model_id"], kind="mergesort"
    )
    selected = ordered.groupby(["owner", "family"], sort=True).head(1).copy()
    selected["model_half"] = selected["owner"].map(owner_halves)
    if selected["model_half"].isna().any():
        raise ValueError("lexicographic candidate contains an owner without a frozen half")
    if selected.groupby(["owner", "family"]).size().max() != 1:
        raise AssertionError("lexicographic selection is not one checkpoint per owner-family")
    if (
        selected.groupby("family").size() < ROBUSTNESS_MIN_FAMILY_MODELS
    ).any():
        raise ValueError("lexicographic selection violates the family coverage rule")
    return selected.reset_index(drop=True)


def build_variants(
    model_ids: np.ndarray,
    responses: np.ndarray,
    baseline_path: Path,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    accuracy = responses.mean(axis=0, dtype=np.float64)
    baseline_family_path = baseline_path / "family_models.csv"
    if not baseline_family_path.exists():
        baseline_family_path = (
            HERE / "outputs/mmlu_pro_family_dif_minimal/family_models.csv"
        )
    baseline = pd.read_csv(baseline_family_path)
    owner_halves = (
        baseline[["owner", "model_half"]]
        .drop_duplicates()
        .set_index("owner")["model_half"]
        .to_dict()
    )
    candidates = all_family_candidates(model_ids, accuracy)
    variants: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    for owner_cap in (1, 3):
        selected = select_and_split_family_models(
            model_ids,
            accuracy,
            DEFAULT_FAMILIES,
            owner_cap,
            ROBUSTNESS_MIN_FAMILY_MODELS,
            FAMILY_SEED,
        )
        variants["owner_cap_%d" % owner_cap] = (
            selected,
            select_owner_family_representatives(selected),
        )
    baseline_representatives = pd.read_csv(
        baseline_path / "owner_family_representatives.csv"
    )
    variants["owner_cap_5_baseline"] = (baseline, baseline_representatives)

    lexicographic = lexicographic_family_models(candidates, owner_halves)
    variants["lexicographic_checkpoint"] = (lexicographic, lexicographic.copy())

    strict_keep = np.asarray(
        [not STRICT_LINEAGE_PATTERN.search(model_id) for model_id in model_ids],
        dtype=bool,
    )
    strict = select_filtered_family_models(
        model_ids,
        accuracy,
        strict_keep,
        DEFAULT_FAMILIES,
        5,
    )
    variants["strict_lineage"] = (
        strict,
        select_owner_family_representatives(strict),
    )

    for omitted in ("gemma", "llama", "mistral", "phi", "qwen"):
        requested = tuple(family for family in DEFAULT_FAMILIES if family != omitted)
        selected = select_and_split_family_models(
            model_ids,
            accuracy,
            requested,
            5,
            ROBUSTNESS_MIN_FAMILY_MODELS,
            FAMILY_SEED,
        )
        variants["omit_%s" % omitted] = (
            selected,
            select_owner_family_representatives(selected),
        )
    if tuple(variants) != VARIANT_ORDER:
        raise AssertionError("robustness variants are not in the frozen order")
    return variants


def selection_audit_rows(
    benchmark: str,
    variant: str,
    family_models: pd.DataFrame,
    representatives: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family, group in family_models.groupby("family", sort=True):
        reps = representatives[representatives["family"].eq(family)]
        rows.append(
            {
                "benchmark": benchmark,
                "variant": variant,
                "family": family,
                "audit_models": len(group),
                "owners": group["owner"].nunique(),
                "discovery_audit_models": int(group["model_half"].eq("discovery").sum()),
                "validation_audit_models": int(group["model_half"].eq("validation").sum()),
                "representatives": len(reps),
                "discovery_representatives": int(reps["model_half"].eq("discovery").sum()),
                "validation_representatives": int(reps["model_half"].eq("validation").sum()),
            }
        )
    return rows


def benchmark_sources(path: Path, n_items: int) -> np.ndarray:
    anchors = pd.read_csv(path / "crossfit_anchor_items.csv")
    anchors = anchors[
        np.isclose(anchors["anchor_fraction"], PRIMARY_ANCHOR_FRACTION)
        & anchors["audit_half"].eq("discovery")
    ].sort_values("dataset_item_index")
    if not np.array_equal(
        anchors["dataset_item_index"].to_numpy(dtype=np.int64), np.arange(n_items)
    ):
        raise ValueError("baseline anchor item metadata is not dense")
    return anchors["source"].astype(str).to_numpy()


def run_variant(
    benchmark: str,
    variant: str,
    responses: np.ndarray,
    sources: np.ndarray,
    family_models: pd.DataFrame,
    representatives: pd.DataFrame,
    dimensions_by_half: Mapping[str, int],
    control_replicates: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    family_models = family_models.copy()
    family_names = sorted(family_models["family"].unique())
    family_to_code = {family: index for index, family in enumerate(family_names)}
    family_models["family_code"] = family_models["family"].map(family_to_code)
    observed_rows: List[Dict[str, Any]] = []
    control_parts: List[pd.DataFrame] = []
    for audit_half, target_half in (
        ("discovery", "validation"),
        ("validation", "discovery"),
    ):
        audit = family_models[family_models["model_half"].eq(audit_half)]
        target = representatives[representatives["model_half"].eq(target_half)].copy()
        audit_indices = audit["dataset_model_index"].to_numpy(dtype=np.int32)
        target_indices = target["dataset_model_index"].to_numpy(dtype=np.int32)
        selected_dimension = int(dimensions_by_half[audit_half])
        fit = purify_spectral_mirt_anchors(
            responses[:, audit_indices],
            audit["family_code"].to_numpy(dtype=np.int32),
            sources,
            selected_dimension,
            20260826 + (100 if audit_half == "discovery" else 1100),
            difficulty_strata_count=4,
            dif_penalty=1.0,
            dif_cycles=5,
            rounds=3,
            anchor_fraction=PRIMARY_ANCHOR_FRACTION,
        )
        target_responses = responses[:, target_indices]
        full_scores = target_responses.mean(axis=0, dtype=np.float64)
        anchor_scores = weighted_accuracy(target_responses, fit.weights)
        metrics = pairwise_ranking_metrics(
            full_scores,
            anchor_scores,
            target["family"].astype(str).to_numpy(),
            target["owner"].astype(str).to_numpy(),
            CLOSE_GAP,
        )
        metrics.pop("close_pair_mask")
        family_shifts = pd.DataFrame(
            {
                "family": target["family"].astype(str).to_numpy(),
                "score_shift": anchor_scores - full_scores,
            }
        ).groupby("family")["score_shift"].mean()
        observed_rows.append(
            {
                "benchmark": benchmark,
                "variant": variant,
                "audit_half": audit_half,
                "target_half": target_half,
                "n_audit_models": len(audit),
                "n_representatives": len(target),
                "n_families": len(family_names),
                "selected_dimension": selected_dimension,
                "n_anchors": int(fit.selected.sum()),
                "family_score_shift_range_points": float(100.0 * np.ptp(family_shifts)),
                **metrics,
            }
        )
        controls = matched_random_anchor_controls(
            target_responses,
            fit,
            target["family"].astype(str).to_numpy(),
            target["owner"].astype(str).to_numpy(),
            CLOSE_GAP,
            control_replicates,
            CONTROL_SEED + (0 if target_half == "validation" else 1),
        )
        controls["benchmark"] = benchmark
        controls["variant"] = variant
        controls["audit_half"] = audit_half
        controls["target_half"] = target_half
        control_parts.append(controls)
    return pd.DataFrame(observed_rows), pd.concat(control_parts, ignore_index=True)


def baseline_result(benchmark: str, path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = pd.read_csv(path / "ranking_impact_metrics.csv")
    observed = observed[np.isclose(observed["anchor_fraction"], PRIMARY_ANCHOR_FRACTION)].copy()
    observed["benchmark"] = benchmark
    observed["variant"] = "owner_cap_5_baseline"
    observed["n_families"] = len(DEFAULT_FAMILIES)
    family_shifts = pd.read_csv(path / "family_rank_shifts.csv")
    family_shifts = family_shifts[
        np.isclose(family_shifts["anchor_fraction"], PRIMARY_ANCHOR_FRACTION)
    ]
    ranges = (
        family_shifts.groupby("target_half")["mean_score_shift_points"]
        .agg(lambda values: float(np.ptp(values)))
        .to_dict()
    )
    observed["family_score_shift_range_points"] = observed["target_half"].map(ranges)
    controls = pd.read_csv(path / "matched_random_anchor_controls.csv").copy()
    controls["benchmark"] = benchmark
    controls["variant"] = "owner_cap_5_baseline"
    return observed, controls


def summarize_results(observed: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    observed_summary = observed.groupby(["benchmark", "variant"], as_index=False).agg(
        mean_kendall_tau_b=("kendall_tau_b", "mean"),
        close_flips=("close_cross_family_flips", "sum"),
        close_pairs=("close_cross_family_eligible_pairs", "sum"),
        cross_flips=("cross_family_flips", "sum"),
        cross_pairs=("cross_family_eligible_pairs", "sum"),
        representatives=("n_representatives", "sum"),
        n_families=("n_families", "max"),
        family_shift_range_points=("family_score_shift_range_points", "mean"),
    )
    observed_summary["close_flip_rate"] = (
        observed_summary["close_flips"] / observed_summary["close_pairs"]
    )
    observed_summary["all_cross_family_flip_rate"] = (
        observed_summary["cross_flips"] / observed_summary["cross_pairs"]
    )
    pooled = controls.groupby(["benchmark", "variant", "replicate"], as_index=False).agg(
        close_flips=("close_cross_family_flips", "sum"),
        close_pairs=("close_cross_family_eligible_pairs", "sum"),
    )
    pooled["random_close_flip_rate"] = pooled["close_flips"] / pooled["close_pairs"]
    control_summary = pooled.groupby(["benchmark", "variant"])[
        "random_close_flip_rate"
    ].agg(
        random_ci_low=lambda values: values.quantile(0.025),
        random_median="median",
        random_ci_high=lambda values: values.quantile(0.975),
        random_replicates="size",
    ).reset_index()
    summary = observed_summary.merge(
        control_summary, on=["benchmark", "variant"], validate="one_to_one"
    )
    lookup = pooled.groupby(["benchmark", "variant"])
    p_values: List[float] = []
    for row in summary.itertuples(index=False):
        values = lookup.get_group((row.benchmark, row.variant))[
            "random_close_flip_rate"
        ].to_numpy(dtype=np.float64)
        p_values.append(
            float((np.sum(values >= row.close_flip_rate) + 1) / (len(values) + 1))
        )
    summary["one_sided_randomization_p"] = p_values
    summary["excess_close_flip_rate"] = (
        summary["close_flip_rate"] - summary["random_median"]
    )
    summary["positive_excess"] = summary["excess_close_flip_rate"] > 0.0
    summary["significant_positive_excess"] = (
        summary["positive_excess"] & (summary["one_sided_randomization_p"] <= 0.05)
    )
    benchmark_order = {benchmark: index for index, benchmark in enumerate(RUNS)}
    variant_order = {variant: index for index, variant in enumerate(VARIANT_ORDER)}
    summary["_benchmark_order"] = summary["benchmark"].map(benchmark_order)
    summary["_variant_order"] = summary["variant"].map(variant_order)
    return summary.sort_values(["_benchmark_order", "_variant_order"]).drop(
        columns=["_benchmark_order", "_variant_order"]
    ).reset_index(drop=True)


def criterion_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        group = summary[summary["variant"].eq(variant)]
        positive = int(group["positive_excess"].sum())
        significant = int(group["significant_positive_excess"].sum())
        rows.append(
            {
                "variant": variant,
                "positive_benchmarks_out_of_5": positive,
                "significant_benchmarks_out_of_5": significant,
                "passes_direction_gate": positive >= 3,
                "passes_significance_gate": significant >= 3,
            }
        )
    return pd.DataFrame(rows)


def build_matrix_svg(summary: pd.DataFrame) -> str:
    """Render a dependency-free benchmark-by-variant excess-flip matrix."""
    benchmarks = list(RUNS)
    variants = list(VARIANT_ORDER)
    lookup = summary.set_index(["benchmark", "variant"])
    cell_width = 112
    cell_height = 48
    left = 145
    top = 155
    width = left + len(variants) * cell_width + 38
    height = top + len(benchmarks) * cell_height + 88
    max_excess = max(0.30, float(summary["excess_close_flip_rate"].max()))

    def fill_color(value: float) -> str:
        if value >= 0.0:
            strength = min(1.0, value / max_excess)
            red = round(244 - 193 * strength)
            green = round(248 - 119 * strength)
            blue = round(255 - 42 * strength)
        else:
            strength = min(1.0, abs(value) / max_excess)
            red = round(255 - 30 * strength)
            green = round(246 - 175 * strength)
            blue = round(242 - 180 * strength)
        return f"rgb({red},{green},{blue})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}.title{font-size:20px;font-weight:700}.subtitle{font-size:12px;fill:#5a6578}.axis{font-size:12px;font-weight:600}.value{font-size:12px;font-weight:700}.note{font-size:11px;fill:#5a6578}</style>',
        '<text class="title" x="18" y="28">Family/owner robustness at the frozen 1 pp gap</text>',
        '<text class="subtitle" x="18" y="49">Cell = observed close-pair flip rate minus matched-random median; * denotes one-sided p ≤ .05</text>',
    ]
    for column, variant in enumerate(variants):
        x = left + column * cell_width + cell_width / 2
        parts.append(
            f'<text class="axis" text-anchor="start" transform="translate({x - 4},{top - 12}) rotate(-45)">{escape(variant)}</text>'
        )
    for row_index, benchmark in enumerate(benchmarks):
        y = top + row_index * cell_height
        parts.append(
            f'<text class="axis" text-anchor="end" x="{left - 12}" y="{y + 29}">{escape(benchmark)}</text>'
        )
        for column, variant in enumerate(variants):
            result = lookup.loc[(benchmark, variant)]
            x = left + column * cell_width
            excess = float(result["excess_close_flip_rate"])
            significant = bool(result["significant_positive_excess"])
            label = f'{100.0 * excess:+.1f} pp' + ('*' if significant else '')
            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{fill_color(excess)}" stroke="white" stroke-width="2"/>',
                    f'<text class="value" text-anchor="middle" x="{x + cell_width / 2}" y="{y + 29}">{label}</text>',
                ]
            )
    note_y = top + len(benchmarks) * cell_height + 28
    parts.append(
        f'<text class="note" x="{left}" y="{note_y}">All five benchmarks are audited across {len(variants)} owner/family perturbations.</text>'
    )
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def build_report(summary: pd.DataFrame, criteria: pd.DataFrame) -> str:
    lines = [
        "# V3 family/owner robustness",
        "",
        "## Verdict",
        "",
    ]
    key_variants = [
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
    key = criteria[criteria["variant"].isin(key_variants)]
    direction_pass = bool(key["passes_direction_gate"].all())
    significance_pass = bool(key["passes_significance_gate"].all())
    if direction_pass and significance_pass:
        verdict = "PASS: at least three of five benchmarks retain the result under every specified family/owner perturbation"
    elif direction_pass:
        verdict = "PARTIAL PASS: effect directions are robust, but some variants lose randomization significance"
    else:
        verdict = "FAIL: at least one frozen perturbation removes the effect in more than one positive benchmark"
    lines.extend(
        [
            "**%s.**" % verdict,
            "",
            "The gate asks whether at least three of all five benchmarks retain "
            "positive excess close-pair flips under each variant. The stricter "
            "significance gate also requires p<=.05 in at least three.",
            "",
            "![Robustness matrix](robustness_matrix.svg)",
            "",
            "## Gate summary",
            "",
            "| variant | positive / 5 | significant / 5 | direction gate | significance gate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in criteria.itertuples(index=False):
        lines.append(
            "| %s | %d | %d | %s | %s |"
            % (
                row.variant,
                row.positive_benchmarks_out_of_5,
                row.significant_benchmarks_out_of_5,
                "pass" if row.passes_direction_gate else "fail",
                "pass" if row.passes_significance_gate else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## Benchmark x variant results",
            "",
            "| benchmark | variant | close flips | random median | excess | p | families |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            "| %s | %s | %.1f%% | %.1f%% | %+.1f pp | %.4f | %d |"
            % (
                row.benchmark,
                row.variant,
                100.0 * row.close_flip_rate,
                100.0 * row.random_median,
                100.0 * row.excess_close_flip_rate,
                row.one_sided_randomization_p,
                row.n_families,
            )
        )
    lines.extend(
        [
            "",
            "All variants retain the original owner-disjoint cross-fit. The "
            "lexicographic policy chooses one checkpoint per owner-family without "
            "using benchmark accuracy. The strict-lineage filter is a conservative "
            "name-based sensitivity analysis, not a claim that all excluded models "
            "are invalid derivatives.",
            "",
            "## Interpretation limits",
            "",
            "WinoGrande is retained despite its null primary near-tie excess. The "
            "gate is deliberately based on replication across at least three of all "
            "five benchmarks rather than redefining a positive subset after results.",
            "",
            "New robustness variants use 200 random-control replicates (minimum "
            "attainable p=1/201); imported baseline rows retain their original 1,000 "
            "replicates. These tests rule out the specified family/owner definition "
            "artifacts, but they do not establish a causal mechanism for the item "
            "composition effect.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.control_replicates <= 0:
        raise ValueError("control replicates must be positive")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    selection_rows: List[Dict[str, Any]] = []
    prepared: Dict[str, Dict[str, Any]] = {}

    for benchmark, config in RUNS.items():
        print("prepare", benchmark, flush=True)
        responses, model_ids = load_responses(config)
        path = Path(config["path"])
        sources = benchmark_sources(path, responses.shape[0])
        dimension_frame = pd.read_csv(path / "dimension_selection.csv")
        dimensions_by_half = {
            str(row.audit_half): int(row.selected_dimension)
            for row in dimension_frame.itertuples(index=False)
        }
        variants = build_variants(model_ids, responses, path)
        for variant, (family_models, representatives) in variants.items():
            selection_rows.extend(
                selection_audit_rows(
                    benchmark, variant, family_models, representatives
                )
            )
        prepared[benchmark] = {
            "responses": responses,
            "model_ids": model_ids,
            "sources": sources,
            "variants": variants,
            "dimensions_by_half": dimensions_by_half,
            "path": path,
        }

    selection_audit = pd.DataFrame(selection_rows)
    selection_audit.to_csv(OUTPUT / "selection_audit.csv", index=False)
    write_json(
        OUTPUT / "protocol.json",
        {
            "experiment": "spectral_mirt_primary_family_owner_robustness",
            "frozen_variants": list(VARIANT_ORDER),
            "owner_caps": [1, 3, 5],
            "lexicographic_checkpoint": "one model per owner-family, selected by model_id without benchmark accuracy",
            "strict_lineage_exclusion_regex": STRICT_LINEAGE_PATTERN.pattern,
            "leave_one_family_out": list(DEFAULT_FAMILIES),
            "primary_anchor_fraction": PRIMARY_ANCHOR_FRACTION,
            "close_pair_gap": CLOSE_GAP,
            "matched_random_control_replicates_new_variants": args.control_replicates,
            "matched_random_control_replicates_imported_baseline": 1000,
            "robustness_minimum_family_models": ROBUSTNESS_MIN_FAMILY_MODELS,
            "family_seed": FAMILY_SEED,
            "control_seed": CONTROL_SEED,
            "dimension_policy": "freeze benchmark-fold K selected in the primary population",
            "direction_gate": "positive excess in at least 3 of all 5 benchmarks for every key variant",
            "significance_gate": "p<=.05 in at least 3 of all 5 benchmarks for every key variant",
            "selection_audit_written_before_dif_variants": True,
        },
    )
    if args.audit_only:
        print(selection_audit.to_string(index=False), flush=True)
        return

    observed_parts: List[pd.DataFrame] = []
    control_parts: List[pd.DataFrame] = []
    for benchmark, data in prepared.items():
        for variant in VARIANT_ORDER:
            print("run", benchmark, variant, flush=True)
            if variant == "owner_cap_5_baseline":
                observed, controls = baseline_result(benchmark, data["path"])
            else:
                family_models, representatives = data["variants"][variant]
                observed, controls = run_variant(
                    benchmark,
                    variant,
                    data["responses"],
                    data["sources"],
                    family_models,
                    representatives,
                    data["dimensions_by_half"],
                    args.control_replicates,
                )
            observed_parts.append(observed)
            control_parts.append(controls)

    observed = pd.concat(observed_parts, ignore_index=True)
    controls = pd.concat(control_parts, ignore_index=True)
    summary = summarize_results(observed, controls)
    criteria = criterion_rows(summary)
    observed.to_csv(OUTPUT / "robustness_by_fold.csv", index=False)
    controls.to_csv(OUTPUT / "matched_random_controls.csv", index=False)
    summary.to_csv(OUTPUT / "robustness_summary.csv", index=False)
    criteria.to_csv(OUTPUT / "criterion_summary.csv", index=False)
    (OUTPUT / "robustness_matrix.svg").write_text(
        build_matrix_svg(summary), encoding="utf-8"
    )
    (OUTPUT / "report.md").write_text(
        build_report(summary, criteria), encoding="utf-8"
    )
    with (OUTPUT / "protocol.json").open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    protocol["elapsed_seconds"] = float(time.monotonic() - started)
    write_json(OUTPUT / "protocol.json", protocol)
    print(criteria.to_string(index=False), flush=True)
    print("written to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
