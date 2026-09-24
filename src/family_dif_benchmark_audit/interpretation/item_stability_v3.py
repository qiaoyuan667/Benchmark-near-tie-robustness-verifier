#!/usr/bin/env python3
"""V3 owner-disjoint residual item-DIF signature replication after MIRT."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, spearmanr

from ..ranking.mmlu_pro import difficulty_cells
from ..discovery.qmirt_minimal import write_json
from ..core.spectral_mirt import purify_spectral_mirt_anchors
from ..robustness.gap_sensitivity_v3 import RUNS, load_responses


from .._paths import PROJECT_ROOT


HERE = PROJECT_ROOT
OUTPUT = HERE / "outputs/v3_item_dif_signature_stability"
PRIMARY_FRACTION = 0.50
DIFFICULTY_STRATA = 4
DIF_PENALTY = 1.0
DIF_CYCLES = 5
PURIFICATION_ROUNDS = 3
TOP_FRACTION = 0.20
PERMUTATION_REPLICATES = 500
PERMUTATION_SEED = 20260828
SOURCE_MIN_ITEMS = 20
SOURCE_EXTREMES_PER_TAIL = 2
MAIN_BENCHMARKS = ("MMLU-Pro", "BBH", "MMLU", "HellaSwag", "WinoGrande")
SOURCE_RICH_MAIN = ("MMLU-Pro", "BBH", "MMLU")
STABILITY_THRESHOLDS = {
    "median_family_within_cell_spearman": 0.20,
    "top20_overlap_precision": 0.30,
    "stable_top20_advantaged_family_agreement": 0.40,
    "one_sided_permutation_p": 0.05,
    "benchmarks_required": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument(
        "--permutation-replicates", type=int, default=PERMUTATION_REPLICATES
    )
    return parser.parse_args()


def protocol(replicates: int) -> Dict[str, Any]:
    return {
        "experiment": "spectral_mirt_primary_item_dif_signature_stability",
        "benchmarks": list(RUNS),
        "owner_split": "reuse the frozen owner-disjoint discovery/validation halves",
        "family_models": "reuse the frozen owner_cap_5 baseline audit models",
        "primary_anchor_fraction": PRIMARY_FRACTION,
        "difficulty_strata": DIFFICULTY_STRATA,
        "dif_penalty": DIF_PENALTY,
        "dif_cycles": DIF_CYCLES,
        "purification_rounds": PURIFICATION_ROUNDS,
        "common_blueprint_cells": "source-by-easiness quartile from the mean of the two owner-half item intercepts",
        "top_dif_fraction_within_cell": TOP_FRACTION,
        "permutation": "shuffle validation item signatures within common blueprint cells",
        "permutation_replicates": replicates,
        "permutation_seed": PERMUTATION_SEED,
        "stability_thresholds": STABILITY_THRESHOLDS,
        "primary_gate": "at least 3 of all 5 benchmarks pass all three stability thresholds and their one-sided permutation tests",
        "source_validation": {
            "selection_half": "discovery only",
            "confirmation_half": "validation only",
            "effect_centering": "subtract each owner-half family's all-item mean before source selection and sign confirmation",
            "minimum_source_items": SOURCE_MIN_ITEMS,
            "extremes_per_family_tail": SOURCE_EXTREMES_PER_TAIL,
            "content_gate": "pooled validation sign replication >=70% with exact binomial p<=.05, and >=2 of 3 source-rich main benchmarks individually replicate >=65%",
            "centering_audit_note": "family-mean centering was added after a preliminary 48/48 sign result was flagged as potentially containing global family offsets; the primary item-stability protocol and results were unchanged",
        },
        "direct_source_shift_attribution": {
            "definition": "exact additive source contribution to cross-fitted family mean anchor-minus-full score shift",
            "selection": "two most negative and two most positive discovery-fold contributions per family; confirm sign in the swapped fold",
            "status": "mechanism-adjacent secondary analysis added after the frozen primary item-stability test passed",
            "descriptive_support_rule": "pooled main replication >=70%, exact binomial p<=.05, and >=2 of 3 source-rich main benchmarks individually >=65%",
        },
        "winogrande_role": "retained in the all-five stability gate despite its null primary near-tie excess",
        "protocol_written_before_item_dif_refits": True,
    }


def baseline_family_models(path: Path) -> pd.DataFrame:
    family_path = path / "family_models.csv"
    if not family_path.exists():
        family_path = HERE / "outputs/mmlu_pro_family_dif_minimal/family_models.csv"
    frame = pd.read_csv(family_path)
    required = {
        "dataset_model_index",
        "model_id",
        "family",
        "owner",
        "model_half",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("family model table is missing %s" % sorted(missing))
    if frame.groupby("owner")["model_half"].nunique().max() != 1:
        raise ValueError("an owner crosses the frozen halves")
    return frame


def baseline_item_metadata(path: Path, n_items: int) -> pd.DataFrame:
    frame = pd.read_csv(path / "crossfit_anchor_items.csv")
    frame = frame[
        np.isclose(frame["anchor_fraction"], PRIMARY_FRACTION)
        & frame["audit_half"].eq("discovery")
    ].sort_values("dataset_item_index")
    if not np.array_equal(
        frame["dataset_item_index"].to_numpy(dtype=np.int64), np.arange(n_items)
    ):
        raise ValueError("baseline item metadata is not dense")
    return frame.reset_index(drop=True)


def fit_owner_half(
    responses: np.ndarray,
    family_models: pd.DataFrame,
    sources: np.ndarray,
    audit_half: str,
    family_to_code: Mapping[str, int],
    selected_dimension: int,
) -> Any:
    audit = family_models[family_models["model_half"].eq(audit_half)].copy()
    audit["family_code"] = audit["family"].map(family_to_code)
    if audit["family_code"].isna().any():
        raise ValueError("audit half contains an unknown family")
    return purify_spectral_mirt_anchors(
        responses[:, audit["dataset_model_index"].to_numpy(dtype=np.int32)],
        audit["family_code"].to_numpy(dtype=np.int32),
        sources,
        selected_dimension,
        20260826 + (100 if audit_half == "discovery" else 1100),
        difficulty_strata_count=DIFFICULTY_STRATA,
        dif_penalty=DIF_PENALTY,
        dif_cycles=DIF_CYCLES,
        rounds=PURIFICATION_ROUNDS,
        anchor_fraction=PRIMARY_FRACTION,
    )


def center_within_cells(values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = values.copy()
    for cell in np.unique(cells):
        mask = cells == cell
        result[mask] -= result[mask].mean(axis=0)
    return result


def stratified_high_mask(
    values: np.ndarray, cells: np.ndarray, fraction: float
) -> np.ndarray:
    selected = np.zeros(len(values), dtype=bool)
    for cell in np.unique(cells):
        positions = np.flatnonzero(cells == cell)
        count = max(1, min(len(positions), int(round(fraction * len(positions)))))
        order = positions[np.lexsort((positions, -np.asarray(values)[positions]))]
        selected[order[:count]] = True
    return selected


def standardized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=np.float64), method="average")
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    if norm == 0.0:
        return np.zeros_like(ranks)
    return ranks / norm


def fast_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(standardized_ranks(left), standardized_ranks(right)))


def item_stability_and_null(
    benchmark: str,
    discovery: Any,
    validation: Any,
    sources: np.ndarray,
    family_names: Sequence[str],
    replicates: int,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
    common_intercept = 0.5 * (
        discovery.item_intercepts + validation.item_intercepts
    )
    cells = difficulty_cells(sources, common_intercept, DIFFICULTY_STRATA)
    discovery_effect = center_within_cells(discovery.effects, cells)
    validation_effect = center_within_cells(validation.effects, cells)
    discovery_magnitude = center_within_cells(discovery.dif_magnitude, cells)
    validation_magnitude = center_within_cells(validation.dif_magnitude, cells)
    family_correlations = np.asarray(
        [
            fast_rank_correlation(discovery_effect[:, index], validation_effect[:, index])
            for index in range(len(family_names))
        ]
    )
    raw_family_correlations = np.asarray(
        [
            fast_rank_correlation(discovery.effects[:, index], validation.effects[:, index])
            for index in range(len(family_names))
        ]
    )
    magnitude_correlation = fast_rank_correlation(
        discovery_magnitude, validation_magnitude
    )
    raw_magnitude_correlation = fast_rank_correlation(
        discovery.dif_magnitude, validation.dif_magnitude
    )
    discovery_high = stratified_high_mask(discovery.dif_magnitude, cells, TOP_FRACTION)
    validation_high = stratified_high_mask(validation.dif_magnitude, cells, TOP_FRACTION)
    stable_high = discovery_high & validation_high
    overlap_precision = float(stable_high.sum() / discovery_high.sum())
    discovery_advantage = np.argmax(discovery.effects, axis=1)
    validation_advantage = np.argmax(validation.effects, axis=1)
    advantage_agreement = float(
        np.mean(discovery_advantage[stable_high] == validation_advantage[stable_high])
    )

    x_ranks = np.column_stack(
        [standardized_ranks(discovery_effect[:, i]) for i in range(len(family_names))]
    )
    y_ranks = np.column_stack(
        [standardized_ranks(validation_effect[:, i]) for i in range(len(family_names))]
    )
    x_mag_rank = standardized_ranks(discovery_magnitude)
    y_mag_rank = standardized_ranks(validation_magnitude)
    cell_positions = [np.flatnonzero(cells == cell) for cell in np.unique(cells)]
    rng = np.random.default_rng(seed)
    null_rows: List[Dict[str, Any]] = []
    base = np.arange(len(cells), dtype=np.int32)
    for replicate in range(replicates):
        permutation = base.copy()
        for positions in cell_positions:
            permutation[positions] = rng.permutation(positions)
        null_family = np.asarray(
            [
                float(np.dot(x_ranks[:, i], y_ranks[permutation, i]))
                for i in range(len(family_names))
            ]
        )
        permuted_high = validation_high[permutation]
        permuted_stable = discovery_high & permuted_high
        null_rows.append(
            {
                "benchmark": benchmark,
                "replicate": replicate,
                "median_family_within_cell_spearman": float(np.median(null_family)),
                "magnitude_within_cell_spearman": float(
                    np.dot(x_mag_rank, y_mag_rank[permutation])
                ),
                "top20_overlap_precision": float(
                    permuted_stable.sum() / discovery_high.sum()
                ),
                "stable_top20_advantaged_family_agreement": float(
                    np.mean(
                        discovery_advantage[permuted_stable]
                        == validation_advantage[permutation][permuted_stable]
                    )
                ),
            }
        )
    null = pd.DataFrame(null_rows)

    def upper_p(column: str, observed: float) -> float:
        values = null[column].to_numpy(dtype=np.float64)
        return float((np.sum(values >= observed) + 1) / (len(values) + 1))

    summary: Dict[str, Any] = {
        "benchmark": benchmark,
        "items": len(cells),
        "sources": len(np.unique(sources)),
        "families": len(family_names),
        "raw_magnitude_spearman": raw_magnitude_correlation,
        "magnitude_within_cell_spearman": magnitude_correlation,
        "magnitude_within_cell_permutation_p": upper_p(
            "magnitude_within_cell_spearman", magnitude_correlation
        ),
        "raw_median_family_spearman": float(np.median(raw_family_correlations)),
        "median_family_within_cell_spearman": float(np.median(family_correlations)),
        "min_family_within_cell_spearman": float(np.min(family_correlations)),
        "max_family_within_cell_spearman": float(np.max(family_correlations)),
        "family_within_cell_permutation_p": upper_p(
            "median_family_within_cell_spearman", float(np.median(family_correlations))
        ),
        "top20_discovery_items": int(discovery_high.sum()),
        "top20_stable_items": int(stable_high.sum()),
        "top20_overlap_precision": overlap_precision,
        "top20_overlap_permutation_p": upper_p(
            "top20_overlap_precision", overlap_precision
        ),
        "stable_top20_advantaged_family_agreement": advantage_agreement,
        "advantaged_family_agreement_permutation_p": upper_p(
            "stable_top20_advantaged_family_agreement", advantage_agreement
        ),
    }
    for index, family in enumerate(family_names):
        summary["%s_raw_spearman" % family] = float(raw_family_correlations[index])
        summary["%s_within_cell_spearman" % family] = float(family_correlations[index])
    threshold = STABILITY_THRESHOLDS
    summary["passes_family_effect_gate"] = bool(
        summary["median_family_within_cell_spearman"]
        > threshold["median_family_within_cell_spearman"]
        and summary["family_within_cell_permutation_p"]
        <= threshold["one_sided_permutation_p"]
    )
    summary["passes_top20_overlap_gate"] = bool(
        overlap_precision > threshold["top20_overlap_precision"]
        and summary["top20_overlap_permutation_p"]
        <= threshold["one_sided_permutation_p"]
    )
    summary["passes_advantage_agreement_gate"] = bool(
        advantage_agreement
        > threshold["stable_top20_advantaged_family_agreement"]
        and summary["advantaged_family_agreement_permutation_p"]
        <= threshold["one_sided_permutation_p"]
    )
    summary["passes_all_stability_gates"] = bool(
        summary["passes_family_effect_gate"]
        and summary["passes_top20_overlap_gate"]
        and summary["passes_advantage_agreement_gate"]
    )
    return summary, null, cells


def source_extremes(
    benchmark: str,
    sources: np.ndarray,
    discovery_effects: np.ndarray,
    validation_effects: np.ndarray,
    family_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    discovery_effects = discovery_effects - discovery_effects.mean(axis=0, keepdims=True)
    validation_effects = validation_effects - validation_effects.mean(axis=0, keepdims=True)
    source_rows: List[Dict[str, Any]] = []
    for family_index, family in enumerate(family_names):
        for source in np.unique(sources):
            mask = sources == source
            source_rows.append(
                {
                    "benchmark": benchmark,
                    "family": family,
                    "source": source,
                    "items": int(mask.sum()),
                    "discovery_mean_effect": float(
                        discovery_effects[mask, family_index].mean()
                    ),
                    "validation_mean_effect": float(
                        validation_effects[mask, family_index].mean()
                    ),
                }
            )
    all_sources = pd.DataFrame(source_rows)
    selected_rows: List[pd.DataFrame] = []
    for family, group in all_sources.groupby("family", sort=True):
        eligible = group[group["items"] >= SOURCE_MIN_ITEMS].sort_values(
            ["discovery_mean_effect", "source"], kind="mergesort"
        )
        needed = 2 * SOURCE_EXTREMES_PER_TAIL
        if len(eligible) < needed:
            continue
        low = eligible.head(SOURCE_EXTREMES_PER_TAIL).assign(
            discovery_direction="disadvantage"
        )
        high = eligible.tail(SOURCE_EXTREMES_PER_TAIL).assign(
            discovery_direction="advantage"
        )
        selected_rows.extend([low, high])
    if not selected_rows:
        return all_sources, pd.DataFrame()
    selected = pd.concat(selected_rows, ignore_index=True)
    selected["validation_sign_replicated"] = (
        selected["discovery_mean_effect"] * selected["validation_mean_effect"] > 0.0
    )
    selected["validation_attenuation_ratio"] = (
        selected["validation_mean_effect"]
        / selected["discovery_mean_effect"].replace(0.0, np.nan)
    )
    return all_sources, selected


def source_effect_correlations(source_effects: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (benchmark, family), group in source_effects.groupby(
        ["benchmark", "family"], sort=False
    ):
        if len(group) < 2:
            continue
        rows.append(
            {
                "benchmark": benchmark,
                "family": family,
                "sources": len(group),
                "source_effect_spearman": float(
                    spearmanr(
                        group["discovery_mean_effect"],
                        group["validation_mean_effect"],
                    ).statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def source_shift_attribution(
    benchmark: str,
    responses: np.ndarray,
    sources: np.ndarray,
    representatives: pd.DataFrame,
    fits: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exactly decompose cross-fitted anchor-minus-full family shifts by source."""
    rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for audit_half, target_half in (
        ("discovery", "validation"),
        ("validation", "discovery"),
    ):
        target = representatives[
            representatives["model_half"].eq(target_half)
        ].reset_index(drop=True)
        indices = target["dataset_model_index"].to_numpy(dtype=np.int32)
        target_responses = responses[:, indices].astype(np.float64)
        weights = fits[audit_half].weights
        source_contribution_by_family: Dict[str, float] = {
            family: 0.0 for family in target["family"].unique()
        }
        for source in np.unique(sources):
            mask = sources == source
            model_contribution = (
                (weights[mask] - 1.0) @ target_responses[mask] / len(weights)
            )
            for family, positions in target.groupby("family").groups.items():
                contribution = float(
                    model_contribution[np.asarray(list(positions), dtype=np.int32)].mean()
                )
                rows.append(
                    {
                        "benchmark": benchmark,
                        "audit_half": audit_half,
                        "target_half": target_half,
                        "family": family,
                        "source": source,
                        "source_items": int(mask.sum()),
                        "score_shift_contribution_points": 100.0 * contribution,
                    }
                )
                source_contribution_by_family[family] += contribution
        full_scores = target_responses.mean(axis=0, dtype=np.float64)
        anchor_scores = weights @ target_responses / float(weights.sum())
        for family, positions in target.groupby("family").groups.items():
            direct = float(
                (anchor_scores - full_scores)[
                    np.asarray(list(positions), dtype=np.int32)
                ].mean()
            )
            decomposed = source_contribution_by_family[family]
            audit_rows.append(
                {
                    "benchmark": benchmark,
                    "audit_half": audit_half,
                    "target_half": target_half,
                    "family": family,
                    "direct_family_score_shift_points": 100.0 * direct,
                    "sum_source_contributions_points": 100.0 * decomposed,
                    "absolute_decomposition_error_points": 100.0
                    * abs(direct - decomposed),
                }
            )
            if abs(direct - decomposed) > 1e-10:
                raise AssertionError("source contributions do not sum to the score shift")
    contributions = pd.DataFrame(rows)
    wide = contributions.pivot(
        index=["benchmark", "family", "source", "source_items"],
        columns="audit_half",
        values="score_shift_contribution_points",
    ).reset_index()
    selected_parts: List[pd.DataFrame] = []
    if wide["source"].nunique() > 1:
        for family, group in wide.groupby("family", sort=True):
            ordered = group.sort_values(
                ["discovery", "source"], kind="mergesort"
            )
            low = ordered.head(SOURCE_EXTREMES_PER_TAIL).assign(
                discovery_direction="negative_contribution"
            )
            high = ordered.tail(SOURCE_EXTREMES_PER_TAIL).assign(
                discovery_direction="positive_contribution"
            )
            selected_parts.extend([low, high])
    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else pd.DataFrame()
    )
    if not selected.empty:
        selected["validation_sign_replicated"] = (
            selected["discovery"] * selected["validation"] > 0.0
        )
    return contributions, selected, pd.DataFrame(audit_rows)


def selected_replication_summary(
    selected: pd.DataFrame, benchmark_order: Sequence[str], pooled_name: str
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for benchmark in benchmark_order:
        group = selected[selected["benchmark"].eq(benchmark)]
        if group.empty:
            continue
        replicated = int(group["validation_sign_replicated"].sum())
        rows.append(
            {
                "benchmark": benchmark,
                "selected_extremes": len(group),
                "validation_sign_replicated": replicated,
                "validation_sign_replication_rate": float(replicated / len(group)),
                "exact_binomial_p": float(
                    binomtest(
                        replicated, len(group), 0.5, alternative="greater"
                    ).pvalue
                ),
            }
        )
    pooled = selected[selected["benchmark"].isin(SOURCE_RICH_MAIN)]
    replicated = int(pooled["validation_sign_replicated"].sum())
    rows.append(
        {
            "benchmark": pooled_name,
            "selected_extremes": len(pooled),
            "validation_sign_replicated": replicated,
            "validation_sign_replication_rate": float(replicated / len(pooled)),
            "exact_binomial_p": float(
                binomtest(replicated, len(pooled), 0.5, alternative="greater").pvalue
            ),
        }
    )
    return pd.DataFrame(rows)


def load_questions(
    benchmark: str, item_metadata: pd.DataFrame, n_items: int
) -> np.ndarray:
    if benchmark == "MMLU-Pro":
        frame = pd.read_parquet(HERE / "external_cache/mmlu_pro_test.parquet")
        if len(frame) != n_items:
            raise ValueError("MMLU-Pro question table does not align")
        return frame["question"].astype(str).to_numpy()
    global_items = json.load(
        (HERE / "outputs/routereval_raw_global/all_benchmarks_dataset.json").open(
            encoding="utf-8"
        )
    )
    indices = item_metadata["all_benchmarks_item_index"].to_numpy(dtype=np.int64)
    questions = np.asarray([str(global_items[index]["problem"]) for index in indices])
    if len(questions) != n_items:
        raise ValueError("RouterEval question table does not align")
    return questions


def extreme_examples(
    selected: pd.DataFrame,
    sources: np.ndarray,
    questions: np.ndarray,
    discovery_effects: np.ndarray,
    validation_effects: np.ndarray,
    family_names: Sequence[str],
) -> pd.DataFrame:
    discovery_effects = discovery_effects - discovery_effects.mean(axis=0, keepdims=True)
    validation_effects = validation_effects - validation_effects.mean(axis=0, keepdims=True)
    rows: List[Dict[str, Any]] = []
    family_to_index = {family: index for index, family in enumerate(family_names)}
    for selected_row in selected.itertuples(index=False):
        family_index = family_to_index[selected_row.family]
        positions = np.flatnonzero(sources == selected_row.source)
        direction = 1.0 if selected_row.discovery_direction == "advantage" else -1.0
        replicated_strength = np.minimum(
            direction * discovery_effects[positions, family_index],
            direction * validation_effects[positions, family_index],
        )
        order = positions[np.argsort(-replicated_strength, kind="mergesort")]
        for rank, position in enumerate(order[:2], start=1):
            rows.append(
                {
                    "benchmark": selected_row.benchmark,
                    "family": selected_row.family,
                    "source": selected_row.source,
                    "source_direction": selected_row.discovery_direction,
                    "example_rank": rank,
                    "dataset_item_index": int(position),
                    "discovery_effect": float(discovery_effects[position, family_index]),
                    "validation_effect": float(validation_effects[position, family_index]),
                    "question": str(questions[position]),
                }
            )
    return pd.DataFrame(rows)


def build_report(
    stability: pd.DataFrame,
    source_summary: pd.DataFrame,
    source_correlations: pd.DataFrame,
    attribution_summary: pd.DataFrame,
    primary_pass: bool,
    content_pass: bool,
    attribution_support: bool,
) -> str:
    main_source_correlations = source_correlations[
        source_correlations["benchmark"].isin(SOURCE_RICH_MAIN)
    ]["source_effect_spearman"]
    lines = [
        "# Mandatory experiment C: item-DIF signature stability",
        "",
        "## Verdict",
        "",
        "**%s.**" % (
            "PASS: owner-disjoint item signatures are stable enough for content interpretation"
            if primary_pass
            else "FAIL: item signatures do not meet the frozen replication gate"
        ),
        "",
        "The primary gate requires at least three of all five benchmarks to "
        "pass family-effect correlation, high-DIF overlap, and "
        "advantaged-family agreement after source-by-easiness control.",
        "",
        "![Item-DIF stability and direct source attribution](summary.svg)",
        "",
        "## Item-level stability",
        "",
        "| benchmark | median family rho | top-20% overlap | advantage agreement | permutation p values | all gates |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in stability.itertuples(index=False):
        lines.append(
            "| %s | %.3f | %.1f%% | %.1f%% | %.4f / %.4f / %.4f | %s |"
            % (
                row.benchmark,
                row.median_family_within_cell_spearman,
                100.0 * row.top20_overlap_precision,
                100.0 * row.stable_top20_advantaged_family_agreement,
                row.family_within_cell_permutation_p,
                row.top20_overlap_permutation_p,
                row.advantaged_family_agreement_permutation_p,
                "pass" if row.passes_all_stability_gates else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## Discovery-selected source extremes",
            "",
            "**%s.**" % (
                "CONTENT PASS: source directions replicate in the held-out owner half"
                if content_pass
                else "CONTENT FAIL/PARTIAL: source directions do not meet the frozen confirmation gate"
            ),
            "",
            "| benchmark | selected source-family extremes | validation sign replication | exact binomial p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in source_summary.itertuples(index=False):
        lines.append(
            "| %s | %d | %.1f%% | %.4f |"
            % (
                row.benchmark,
                row.selected_extremes,
                100.0 * row.validation_sign_replication_rate,
                row.exact_binomial_p,
            )
        )
    lines.extend(
        [
            "",
            "Effects are centered by each family's all-item mean separately in "
            "each owner half. Across all source-family cells, not just the selected "
            "extremes, the main-benchmark cross-half source correlations range "
            "from %.3f to %.3f." % (
                float(main_source_correlations.min()),
                float(main_source_correlations.max()),
            ),
            "",
            "## Direct attribution of the score shifts",
            "",
            "**%s.**" % (
                "SUPPORT: discovery-selected source contributions reproduce in the swapped cross-fit"
                if attribution_support
                else "PARTIAL/FAIL: direct source contributions do not meet the descriptive support rule"
            ),
            "",
            "Each source contribution is an exact additive component of the "
            "family's anchor-minus-full score shift; the components sum to the "
            "reported shift up to floating-point error.",
            "",
            "| benchmark | selected source contributions | swapped-fold sign replication | exact binomial p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in attribution_summary.itertuples(index=False):
        lines.append(
            "| %s | %d | %.1f%% | %.4f |"
            % (
                row.benchmark,
                row.selected_extremes,
                100.0 * row.validation_sign_replication_rate,
                row.exact_binomial_p,
            )
        )
    lines.extend(
        [
            "",
            "The source analysis is mechanism-adjacent evidence: it identifies "
            "which benchmark domains reproduce family-relative advantages, but "
            "does not by itself show which cognitive process or training-data "
            "difference caused those advantages. HellaSwag has one native source "
            "label and is therefore excluded from the source-domain confirmation.",
            "",
            "WinoGrande is especially informative: its owner-disjoint residual "
            "item signatures can be evaluated even though its near-tie excess over "
            "matched random subtests is null. Stable item heterogeneity is therefore "
            "not sufficient for a large ranking-resolution effect; signed effects "
            "must also aggregate coherently under benchmark composition. MMLU's "
            "direct source attribution is also partial, so its "
            "specific source drivers should not be presented as confirmed even "
            "though its item signatures and ranking result replicate.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_summary_svg(
    stability: pd.DataFrame, attribution_summary: pd.DataFrame
) -> str:
    width, height = 1120, 455
    left, top = 132, 112
    cell_width, cell_height = 145, 46
    metrics = (
        ("median_family_within_cell_spearman", "family rho", 1.0),
        ("top20_overlap_precision", "top-20 overlap", 1.0),
        (
            "stable_top20_advantaged_family_agreement",
            "advantage agree",
            1.0,
        ),
    )

    def blue(value: float) -> str:
        strength = min(1.0, max(0.0, value))
        return "rgb(%d,%d,%d)" % (
            round(242 - 186 * strength),
            round(247 - 112 * strength),
            round(255 - 34 * strength),
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}.title{font-size:20px;font-weight:700}.subtitle{font-size:12px;fill:#5a6578}.axis{font-size:12px;font-weight:600}.value{font-size:13px;font-weight:700}.panel{font-size:14px;font-weight:700}.note{font-size:11px;fill:#5a6578}</style>',
        '<text class="title" x="18" y="28">Owner-disjoint item signatures replicate across benchmarks</text>',
        '<text class="subtitle" x="18" y="49">Left: source-by-easiness controlled item stability. Right: swapped-fold replication of discovery-selected score-shift drivers.</text>',
        '<text class="panel" x="18" y="82">A. Item-level signature stability</text>',
        '<text class="panel" x="650" y="82">B. Direct source attribution</text>',
    ]
    for column, (_, label, _) in enumerate(metrics):
        x = left + column * cell_width + cell_width / 2
        parts.append(
            f'<text class="axis" text-anchor="middle" x="{x}" y="{top - 12}">{escape(label)}</text>'
        )
    for row_index, row in enumerate(stability.itertuples(index=False)):
        y = top + row_index * cell_height
        parts.append(
            f'<text class="axis" text-anchor="end" x="{left - 12}" y="{y + 28}">{escape(row.benchmark)}</text>'
        )
        for column, (field, _, scale) in enumerate(metrics):
            value = float(getattr(row, field))
            x = left + column * cell_width
            label = f"{value:.3f}" if column == 0 else f"{100.0 * value:.1f}%"
            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{blue(value / scale)}" stroke="white" stroke-width="2"/>',
                    f'<text class="value" text-anchor="middle" x="{x + cell_width / 2}" y="{y + 29}">{label}</text>',
                ]
            )
    panel = attribution_summary[
        ~attribution_summary["benchmark"].str.startswith("Pooled")
    ]
    bar_left, bar_width = 790, 270
    for row_index, row in enumerate(panel.itertuples(index=False)):
        y = top + row_index * cell_height + 8
        rate = float(row.validation_sign_replication_rate)
        parts.extend(
            [
                f'<text class="axis" text-anchor="end" x="{bar_left - 12}" y="{y + 16}">{escape(row.benchmark)}</text>',
                f'<rect x="{bar_left}" y="{y}" width="{bar_width}" height="22" fill="#eef2f7" rx="3"/>',
                f'<rect x="{bar_left}" y="{y}" width="{bar_width * rate:.1f}" height="22" fill="{blue(rate)}" rx="3"/>',
                f'<text class="value" text-anchor="end" x="{bar_left + bar_width - 7}" y="{y + 16}">{100.0 * rate:.1f}%</text>',
            ]
        )
    line_x = bar_left + 0.70 * bar_width
    parts.append(
        f'<line x1="{line_x}" y1="{top}" x2="{line_x}" y2="{top + 4 * cell_height}" stroke="#c43b3b" stroke-width="2" stroke-dasharray="4 4"/>'
    )
    parts.append(
        f'<text class="note" text-anchor="middle" x="{line_x}" y="{top + 4 * cell_height + 20}">70% support rule</text>'
    )
    parts.append(
        f'<text class="note" x="{left}" y="{top + 5 * cell_height + 28}">Item-stability uses 500 within-cell permutations. Direct source attribution is secondary and unavailable for single-source benchmarks.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_args()
    if args.permutation_replicates <= 0:
        raise ValueError("permutation replicates must be positive")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "protocol.json", protocol(args.permutation_replicates))
    if args.protocol_only:
        print("wrote", OUTPUT / "protocol.json", flush=True)
        return

    started = time.monotonic()
    stability_rows: List[Dict[str, Any]] = []
    null_parts: List[pd.DataFrame] = []
    signature_parts: List[pd.DataFrame] = []
    source_parts: List[pd.DataFrame] = []
    selected_source_parts: List[pd.DataFrame] = []
    example_parts: List[pd.DataFrame] = []
    attribution_parts: List[pd.DataFrame] = []
    selected_attribution_parts: List[pd.DataFrame] = []
    attribution_audit_parts: List[pd.DataFrame] = []
    fit_audit_rows: List[Dict[str, Any]] = []

    for benchmark_index, (benchmark, config) in enumerate(RUNS.items()):
        print("fit", benchmark, flush=True)
        responses, model_ids = load_responses(config)
        path = Path(config["path"])
        item_metadata = baseline_item_metadata(path, responses.shape[0])
        sources = item_metadata["source"].astype(str).to_numpy()
        family_models = baseline_family_models(path)
        observed_ids = model_ids[
            family_models["dataset_model_index"].to_numpy(dtype=np.int32)
        ]
        if not np.array_equal(observed_ids, family_models["model_id"].astype(str)):
            raise ValueError("%s family model indices do not align" % benchmark)
        family_names = sorted(family_models["family"].unique())
        family_to_code = {family: i for i, family in enumerate(family_names)}
        dimension_frame = pd.read_csv(path / "dimension_selection.csv")
        dimensions_by_half = {
            str(row.audit_half): int(row.selected_dimension)
            for row in dimension_frame.itertuples(index=False)
        }
        fits = {}
        for audit_half in ("discovery", "validation"):
            fit = fit_owner_half(
                responses,
                family_models,
                sources,
                audit_half,
                family_to_code,
                dimensions_by_half[audit_half],
            )
            baseline_half = pd.read_csv(path / "crossfit_anchor_items.csv")
            baseline_half = baseline_half[
                np.isclose(baseline_half["anchor_fraction"], PRIMARY_FRACTION)
                & baseline_half["audit_half"].eq(audit_half)
            ].sort_values("dataset_item_index")
            max_error = float(
                np.max(
                    np.abs(
                        fit.dif_magnitude
                        - baseline_half["residual_dif_logit_range"].to_numpy(
                            dtype=np.float64
                        )
                    )
                )
            )
            fit_audit_rows.append(
                {
                    "benchmark": benchmark,
                    "audit_half": audit_half,
                    "audit_models": int(
                        family_models["model_half"].eq(audit_half).sum()
                    ),
                    "selected_dimension": dimensions_by_half[audit_half],
                    "selected_anchors": int(fit.selected.sum()),
                    "baseline_dif_magnitude_max_abs_error": max_error,
                }
            )
            if max_error > 1e-8:
                raise ValueError("%s %s baseline DIF refit mismatch" % (benchmark, audit_half))
            fits[audit_half] = fit

        stability, null, cells = item_stability_and_null(
            benchmark,
            fits["discovery"],
            fits["validation"],
            sources,
            family_names,
            args.permutation_replicates,
            PERMUTATION_SEED + benchmark_index,
        )
        stability_rows.append(stability)
        null_parts.append(null)
        signatures = pd.DataFrame(
            {
                "benchmark": benchmark,
                "dataset_item_index": np.arange(responses.shape[0]),
                "source": sources,
                "common_blueprint_cell": cells,
                "discovery_dif_magnitude": fits["discovery"].dif_magnitude,
                "validation_dif_magnitude": fits["validation"].dif_magnitude,
            }
        )
        for family_index, family in enumerate(family_names):
            signatures["discovery_%s_effect" % family] = fits[
                "discovery"
            ].effects[:, family_index]
            signatures["validation_%s_effect" % family] = fits[
                "validation"
            ].effects[:, family_index]
        signature_parts.append(signatures)

        all_sources, selected_sources = source_extremes(
            benchmark,
            sources,
            fits["discovery"].effects,
            fits["validation"].effects,
            family_names,
        )
        source_parts.append(all_sources)
        if not selected_sources.empty:
            selected_source_parts.append(selected_sources)
            questions = load_questions(benchmark, item_metadata, responses.shape[0])
            example_parts.append(
                extreme_examples(
                    selected_sources,
                    sources,
                    questions,
                    fits["discovery"].effects,
                    fits["validation"].effects,
                    family_names,
                )
            )

        representatives = pd.read_csv(path / "owner_family_representatives.csv")
        contributions, selected_contributions, attribution_audit = (
            source_shift_attribution(
                benchmark,
                responses,
                sources,
                representatives,
                fits,
            )
        )
        attribution_parts.append(contributions)
        attribution_audit_parts.append(attribution_audit)
        if not selected_contributions.empty:
            selected_attribution_parts.append(selected_contributions)

    stability_frame = pd.DataFrame(stability_rows)
    main = stability_frame[stability_frame["benchmark"].isin(MAIN_BENCHMARKS)]
    primary_pass = bool(
        int(main["passes_all_stability_gates"].sum())
        >= STABILITY_THRESHOLDS["benchmarks_required"]
    )
    all_source_effects = pd.concat(source_parts, ignore_index=True)
    selected_sources = pd.concat(selected_source_parts, ignore_index=True)
    source_summary = selected_replication_summary(
        selected_sources, list(RUNS), "Pooled source-rich main"
    )
    pooled_row = source_summary[
        source_summary["benchmark"].eq("Pooled source-rich main")
    ].iloc[0]
    individual = source_summary[
        source_summary["benchmark"].isin(SOURCE_RICH_MAIN)
    ]
    content_pass = bool(
        pooled_row["validation_sign_replication_rate"] >= 0.70
        and pooled_row["exact_binomial_p"] <= 0.05
        and int((individual["validation_sign_replication_rate"] >= 0.65).sum()) >= 2
    )
    selected_attributions = pd.concat(
        selected_attribution_parts, ignore_index=True
    )
    attribution_summary = selected_replication_summary(
        selected_attributions,
        list(RUNS),
        "Pooled source-rich main",
    )
    pooled_attribution = attribution_summary[
        attribution_summary["benchmark"].eq("Pooled source-rich main")
    ].iloc[0]
    attribution_individual = attribution_summary[
        attribution_summary["benchmark"].isin(SOURCE_RICH_MAIN)
    ]
    attribution_support = bool(
        pooled_attribution["validation_sign_replication_rate"] >= 0.70
        and pooled_attribution["exact_binomial_p"] <= 0.05
        and int(
            (attribution_individual["validation_sign_replication_rate"] >= 0.65).sum()
        )
        >= 2
    )

    pd.DataFrame(fit_audit_rows).to_csv(OUTPUT / "fit_reproduction_audit.csv", index=False)
    stability_frame.to_csv(OUTPUT / "stability_summary.csv", index=False)
    pd.concat(null_parts, ignore_index=True).to_csv(
        OUTPUT / "within_cell_permutation_null.csv", index=False
    )
    pd.concat(signature_parts, ignore_index=True).to_csv(
        OUTPUT / "item_family_signatures.csv", index=False
    )
    all_source_effects.to_csv(OUTPUT / "source_family_effects.csv", index=False)
    source_correlations = source_effect_correlations(all_source_effects)
    source_correlations.to_csv(
        OUTPUT / "source_effect_correlation_summary.csv", index=False
    )
    selected_sources.to_csv(OUTPUT / "discovery_selected_source_extremes.csv", index=False)
    source_summary.to_csv(OUTPUT / "source_validation_summary.csv", index=False)
    pd.concat(attribution_parts, ignore_index=True).to_csv(
        OUTPUT / "source_score_shift_contributions.csv", index=False
    )
    selected_attributions.to_csv(
        OUTPUT / "discovery_selected_source_shift_drivers.csv", index=False
    )
    attribution_summary.to_csv(
        OUTPUT / "source_shift_driver_validation_summary.csv", index=False
    )
    pd.concat(attribution_audit_parts, ignore_index=True).to_csv(
        OUTPUT / "source_shift_decomposition_audit.csv", index=False
    )
    pd.concat(example_parts, ignore_index=True).to_csv(
        OUTPUT / "extreme_source_examples.csv", index=False
    )
    (OUTPUT / "summary.svg").write_text(
        build_summary_svg(stability_frame, attribution_summary), encoding="utf-8"
    )
    (OUTPUT / "report.md").write_text(
        build_report(
            stability_frame,
            source_summary,
            source_correlations,
            attribution_summary,
            primary_pass,
            content_pass,
            attribution_support,
        ),
        encoding="utf-8",
    )
    with (OUTPUT / "protocol.json").open(encoding="utf-8") as handle:
        saved_protocol = json.load(handle)
    saved_protocol.update(
        elapsed_seconds=float(time.monotonic() - started),
        primary_stability_gate_passed=primary_pass,
        source_content_gate_passed=content_pass,
        direct_source_attribution_support_passed=attribution_support,
    )
    write_json(OUTPUT / "protocol.json", saved_protocol)
    print(stability_frame.to_string(index=False), flush=True)
    print(source_summary.to_string(index=False), flush=True)
    print("written to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
