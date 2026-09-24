#!/usr/bin/env python3
"""Decisive cross-fitted family-DIF ranking-impact experiment on MMLU-Pro.

The inferential target is not whether model families have different average
accuracy.  It is whether MMLU-Pro yields the same ordering after removing the
items with the largest owner-disjoint family DIF while preserving the original
source and difficulty blueprint.

For each owner-disjoint half, the opposite half is used to estimate Rasch
abilities and shrinkage-stabilized item-by-family effects.  Low-DIF anchors are
selected within source-by-easiness strata, and inverse selection weights make
each stratum contribute exactly its original item mass.  The frozen weighted
anchor score is then applied to one representative model per owner-family in
the held-out half.  The halves are swapped and results are combined.  Source
block bootstrap quantifies item-composition uncertainty.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import kendalltau, rankdata

from ..core.family_dif import fit_family_dif, fit_rasch_abilities
from ..discovery.qmirt_minimal import load_inputs, sha256, write_json


PRIMARY_ANCHOR_FRACTION = 0.50
SENSITIVITY_ANCHOR_FRACTIONS = (0.30, 0.70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("outputs/routereval_raw_matrices/mmlu_pro_response_matrix.npz"),
    )
    parser.add_argument(
        "--items", type=Path, default=Path("external_cache/mmlu_pro_test.parquet")
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=Path("sae_slices/mmlu_pro_categories.json"),
    )
    parser.add_argument(
        "--meta", type=Path, default=Path("sae_slices/mmlu_pro_meta.json")
    )
    parser.add_argument(
        "--family-models",
        type=Path,
        default=Path("outputs/mmlu_pro_family_dif_minimal/family_models.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mmlu_pro_family_ranking_impact"),
    )
    parser.add_argument("--anchor-fractions", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--primary-anchor-fraction", type=float, default=0.5)
    parser.add_argument("--difficulty-strata", type=int, default=4)
    parser.add_argument("--purification-rounds", type=int, default=3)
    parser.add_argument("--dif-penalty", type=float, default=1.0)
    parser.add_argument("--dif-cycles", type=int, default=5)
    parser.add_argument("--close-pair-gap", type=float, default=0.01)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--negative-control-replicates", type=int, default=1000)
    parser.add_argument("--negative-control-seed", type=int, default=20260825)
    return parser.parse_args()


def validate_family_models(
    family_models: pd.DataFrame, model_ids: np.ndarray
) -> pd.DataFrame:
    required = {
        "dataset_model_index",
        "model_id",
        "family",
        "owner",
        "model_half",
        "train_accuracy",
    }
    missing = required.difference(family_models.columns)
    if missing:
        raise ValueError("family model table is missing: %s" % sorted(missing))
    frame = family_models.copy()
    frame["dataset_model_index"] = frame["dataset_model_index"].astype(np.int32)
    for row in frame.itertuples(index=False):
        observed = str(model_ids[int(row.dataset_model_index)])
        if observed != str(row.model_id):
            raise ValueError(
                "family model index mismatch: expected %s, observed %s"
                % (row.model_id, observed)
            )
    if set(frame["model_half"]) != {"discovery", "validation"}:
        raise ValueError("expected discovery and validation owner halves")
    if frame.groupby("owner")["model_half"].nunique().max() != 1:
        raise ValueError("an owner crosses the frozen halves")
    return frame


def select_owner_family_representatives(family_models: pd.DataFrame) -> pd.DataFrame:
    """Keep one deterministic median-ability checkpoint per owner-family."""
    rows: List[pd.Series] = []
    for _, group in family_models.groupby(["owner", "family"], sort=True):
        median = float(group["train_accuracy"].median())
        ordered = group.assign(
            distance_to_owner_family_median=(group["train_accuracy"] - median).abs()
        ).sort_values(
            ["distance_to_owner_family_median", "model_id"], kind="mergesort"
        )
        rows.append(ordered.iloc[0])
    result = pd.DataFrame(rows).reset_index(drop=True)
    if result.groupby(["owner", "family"]).size().max() != 1:
        raise AssertionError("owner-family representative selection failed")
    return result.sort_values(
        ["model_half", "family", "owner", "model_id"]
    ).reset_index(drop=True)


def difficulty_cells(
    sources: np.ndarray, item_intercepts: np.ndarray, n_strata: int
) -> np.ndarray:
    """Create deterministic within-source difficulty strata of near-equal size."""
    if n_strata <= 0:
        raise ValueError("difficulty strata must be positive")
    sources = np.asarray(sources).astype(str)
    item_intercepts = np.asarray(item_intercepts, dtype=np.float64)
    if len(sources) != len(item_intercepts):
        raise ValueError("source and difficulty vectors differ")
    cells = np.empty(len(sources), dtype=object)
    for source in np.unique(sources):
        positions = np.flatnonzero(sources == source)
        order = positions[
            np.lexsort((positions, item_intercepts[positions]))
        ]
        for stratum, part in enumerate(np.array_split(order, min(n_strata, len(order)))):
            cells[part] = "%s|difficulty_%d" % (source, stratum)
    return cells.astype(str)


def stratified_anchor_weights(
    dif_magnitude: np.ndarray,
    cells: np.ndarray,
    fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select the lowest-DIF fraction per cell and restore exact cell mass."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("anchor fraction must lie strictly between zero and one")
    dif_magnitude = np.asarray(dif_magnitude, dtype=np.float64)
    cells = np.asarray(cells).astype(str)
    if len(dif_magnitude) != len(cells):
        raise ValueError("DIF magnitude and cells differ")
    selected = np.zeros(len(cells), dtype=bool)
    weights = np.zeros(len(cells), dtype=np.float64)
    for cell in np.unique(cells):
        positions = np.flatnonzero(cells == cell)
        n_selected = max(1, min(len(positions), int(round(fraction * len(positions)))))
        order = positions[np.lexsort((positions, dif_magnitude[positions]))]
        chosen = order[:n_selected]
        selected[chosen] = True
        weights[chosen] = float(len(positions)) / float(n_selected)
    if not np.isclose(weights.sum(), len(weights), rtol=0.0, atol=1e-8):
        raise AssertionError("anchor weights do not preserve total item mass")
    for cell in np.unique(cells):
        mask = cells == cell
        if not np.isclose(weights[mask].sum(), mask.sum(), rtol=0.0, atol=1e-8):
            raise AssertionError("anchor weights do not preserve a stratum")
    return selected, weights


@dataclass
class AnchorFit:
    selected: np.ndarray
    weights: np.ndarray
    effects: np.ndarray
    item_intercepts: np.ndarray
    abilities: np.ndarray
    dif_magnitude: np.ndarray
    cells: np.ndarray
    rounds: List[Dict[str, Any]]


def purify_invariant_anchors(
    responses: np.ndarray,
    family_codes: np.ndarray,
    sources: np.ndarray,
    fraction: float,
    difficulty_strata_count: int,
    dif_penalty: float,
    dif_cycles: int,
    rounds: int,
) -> AnchorFit:
    """Iteratively estimate ability on low-DIF anchors and reselect anchors."""
    responses = np.asarray(responses, dtype=np.float64)
    family_codes = np.asarray(family_codes, dtype=np.int32)
    selected = np.ones(responses.shape[0], dtype=bool)
    weights = np.ones(responses.shape[0], dtype=np.float64)
    history: List[Dict[str, Any]] = []
    previous = None
    final: Dict[str, np.ndarray] = {}
    for round_index in range(rounds):
        ability, _ = fit_rasch_abilities(responses[selected])
        item, effects = fit_family_dif(
            responses,
            ability,
            family_codes,
            int(family_codes.max()) + 1,
            dif_penalty,
            cycles=dif_cycles,
        )
        magnitude = np.ptp(effects, axis=1)
        cells = difficulty_cells(sources, item, difficulty_strata_count)
        new_selected, new_weights = stratified_anchor_weights(
            magnitude, cells, fraction
        )
        jaccard = (
            1.0
            if previous is None
            else float(np.sum(previous & new_selected) / np.sum(previous | new_selected))
        )
        history.append(
            {
                "round": round_index + 1,
                "n_anchors": int(new_selected.sum()),
                "effective_anchor_mass": float(new_weights.sum()),
                "anchor_jaccard_with_previous": jaccard,
                "median_dif_range_all": float(np.median(magnitude)),
                "median_dif_range_anchor": float(np.median(magnitude[new_selected])),
                "p90_dif_range_anchor": float(np.quantile(magnitude[new_selected], 0.90)),
            }
        )
        previous = new_selected.copy()
        selected = new_selected
        weights = new_weights
        final = {
            "effects": effects,
            "item": item,
            "ability": ability,
            "magnitude": magnitude,
            "cells": cells,
        }
    # Refit once on the final frozen anchors so the DTF curves use the final scale.
    ability, _ = fit_rasch_abilities(responses[selected])
    item, effects = fit_family_dif(
        responses,
        ability,
        family_codes,
        int(family_codes.max()) + 1,
        dif_penalty,
        cycles=dif_cycles,
    )
    magnitude = np.ptp(effects, axis=1)
    final.update(effects=effects, item=item, ability=ability, magnitude=magnitude)
    return AnchorFit(
        selected=selected,
        weights=weights,
        effects=final["effects"],
        item_intercepts=final["item"],
        abilities=final["ability"],
        dif_magnitude=final["magnitude"],
        cells=final["cells"],
        rounds=history,
    )


def weighted_accuracy(responses: np.ndarray, weights: np.ndarray) -> np.ndarray:
    responses = np.asarray(responses, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if responses.shape[0] != len(weights):
        raise ValueError("response rows and item weights differ")
    if float(weights.sum()) <= 0.0:
        raise ValueError("item weights have no positive mass")
    return weights @ responses / float(weights.sum())


def matched_random_anchor_weights(
    cells: np.ndarray,
    selected_template: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a random anchor set with the template count in every blueprint cell."""
    cells = np.asarray(cells).astype(str)
    selected_template = np.asarray(selected_template, dtype=bool)
    if len(cells) != len(selected_template):
        raise ValueError("cells and anchor template differ")
    weights = np.zeros(len(cells), dtype=np.float64)
    for cell in np.unique(cells):
        positions = np.flatnonzero(cells == cell)
        n_selected = int(selected_template[positions].sum())
        if n_selected <= 0:
            raise ValueError("every blueprint cell must contain an anchor")
        chosen = rng.choice(positions, size=n_selected, replace=False)
        weights[chosen] = float(len(positions)) / float(n_selected)
    if not np.isclose(weights.sum(), len(weights), rtol=0.0, atol=1e-8):
        raise AssertionError("random control does not preserve total item mass")
    return weights


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 1:
        return np.asarray([0.5])
    ascending_rank = rankdata(scores, method="average") - 1.0
    return ascending_rank / float(len(scores) - 1)


def pairwise_ranking_metrics(
    full_scores: np.ndarray,
    anchor_scores: np.ndarray,
    families: np.ndarray,
    owners: np.ndarray,
    close_gap: float,
    fixed_close_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute strict rank reversals; exact ties are excluded from denominators."""
    full_scores = np.asarray(full_scores, dtype=np.float64)
    anchor_scores = np.asarray(anchor_scores, dtype=np.float64)
    families = np.asarray(families).astype(str)
    owners = np.asarray(owners).astype(str)
    left, right = np.triu_indices(len(full_scores), k=1)
    eligible = owners[left] != owners[right]
    cross_family = eligible & (families[left] != families[right])
    full_difference = full_scores[left] - full_scores[right]
    anchor_difference = anchor_scores[left] - anchor_scores[right]
    non_tied = (full_difference != 0.0) & (anchor_difference != 0.0)
    reversed_pair = full_difference * anchor_difference < 0.0
    if fixed_close_mask is None:
        close = np.abs(full_difference) <= close_gap
    else:
        close = np.asarray(fixed_close_mask, dtype=bool)
        if len(close) != len(left):
            raise ValueError("fixed close-pair mask has wrong length")

    def summarize(mask: np.ndarray, prefix: str) -> Dict[str, Any]:
        denominator = mask & non_tied
        n = int(denominator.sum())
        flips = int((denominator & reversed_pair).sum())
        return {
            prefix + "_eligible_pairs": n,
            prefix + "_flips": flips,
            prefix + "_flip_rate": float(flips / n) if n else float("nan"),
        }

    tau = kendalltau(full_scores, anchor_scores, variant="b").statistic
    result: Dict[str, Any] = {"kendall_tau_b": float(tau)}
    result.update(summarize(eligible, "all"))
    result.update(summarize(cross_family, "cross_family"))
    result.update(summarize(cross_family & close, "close_cross_family"))
    result["close_pair_mask"] = close
    return result


def source_aggregates(
    responses: np.ndarray,
    sources: np.ndarray,
    anchor_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_names = np.unique(np.asarray(sources).astype(str))
    full_success = np.zeros((len(source_names), responses.shape[1]), dtype=np.float64)
    full_mass = np.zeros(len(source_names), dtype=np.float64)
    anchor_success = np.zeros_like(full_success)
    anchor_mass = np.zeros(len(source_names), dtype=np.float64)
    for source_index, source in enumerate(source_names):
        mask = np.asarray(sources).astype(str) == source
        full_success[source_index] = responses[mask].sum(axis=0)
        full_mass[source_index] = int(mask.sum())
        weights = anchor_weights[mask]
        anchor_success[source_index] = weights @ responses[mask]
        anchor_mass[source_index] = float(weights.sum())
    return source_names, full_success, full_mass, anchor_success, anchor_mass


def source_block_bootstrap(
    responses: np.ndarray,
    sources: np.ndarray,
    anchor_weights: np.ndarray,
    families: np.ndarray,
    owners: np.ndarray,
    close_gap: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    (
        source_names,
        full_success,
        full_mass,
        anchor_success,
        anchor_mass,
    ) = source_aggregates(responses, sources, anchor_weights)
    observed_full = responses.mean(axis=0)
    observed_anchor = weighted_accuracy(responses, anchor_weights)
    observed_metrics = pairwise_ranking_metrics(
        observed_full, observed_anchor, families, owners, close_gap
    )
    fixed_close_mask = np.asarray(observed_metrics["close_pair_mask"], dtype=bool)
    rng = np.random.default_rng(seed)
    source_weights = rng.multinomial(
        len(source_names), np.full(len(source_names), 1.0 / len(source_names)), size=replicates
    ).astype(np.float64)
    full_scores = (source_weights @ full_success) / (
        source_weights @ full_mass
    )[:, None]
    anchor_scores = (source_weights @ anchor_success) / (
        source_weights @ anchor_mass
    )[:, None]
    rows: List[Dict[str, Any]] = []
    for replicate in range(replicates):
        metrics = pairwise_ranking_metrics(
            full_scores[replicate],
            anchor_scores[replicate],
            families,
            owners,
            close_gap,
            fixed_close_mask=fixed_close_mask,
        )
        metrics.pop("close_pair_mask")
        metrics["replicate"] = replicate
        rows.append(metrics)
    return pd.DataFrame(rows)


def matched_random_anchor_controls(
    responses: np.ndarray,
    fit: AnchorFit,
    families: np.ndarray,
    owners: np.ndarray,
    close_gap: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Negative controls for item-count noise under matched random subtests."""
    full_scores = responses.mean(axis=0, dtype=np.float64)
    full_percentiles = percentile_ranks(full_scores)
    families = np.asarray(families).astype(str)
    owners = np.asarray(owners).astype(str)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for replicate in range(replicates):
        weights = matched_random_anchor_weights(fit.cells, fit.selected, rng)
        control_scores = weighted_accuracy(responses, weights)
        metrics = pairwise_ranking_metrics(
            full_scores, control_scores, families, owners, close_gap
        )
        metrics.pop("close_pair_mask")
        control_percentiles = percentile_ranks(control_scores)
        score_shifts = control_scores - full_scores
        percentile_shifts = control_percentiles - full_percentiles
        family_score_means = [
            float(score_shifts[families == family].mean())
            for family in np.unique(families)
        ]
        family_percentile_means = [
            float(percentile_shifts[families == family].mean())
            for family in np.unique(families)
        ]
        rows.append(
            {
                "replicate": replicate,
                "family_score_shift_range_points": 100.0
                * float(np.ptp(family_score_means)),
                "family_percentile_shift_range": float(
                    np.ptp(family_percentile_means)
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def negative_control_summary(
    observed: pd.DataFrame,
    scored_models: pd.DataFrame,
    controls: pd.DataFrame,
    primary_fraction: float,
) -> pd.DataFrame:
    """Compare low-DIF ranking changes with matched random-anchor changes."""
    primary_observed = observed[
        np.isclose(observed["anchor_fraction"], primary_fraction)
    ]
    primary_scored = scored_models[
        np.isclose(scored_models["anchor_fraction"], primary_fraction)
    ]
    observed_family_ranges: List[Tuple[float, float]] = []
    for _, fold in primary_scored.groupby("target_half"):
        family_means = fold.groupby("family").agg(
            score_shift=("anchor_minus_full", "mean"),
            percentile_shift=("percentile_shift", "mean"),
        )
        observed_family_ranges.append(
            (
                100.0 * float(np.ptp(family_means["score_shift"])),
                float(np.ptp(family_means["percentile_shift"])),
            )
        )
    pooled = controls.groupby("replicate").agg(
        kendall_tau_b=("kendall_tau_b", "mean"),
        cross_flips=("cross_family_flips", "sum"),
        cross_pairs=("cross_family_eligible_pairs", "sum"),
        close_flips=("close_cross_family_flips", "sum"),
        close_pairs=("close_cross_family_eligible_pairs", "sum"),
        family_score_shift_range_points=(
            "family_score_shift_range_points",
            "mean",
        ),
        family_percentile_shift_range=("family_percentile_shift_range", "mean"),
    )
    pooled["cross_family_flip_rate"] = pooled["cross_flips"] / pooled["cross_pairs"]
    pooled["close_cross_family_flip_rate"] = pooled["close_flips"] / pooled["close_pairs"]
    observed_values = {
        "kendall_tau_b": float(primary_observed["kendall_tau_b"].mean()),
        "cross_family_flip_rate": float(
            primary_observed["cross_family_flips"].sum()
            / primary_observed["cross_family_eligible_pairs"].sum()
        ),
        "close_cross_family_flip_rate": float(
            primary_observed["close_cross_family_flips"].sum()
            / primary_observed["close_cross_family_eligible_pairs"].sum()
        ),
        "family_score_shift_range_points": float(
            np.mean([value[0] for value in observed_family_ranges])
        ),
        "family_percentile_shift_range": float(
            np.mean([value[1] for value in observed_family_ranges])
        ),
    }
    lower_is_more_extreme = {"kendall_tau_b"}
    rows: List[Dict[str, Any]] = []
    for metric, observed_value in observed_values.items():
        control_values = pooled[metric].to_numpy(dtype=np.float64)
        if metric in lower_is_more_extreme:
            count = int(np.sum(control_values <= observed_value))
            direction = "control <= observed"
        else:
            count = int(np.sum(control_values >= observed_value))
            direction = "control >= observed"
        rows.append(
            {
                "metric": metric,
                "observed_low_dif": observed_value,
                "random_control_median": float(np.median(control_values)),
                "random_control_ci_low": float(np.quantile(control_values, 0.025)),
                "random_control_ci_high": float(np.quantile(control_values, 0.975)),
                "one_sided_randomization_p": float(
                    (count + 1) / (len(control_values) + 1)
                ),
                "extreme_direction": direction,
                "replicates": len(control_values),
            }
        )
    return pd.DataFrame(rows)


def dtf_curve_rows(
    fit: AnchorFit,
    family_names: Sequence[str],
    audit_half: str,
    anchor_fraction: float,
    theta_grid: Sequence[float] = (-2.0, -1.0, 0.0, 1.0, 2.0),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_item = fit.item_intercepts
    for theta in theta_grid:
        reference = expit(base_item + theta)
        for family_index, family in enumerate(family_names):
            family_probability = expit(
                base_item + theta + fit.effects[:, family_index]
            )
            rows.append(
                {
                    "audit_half": audit_half,
                    "anchor_fraction": anchor_fraction,
                    "family": family,
                    "theta": theta,
                    "full_dtf_probability_points": float(
                        family_probability.mean() - reference.mean()
                    ),
                    "anchor_dtf_probability_points": float(
                        np.average(
                            family_probability - reference, weights=fit.weights
                        )
                    ),
                }
            )
    return rows


def quantile_interval(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def family_shift_rows(scored_models: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (fraction, target_half, family), group in scored_models.groupby(
        ["anchor_fraction", "target_half", "family"], sort=True
    ):
        rows.append(
            {
                "anchor_fraction": fraction,
                "target_half": target_half,
                "family": family,
                "n_owner_family_representatives": len(group),
                "mean_percentile_shift": float(group["percentile_shift"].mean()),
                "median_percentile_shift": float(group["percentile_shift"].median()),
                "mean_score_shift_points": float(
                    100.0 * group["anchor_minus_full"].mean()
                ),
            }
        )
    return rows


def owner_bootstrap_family_shifts(
    scored_models: pd.DataFrame,
    primary_fraction: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    primary = scored_models[
        np.isclose(scored_models["anchor_fraction"], primary_fraction)
    ].copy()
    owners = np.sort(primary["owner"].unique())
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for replicate in range(replicates):
        sampled = rng.choice(owners, size=len(owners), replace=True)
        parts = [primary[primary["owner"] == owner] for owner in sampled]
        bootstrap = pd.concat(parts, ignore_index=True)
        for family, group in bootstrap.groupby("family"):
            rows.append(
                {
                    "replicate": replicate,
                    "family": family,
                    "mean_percentile_shift": float(group["percentile_shift"].mean()),
                    "mean_score_shift_points": float(
                        100.0 * group["anchor_minus_full"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_report(
    protocol: Mapping[str, Any],
    observed: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    family_shifts: pd.DataFrame,
    dtf_curves: pd.DataFrame,
    control_summary: pd.DataFrame,
) -> str:
    primary = float(protocol["primary_anchor_fraction"])
    primary_observed = observed[np.isclose(observed["anchor_fraction"], primary)]
    primary_boot = bootstrap_summary[
        np.isclose(bootstrap_summary["anchor_fraction"], primary)
    ].iloc[0]
    mean_tau = float(primary_observed["kendall_tau_b"].mean())
    pooled_close_flips = int(primary_observed["close_cross_family_flips"].sum())
    pooled_close_pairs = int(primary_observed["close_cross_family_eligible_pairs"].sum())
    pooled_close_rate = pooled_close_flips / pooled_close_pairs
    sensitivity = observed[~np.isclose(observed["anchor_fraction"], primary)]
    sensitivity_ok = bool(
        (sensitivity.groupby("anchor_fraction")["kendall_tau_b"].mean() <= 0.97).all()
        and (
            sensitivity.groupby("anchor_fraction").apply(
                lambda x: x["close_cross_family_flips"].sum()
                / x["close_cross_family_eligible_pairs"].sum(),
                include_groups=False,
            )
            >= 0.075
        ).all()
    )
    gate_components = {
        "mean_primary_tau_le_0.95": mean_tau <= 0.95,
        "primary_close_flip_rate_ge_0.10": pooled_close_rate >= 0.10,
        "bootstrap_tau_upper_lt_0.98": float(primary_boot["tau_ci_high"]) < 0.98,
        "bootstrap_close_flip_lower_gt_0.05": float(primary_boot["close_flip_ci_low"])
        > 0.05,
        "sensitivity_consistent": sensitivity_ok,
        "at_least_50_close_pairs": pooled_close_pairs >= 50,
    }
    go = all(gate_components.values())
    diagnostic_control_support = bool(
        (
            control_summary.set_index("metric").loc[
                [
                    "close_cross_family_flip_rate",
                    "family_score_shift_range_points",
                ],
                "one_sided_randomization_p",
            ]
            <= 0.01
        ).all()
    )
    if go:
        scope_decision = "GO: broad ranking-impact gate met"
        scope_note = "This decision follows the frozen broad ranking-impact gate."
    elif diagnostic_control_support:
        scope_decision = (
            "FORMAL NO-GO for broad reranking; NARROW GO for one external "
            "replication of the near-tie robustness claim"
        )
        scope_note = (
            "The narrow continuation label is a validity diagnosis, not a "
            "retroactive change to the frozen gate."
        )
    else:
        scope_decision = "NO-GO: broad gate failed and the local effect did not beat controls"
        scope_note = "This decision follows the frozen broad ranking-impact gate."
    lines = [
        "# Family-DIF ranking-impact decision experiment",
        "",
        "## Decision",
        "",
        "**%s**" % scope_decision,
        "",
        scope_note,
        "",
        "The 50%% blueprint-balanced invariant-anchor score produced mean Kendall "
        "tau-b %.4f across the two owner-held-out folds. Among %d cross-family "
        "model pairs separated by at most %.1f accuracy point on the full benchmark, "
        "%d flipped (%.1f%%)."
        % (
            mean_tau,
            pooled_close_pairs,
            100.0 * float(protocol["close_pair_gap"]),
            pooled_close_flips,
            100.0 * pooled_close_rate,
        ),
        "",
        "Composition-resampling 95%% percentile ranges: mean tau [%.4f, %.4f]; "
        "close-pair flip rate [%.1f%%, %.1f%%]. These ranges describe benchmark "
        "recomposition rather than confidence intervals around fixed-benchmark estimates."
        % (
            primary_boot["tau_ci_low"],
            primary_boot["tau_ci_high"],
            100.0 * primary_boot["close_flip_ci_low"],
            100.0 * primary_boot["close_flip_ci_high"],
        ),
        "",
        "## Matched-random-anchor negative control",
        "",
        "The control draws the same number of items from every source-by-easiness "
        "cell. One-sided p-values ask whether the low-DIF subtest is at least as "
        "disruptive or family-structured as random subtests of the same size.",
        "",
        "| metric | low-DIF | random median | random 95% interval | one-sided p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in control_summary.itertuples(index=False):
        lines.append(
            "| %s | %.4f | %.4f | [%.4f, %.4f] | %.4f |"
            % (
                row.metric,
                row.observed_low_dif,
                row.random_control_median,
                row.random_control_ci_low,
                row.random_control_ci_high,
                row.one_sided_randomization_p,
            )
        )
    lines.extend(
        [
        "",
        "## Frozen gate",
        "",
        "| component | passed |",
        "|---|---:|",
        ]
    )
    for component, passed in gate_components.items():
        lines.append("| %s | %s |" % (component, "yes" if passed else "no"))
    lines.extend(
        [
            "",
            "## Cross-fitted ranking results",
            "",
            "| anchor fraction | target owner half | representatives | tau-b | "
            "close cross-family flips | flip rate |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in observed.sort_values(["anchor_fraction", "target_half"]).itertuples(index=False):
        lines.append(
            "| %.0f%% | %s | %d | %.4f | %d/%d | %.1f%% |"
            % (
                100.0 * row.anchor_fraction,
                row.target_half,
                row.n_representatives,
                row.kendall_tau_b,
                row.close_cross_family_flips,
                row.close_cross_family_eligible_pairs,
                100.0 * row.close_cross_family_flip_rate,
            )
        )
    lines.extend(
        [
            "",
            "## Family shifts at the primary anchor fraction",
            "",
            "Positive percentile shift means the family moves upward under the "
            "invariant-anchor score.",
            "",
            "| family | target half | n | mean percentile shift | median shift | "
            "anchor-full score |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    primary_shifts = family_shifts[np.isclose(family_shifts["anchor_fraction"], primary)]
    for row in primary_shifts.sort_values(["family", "target_half"]).itertuples(index=False):
        lines.append(
            "| %s | %s | %d | %+.3f | %+.3f | %+.2f pp |"
            % (
                row.family,
                row.target_half,
                row.n_owner_family_representatives,
                row.mean_percentile_shift,
                row.median_percentile_shift,
                row.mean_score_shift_points,
            )
        )
    zero = dtf_curves[
        np.isclose(dtf_curves["anchor_fraction"], primary)
        & np.isclose(dtf_curves["theta"], 0.0)
    ]
    lines.extend(
        [
            "",
            "## Differential test functioning at theta=0",
            "",
            "These signed probability-point shifts use the low-DIF anchors as the "
            "ability scale; they are descriptive consequences, not proof of unfairness.",
            "",
            "| audit half | family | full-test DTF | anchor DTF |",
            "|---|---|---:|---:|",
        ]
    )
    for row in zero.sort_values(["audit_half", "family"]).itertuples(index=False):
        lines.append(
            "| %s | %s | %+.3f pp | %+.3f pp |"
            % (
                row.audit_half,
                row.family,
                100.0 * row.full_dtf_probability_points,
                100.0 * row.anchor_dtf_probability_points,
            )
        )
    lines.extend(
        [
            "",
            "DIF does not by itself establish benchmark bias: low-DIF scoring defines "
            "a common-core estimand and may intentionally remove genuine family "
            "specialization. The decision concerns whether that choice materially "
            "changes comparative claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    fractions = tuple(
        sorted({float(value.strip()) for value in args.anchor_fractions.split(",")})
    )
    if args.primary_anchor_fraction not in fractions:
        raise ValueError("primary anchor fraction must be in --anchor-fractions")
    if tuple(fractions) != (0.3, 0.5, 0.7):
        print("warning: non-frozen sensitivity fractions requested", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args.responses, args.items, args.categories, args.meta)
    responses = np.asarray(data["responses"], dtype=np.int8)
    item_frame = pd.read_parquet(args.items, columns=["src", "category", "options"])
    sources = item_frame["src"].astype(str).to_numpy()
    family_models = validate_family_models(
        pd.read_csv(args.family_models), data["model_ids"]
    )
    family_names = sorted(family_models["family"].unique())
    family_to_code = {family: index for index, family in enumerate(family_names)}
    family_models["family_code"] = family_models["family"].map(family_to_code)
    representatives = select_owner_family_representatives(family_models)

    scored_parts: List[pd.DataFrame] = []
    observed_rows: List[Dict[str, Any]] = []
    bootstrap_parts: List[pd.DataFrame] = []
    anchor_rows: List[Dict[str, Any]] = []
    purification_rows: List[Dict[str, Any]] = []
    dtf_rows: List[Dict[str, Any]] = []
    control_parts: List[pd.DataFrame] = []

    for fraction_index, fraction in enumerate(fractions):
        for audit_half, target_half in (
            ("discovery", "validation"),
            ("validation", "discovery"),
        ):
            audit = family_models[family_models["model_half"] == audit_half]
            audit_indices = audit["dataset_model_index"].to_numpy(dtype=np.int32)
            audit_families = audit["family_code"].to_numpy(dtype=np.int32)
            print(
                "fit anchors fraction=%.2f audit=%s models=%d"
                % (fraction, audit_half, len(audit)),
                flush=True,
            )
            fit = purify_invariant_anchors(
                responses[:, audit_indices],
                audit_families,
                sources,
                fraction,
                args.difficulty_strata,
                args.dif_penalty,
                args.dif_cycles,
                args.purification_rounds,
            )
            for row in fit.rounds:
                purification_rows.append(
                    {
                        "anchor_fraction": fraction,
                        "audit_half": audit_half,
                        **row,
                    }
                )
            for item_index in range(len(sources)):
                anchor_rows.append(
                    {
                        "anchor_fraction": fraction,
                        "audit_half": audit_half,
                        "dataset_item_index": item_index,
                        "source": sources[item_index],
                        "difficulty_cell": fit.cells[item_index],
                        "selected_anchor": bool(fit.selected[item_index]),
                        "anchor_weight": float(fit.weights[item_index]),
                        "dif_logit_range": float(fit.dif_magnitude[item_index]),
                        "item_intercept": float(fit.item_intercepts[item_index]),
                    }
                )
            dtf_rows.extend(
                dtf_curve_rows(fit, family_names, audit_half, fraction)
            )

            target = representatives[representatives["model_half"] == target_half].copy()
            target_indices = target["dataset_model_index"].to_numpy(dtype=np.int32)
            target_responses = responses[:, target_indices]
            full_scores = target_responses.mean(axis=0, dtype=np.float64)
            anchor_scores = weighted_accuracy(target_responses, fit.weights)
            target["anchor_fraction"] = fraction
            target["audit_half"] = audit_half
            target["target_half"] = target_half
            target["full_accuracy"] = full_scores
            target["invariant_anchor_accuracy"] = anchor_scores
            target["anchor_minus_full"] = anchor_scores - full_scores
            target["full_percentile"] = percentile_ranks(full_scores)
            target["anchor_percentile"] = percentile_ranks(anchor_scores)
            target["percentile_shift"] = (
                target["anchor_percentile"] - target["full_percentile"]
            )
            scored_parts.append(target)

            metrics = pairwise_ranking_metrics(
                full_scores,
                anchor_scores,
                target["family"].to_numpy(),
                target["owner"].to_numpy(),
                args.close_pair_gap,
            )
            metrics.pop("close_pair_mask")
            observed_rows.append(
                {
                    "anchor_fraction": fraction,
                    "audit_half": audit_half,
                    "target_half": target_half,
                    "n_audit_models": len(audit),
                    "n_representatives": len(target),
                    "n_anchors": int(fit.selected.sum()),
                    "effective_anchor_mass": float(fit.weights.sum()),
                    **metrics,
                }
            )
            if np.isclose(fraction, args.primary_anchor_fraction):
                controls = matched_random_anchor_controls(
                    target_responses,
                    fit,
                    target["family"].to_numpy(),
                    target["owner"].to_numpy(),
                    args.close_pair_gap,
                    args.negative_control_replicates,
                    args.negative_control_seed
                    + (0 if target_half == "validation" else 1),
                )
                controls["anchor_fraction"] = fraction
                controls["audit_half"] = audit_half
                controls["target_half"] = target_half
                control_parts.append(controls)
                bootstrap = source_block_bootstrap(
                    target_responses,
                    sources,
                    fit.weights,
                    target["family"].to_numpy(),
                    target["owner"].to_numpy(),
                    args.close_pair_gap,
                    args.bootstrap_replicates,
                    # Both held-out folds use the same items, so pair the source
                    # resamples across folds instead of treating item composition
                    # as two independent sources of uncertainty.
                    args.bootstrap_seed + fraction_index * 100,
                )
                bootstrap["anchor_fraction"] = fraction
                bootstrap["audit_half"] = audit_half
                bootstrap["target_half"] = target_half
                bootstrap_parts.append(bootstrap)

    scored_models = pd.concat(scored_parts, ignore_index=True)
    observed = pd.DataFrame(observed_rows)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    controls = pd.concat(control_parts, ignore_index=True)
    family_shifts = pd.DataFrame(family_shift_rows(scored_models))
    dtf_curves = pd.DataFrame(dtf_rows)
    control_summary = negative_control_summary(
        observed, scored_models, controls, args.primary_anchor_fraction
    )
    owner_bootstrap = owner_bootstrap_family_shifts(
        scored_models,
        args.primary_anchor_fraction,
        args.bootstrap_replicates,
        args.bootstrap_seed + 1000,
    )

    bootstrap_summary_rows: List[Dict[str, Any]] = []
    for fraction, group in bootstrap.groupby("anchor_fraction"):
        per_replicate = group.groupby("replicate").agg(
            kendall_tau_b=("kendall_tau_b", "mean"),
            close_flips=("close_cross_family_flips", "sum"),
            close_pairs=("close_cross_family_eligible_pairs", "sum"),
        )
        per_replicate["close_flip_rate"] = (
            per_replicate["close_flips"] / per_replicate["close_pairs"]
        )
        tau_low, tau_high = quantile_interval(per_replicate["kendall_tau_b"])
        flip_low, flip_high = quantile_interval(per_replicate["close_flip_rate"])
        bootstrap_summary_rows.append(
            {
                "anchor_fraction": fraction,
                "tau_ci_low": tau_low,
                "tau_ci_high": tau_high,
                "close_flip_ci_low": flip_low,
                "close_flip_ci_high": flip_high,
                "replicates": len(per_replicate),
            }
        )
    bootstrap_summary = pd.DataFrame(bootstrap_summary_rows)

    owner_summary_rows: List[Dict[str, Any]] = []
    primary_scored = scored_models[
        np.isclose(scored_models["anchor_fraction"], args.primary_anchor_fraction)
    ]
    for family, group in primary_scored.groupby("family"):
        boot = owner_bootstrap[owner_bootstrap["family"] == family]
        rank_low, rank_high = quantile_interval(boot["mean_percentile_shift"])
        score_low, score_high = quantile_interval(boot["mean_score_shift_points"])
        owner_summary_rows.append(
            {
                "family": family,
                "n_owner_family_representatives": len(group),
                "mean_percentile_shift": float(group["percentile_shift"].mean()),
                "mean_percentile_shift_ci_low": rank_low,
                "mean_percentile_shift_ci_high": rank_high,
                "mean_score_shift_points": float(100.0 * group["anchor_minus_full"].mean()),
                "mean_score_shift_ci_low": score_low,
                "mean_score_shift_ci_high": score_high,
            }
        )
    owner_summary = pd.DataFrame(owner_summary_rows)

    protocol: Dict[str, Any] = {
        "experiment": "mmlu_pro_crossfit_family_dif_ranking_impact",
        "primary_anchor_fraction": args.primary_anchor_fraction,
        "sensitivity_anchor_fractions": [
            value for value in fractions if value != args.primary_anchor_fraction
        ],
        "families": family_names,
        "owner_disjoint_crossfit": True,
        "ranking_population": "one median-accuracy checkpoint per owner-family",
        "anchor_audit_population": "all frozen owner-capped models in opposite half",
        "anchor_selection": "lowest family-DIF logit range within source x Rasch difficulty stratum",
        "blueprint_weighting": "inverse within-stratum selection; exact original stratum mass",
        "difficulty_strata": args.difficulty_strata,
        "purification_rounds": args.purification_rounds,
        "dif_penalty": args.dif_penalty,
        "dif_cycles": args.dif_cycles,
        "close_pair_gap": args.close_pair_gap,
        "bootstrap": {
            "unit": "item source block",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "paired_across_owner_folds": True,
            "close_pair_set": "frozen from observed full-benchmark score",
        },
        "owner_bootstrap": {
            "unit": "uploader owner",
            "replicates": args.bootstrap_replicates,
            "purpose": "family mean score and percentile shifts",
        },
        "matched_random_anchor_control": {
            "replicates": args.negative_control_replicates,
            "seed": args.negative_control_seed,
            "matching": "exact anchor count and inverse weight within every source-by-easiness cell",
            "purpose": "separate family-DIF consequences from generic shorter-test ranking noise",
        },
        "frozen_go_gate": {
            "mean_primary_tau": "<=0.95",
            "primary_close_cross_family_flip_rate": ">=0.10",
            "source_bootstrap_tau_upper": "<0.98",
            "source_bootstrap_close_flip_lower": ">0.05",
            "sensitivity": "each 30/70% fraction mean tau<=0.97 and close flip rate>=0.075",
            "minimum_close_pairs": 50,
        },
        "interpretation_warning": "DIF is non-invariance, not automatically item bias; anchor scoring changes the estimand.",
        "elapsed_seconds": float(time.monotonic() - started),
        "inputs": {
            "responses": str(args.responses.resolve()),
            "responses_sha256": sha256(args.responses),
            "items": str(args.items.resolve()),
            "items_sha256": sha256(args.items),
            "family_models": str(args.family_models.resolve()),
            "family_models_sha256": sha256(args.family_models),
        },
    }

    pd.DataFrame(anchor_rows).to_csv(args.output_dir / "crossfit_anchor_items.csv", index=False)
    pd.DataFrame(purification_rows).to_csv(
        args.output_dir / "anchor_purification.csv", index=False
    )
    scored_models.to_csv(args.output_dir / "crossfit_model_scores.csv", index=False)
    observed.to_csv(args.output_dir / "ranking_impact_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "source_block_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(
        args.output_dir / "source_block_bootstrap_summary.csv", index=False
    )
    controls.to_csv(args.output_dir / "matched_random_anchor_controls.csv", index=False)
    control_summary.to_csv(
        args.output_dir / "matched_random_anchor_control_summary.csv", index=False
    )
    family_shifts.to_csv(args.output_dir / "family_rank_shifts.csv", index=False)
    owner_summary.to_csv(args.output_dir / "owner_bootstrap_family_shifts.csv", index=False)
    dtf_curves.to_csv(args.output_dir / "dtf_curves.csv", index=False)
    representatives.to_csv(
        args.output_dir / "owner_family_representatives.csv", index=False
    )
    write_json(args.output_dir / "protocol.json", protocol)
    (args.output_dir / "report.md").write_text(
        build_report(
            protocol,
            observed,
            bootstrap_summary,
            family_shifts,
            dtf_curves,
            control_summary,
        ),
        encoding="utf-8",
    )
    print(observed.to_string(index=False), flush=True)
    print("written to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
