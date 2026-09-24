#!/usr/bin/env python3
"""Run the cross-fitted spectral-MIRT family-DIF ranking audit.

Unlike the earlier scalar-Rasch pipeline, this module learns a family-label-free
multidimensional response representation in each audit fold.  Its dimension is
chosen by nested owner-held-out and item-held-out response prediction.  Family
DIF is then estimated only after conditioning on the fitted item-by-model MIRT
offsets.  The target owner half never selects the dimension or anchor items.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .._paths import PROJECT_ROOT
from ..core.family_dif import DEFAULT_FAMILIES, select_and_split_family_models
from ..discovery.qmirt_minimal import sha256, write_json
from ..ranking.mmlu_pro import (
    family_shift_rows,
    matched_random_anchor_controls,
    negative_control_summary,
    owner_bootstrap_family_shifts,
    pairwise_ranking_metrics,
    percentile_ranks,
    quantile_interval,
    select_owner_family_representatives,
    source_block_bootstrap,
    weighted_accuracy,
)
from ..ranking.routereval import composition_bootstrap_units, load_benchmark_inputs
from ..core.spectral_mirt import (
    CANDIDATE_DIMENSIONS,
    choose_spectral_dimension,
    purify_spectral_mirt_anchors,
)


BENCHMARKS: Dict[str, str] = {
    "mmlu_pro": "MMLU-Pro",
    "bbh": "BBH",
    "mmlu": "MMLU",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--anchor-fractions", default="0.3,0.5,0.7")
    parser.add_argument("--primary-anchor-fraction", type=float, default=0.5)
    parser.add_argument("--difficulty-strata", type=int, default=4)
    parser.add_argument("--purification-rounds", type=int, default=3)
    parser.add_argument("--dif-penalty", type=float, default=1.0)
    parser.add_argument("--dif-cycles", type=int, default=5)
    parser.add_argument("--close-pair-gap", type=float, default=0.01)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--family-seed", type=int, default=20260821)
    parser.add_argument("--owner-cap", type=int, default=5)
    parser.add_argument("--min-family-models", type=int, default=20)
    parser.add_argument("--nested-owner-seed", type=int, default=20260826)
    parser.add_argument("--svd-seed", type=int, default=20260826)
    parser.add_argument("--coordinate-ridge", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--negative-control-replicates", type=int, default=1000)
    parser.add_argument("--negative-control-seed", type=int, default=20260825)
    parser.add_argument("--item-bootstrap-clusters", type=int, default=100)
    return parser.parse_args()


def load_mmlu_pro() -> Dict[str, Any]:
    response_path = (
        PROJECT_ROOT / "outputs/routereval_raw_matrices/mmlu_pro_response_matrix.npz"
    )
    item_path = PROJECT_ROOT / "external_cache/mmlu_pro_test.parquet"
    archive = np.load(response_path, allow_pickle=False)
    responses = np.asarray(archive["responses_q_by_m"], dtype=np.int8)
    model_ids = np.asarray(archive["model_ids"]).astype(str)
    items = pd.read_parquet(item_path)
    if responses.shape != (len(items), len(model_ids)):
        raise ValueError("MMLU-Pro response matrix does not match metadata")
    if not np.all(np.isin(responses, [0, 1])):
        raise ValueError("MMLU-Pro response matrix is not binary")
    return {
        "responses": responses,
        "model_ids": model_ids,
        "sources": items["src"].astype(str).to_numpy(),
        "all_benchmarks_item_indices": np.arange(len(items), dtype=np.int32),
        "raw_benchmark_items": len(items),
        "dropped_unobserved_items": 0,
        "complete_models": len(model_ids),
        "input_paths": [response_path, item_path],
    }


def load_data(benchmark: str) -> Dict[str, Any]:
    if benchmark == "mmlu_pro":
        return load_mmlu_pro()
    matrix = PROJECT_ROOT / "outputs/routereval_raw_global/all_response_matrix_q_by_m.npy"
    items = PROJECT_ROOT / "outputs/routereval_raw_global/routereval_raw_50265_items.csv"
    models = PROJECT_ROOT / "outputs/routereval_raw_global/routereval_raw_8577_models.csv"
    data = load_benchmark_inputs(matrix, items, models, benchmark)
    data["input_paths"] = [matrix, items, models]
    return data


def public_input_record(path: Path) -> Dict[str, str]:
    """Record a reproducible input hash without leaking a local user path."""
    resolved = path.resolve()
    try:
        public_path = "${PROJECT_ROOT}/%s" % resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        public_path = "${EXTERNAL_INPUT_ROOT}/%s" % resolved.name
    return {"path": public_path, "sha256": sha256(resolved)}


def summarize_bootstrap(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fraction, group in bootstrap.groupby("anchor_fraction", sort=True):
        pooled = group.groupby("replicate").agg(
            kendall_tau_b=("kendall_tau_b", "mean"),
            close_flips=("close_cross_family_flips", "sum"),
            close_pairs=("close_cross_family_eligible_pairs", "sum"),
        )
        pooled["close_flip_rate"] = pooled["close_flips"] / pooled["close_pairs"]
        tau_low, tau_high = quantile_interval(pooled["kendall_tau_b"])
        flip_low, flip_high = quantile_interval(pooled["close_flip_rate"])
        rows.append(
            {
                "anchor_fraction": fraction,
                "tau_ci_low": tau_low,
                "tau_ci_high": tau_high,
                "close_flip_ci_low": flip_low,
                "close_flip_ci_high": flip_high,
                "replicates": len(pooled),
            }
        )
    return pd.DataFrame(rows)


def summarize_owner_bootstrap(
    scored: pd.DataFrame, bootstrap: pd.DataFrame, primary_fraction: float
) -> pd.DataFrame:
    primary = scored[np.isclose(scored["anchor_fraction"], primary_fraction)]
    rows: List[Dict[str, Any]] = []
    for family, group in primary.groupby("family", sort=True):
        boot = bootstrap[bootstrap["family"] == family]
        rank_low, rank_high = quantile_interval(boot["mean_percentile_shift"])
        score_low, score_high = quantile_interval(boot["mean_score_shift_points"])
        rows.append(
            {
                "family": family,
                "n_owner_family_representatives": len(group),
                "mean_percentile_shift": float(group["percentile_shift"].mean()),
                "mean_percentile_shift_ci_low": rank_low,
                "mean_percentile_shift_ci_high": rank_high,
                "mean_score_shift_points": float(
                    100.0 * group["anchor_minus_full"].mean()
                ),
                "mean_score_shift_ci_low": score_low,
                "mean_score_shift_ci_high": score_high,
            }
        )
    return pd.DataFrame(rows)


def build_report(
    benchmark: str,
    observed: pd.DataFrame,
    control_summary: pd.DataFrame,
    dimensions: pd.DataFrame,
) -> str:
    primary = observed[np.isclose(observed["anchor_fraction"], 0.5)]
    flips = int(primary["close_cross_family_flips"].sum())
    pairs = int(primary["close_cross_family_eligible_pairs"].sum())
    rate = flips / pairs if pairs else float("nan")
    random_row = control_summary[
        control_summary["metric"] == "close_cross_family_flip_rate"
    ].iloc[0]
    selections = ", ".join(
        "%s K=%d" % (row.audit_half, row.selected_dimension)
        for row in dimensions.itertuples(index=False)
    )
    return "\n".join(
        [
            "# %s MIRT-primary family-DIF audit" % BENCHMARKS[benchmark],
            "",
            "- Selected family-label-free dimensions: %s." % selections,
            "- Mean Kendall tau-b at the 50%% anchor fraction: %.4f."
            % float(primary["kendall_tau_b"].mean()),
            "- Close cross-family reversals at 1 pp: %d/%d (%.2f%%)."
            % (flips, pairs, 100.0 * rate),
            "- Matched-random median: %.2f%%; one-sided p=%.4g."
            % (
                100.0 * float(random_row["random_control_median"]),
                float(random_row["one_sided_randomization_p"]),
            ),
            "",
            "The dimension and anchors were selected on the opposite owner half; family labels were not used to select K.",
        ]
    )


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    benchmark = args.benchmark
    label = BENCHMARKS[benchmark]
    output = args.output_dir or (
        PROJECT_ROOT / ("outputs/%s_family_ranking_impact" % benchmark)
    )
    output.mkdir(parents=True, exist_ok=True)
    fractions = tuple(
        sorted({float(value.strip()) for value in args.anchor_fractions.split(",")})
    )
    if fractions != (0.3, 0.5, 0.7) or args.primary_anchor_fraction != 0.5:
        raise ValueError("the protocol freezes anchor fractions at 0.3, 0.5, and 0.7")

    data = load_data(benchmark)
    responses = np.asarray(data["responses"], dtype=np.int8)
    model_ids = np.asarray(data["model_ids"]).astype(str)
    sources = np.asarray(data["sources"]).astype(str)
    item_indices = np.asarray(data["all_benchmarks_item_indices"], dtype=np.int32)
    bootstrap_units, bootstrap_strategy = composition_bootstrap_units(
        sources, args.item_bootstrap_clusters
    )
    requested_families = tuple(
        value.strip().lower() for value in args.families.split(",") if value.strip()
    )
    family_models = select_and_split_family_models(
        model_ids,
        responses.mean(axis=0, dtype=np.float64),
        requested_families,
        args.owner_cap,
        args.min_family_models,
        args.family_seed,
    )
    family_names = sorted(family_models["family"].unique())
    family_to_code = {name: index for index, name in enumerate(family_names)}
    family_models["family_code"] = family_models["family"].map(family_to_code)
    representatives = select_owner_family_representatives(family_models)

    scored_parts: List[pd.DataFrame] = []
    observed_rows: List[Dict[str, Any]] = []
    bootstrap_parts: List[pd.DataFrame] = []
    control_parts: List[pd.DataFrame] = []
    anchor_rows: List[Dict[str, Any]] = []
    purification_rows: List[Dict[str, Any]] = []
    cv_parts: List[pd.DataFrame] = []
    dimension_rows: List[Dict[str, Any]] = []
    coordinate_rows: List[Dict[str, Any]] = []

    folds = (("discovery", "validation"), ("validation", "discovery"))
    for fold_index, (audit_half, target_half) in enumerate(folds):
        audit = family_models[family_models["model_half"] == audit_half].copy()
        audit_indices = audit["dataset_model_index"].to_numpy(dtype=np.int32)
        audit_codes = audit["family_code"].to_numpy(dtype=np.int32)
        audit_responses = responses[:, audit_indices]
        selected_dimension, cv, selection = choose_spectral_dimension(
            audit_responses,
            audit["owner"].to_numpy(),
            sources,
            dimensions=CANDIDATE_DIMENSIONS,
            owner_seed=args.nested_owner_seed + fold_index,
            svd_seed=args.svd_seed + fold_index,
            ridge=args.coordinate_ridge,
        )
        cv.insert(0, "benchmark", label)
        cv.insert(1, "audit_half", audit_half)
        cv_parts.append(cv)
        dimension_rows.append(
            {
                "benchmark": label,
                "audit_half": audit_half,
                "target_half": target_half,
                **selection,
            }
        )
        print(
            "%s audit=%s selected K=%d (best K=%d)"
            % (label, audit_half, selected_dimension, selection["best_dimension"]),
            flush=True,
        )

        for fraction_index, fraction in enumerate(fractions):
            print(
                "%s audit=%s fraction=%.1f" % (label, audit_half, fraction),
                flush=True,
            )
            fit = purify_spectral_mirt_anchors(
                audit_responses,
                audit_codes,
                sources,
                selected_dimension,
                args.svd_seed + fold_index * 1000 + fraction_index * 100,
                difficulty_strata_count=args.difficulty_strata,
                dif_penalty=args.dif_penalty,
                dif_cycles=args.dif_cycles,
                rounds=args.purification_rounds,
                anchor_fraction=fraction,
            )
            for row in fit.rounds:
                purification_rows.append(
                    {
                        "benchmark": label,
                        "anchor_fraction": fraction,
                        "audit_half": audit_half,
                        "target_half": target_half,
                        **row,
                    }
                )
            for item_position in range(len(sources)):
                anchor_rows.append(
                    {
                        "benchmark": label,
                        "anchor_fraction": fraction,
                        "audit_half": audit_half,
                        "target_half": target_half,
                        "dataset_item_index": item_position,
                        "all_benchmarks_item_index": int(item_indices[item_position]),
                        "source": sources[item_position],
                        "selected_dimension": selected_dimension,
                        "difficulty_cell": fit.cells[item_position],
                        "selected_anchor": bool(fit.selected[item_position]),
                        "anchor_weight": float(fit.weights[item_position]),
                        "dif_logit_range": float(fit.dif_magnitude[item_position]),
                        "residual_dif_logit_range": float(
                            fit.dif_magnitude[item_position]
                        ),
                        "item_intercept": float(fit.item_intercepts[item_position]),
                    }
                )
            if np.isclose(fraction, args.primary_anchor_fraction):
                for model_position, row in enumerate(audit.itertuples(index=False)):
                    for axis in range(selected_dimension):
                        coordinate_rows.append(
                            {
                                "benchmark": label,
                                "audit_half": audit_half,
                                "model_id": row.model_id,
                                "owner": row.owner,
                                "family": row.family,
                                "axis": axis + 1,
                                "coordinate": float(
                                    fit.model_axes[model_position, axis]
                                ),
                            }
                        )

            target = representatives[
                representatives["model_half"] == target_half
            ].copy()
            target_indices = target["dataset_model_index"].to_numpy(dtype=np.int32)
            target_responses = responses[:, target_indices]
            full_scores = target_responses.mean(axis=0, dtype=np.float64)
            anchor_scores = weighted_accuracy(target_responses, fit.weights)
            target["benchmark"] = label
            target["estimator"] = "spectral_mirt_primary"
            target["selected_dimension"] = selected_dimension
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
                    "benchmark": label,
                    "estimator": "spectral_mirt_primary",
                    "selected_dimension": selected_dimension,
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
                    args.negative_control_seed + fold_index,
                )
                controls["benchmark"] = label
                controls["anchor_fraction"] = fraction
                controls["audit_half"] = audit_half
                controls["target_half"] = target_half
                control_parts.append(controls)
                bootstrap = source_block_bootstrap(
                    target_responses,
                    bootstrap_units,
                    fit.weights,
                    target["family"].to_numpy(),
                    target["owner"].to_numpy(),
                    args.close_pair_gap,
                    args.bootstrap_replicates,
                    args.bootstrap_seed + fold_index,
                )
                bootstrap["benchmark"] = label
                bootstrap["anchor_fraction"] = fraction
                bootstrap["audit_half"] = audit_half
                bootstrap["target_half"] = target_half
                bootstrap_parts.append(bootstrap)

    scored = pd.concat(scored_parts, ignore_index=True)
    observed = pd.DataFrame(observed_rows)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    controls = pd.concat(control_parts, ignore_index=True)
    dimensions = pd.DataFrame(dimension_rows)
    family_shifts = pd.DataFrame(family_shift_rows(scored))
    control_summary = negative_control_summary(
        observed, scored, controls, args.primary_anchor_fraction
    )
    owner_bootstrap = owner_bootstrap_family_shifts(
        scored,
        args.primary_anchor_fraction,
        args.bootstrap_replicates,
        args.bootstrap_seed + 1000,
    )
    bootstrap_summary = summarize_bootstrap(bootstrap)
    owner_summary = summarize_owner_bootstrap(
        scored, owner_bootstrap, args.primary_anchor_fraction
    )

    protocol: Dict[str, Any] = {
        "experiment": "%s_spectral_mirt_primary" % benchmark,
        "benchmark": label,
        "benchmark_key": benchmark,
        "estimator": "family-label-free cross-fitted spectral MIRT",
        "candidate_dimensions": list(CANDIDATE_DIMENSIONS),
        "dimension_selection": "one-SE owner-held-out and item-held-out predictive log loss",
        "primary_anchor_fraction": args.primary_anchor_fraction,
        "sensitivity_anchor_fractions": [0.3, 0.7],
        "families": family_names,
        "owner_cap": args.owner_cap,
        "owner_disjoint_crossfit": True,
        "ranking_population": "one median-accuracy checkpoint per owner-family",
        "anchor_selection": "lowest residual family-DIF logit range within source-by-MIRT-residual-easiness stratum",
        "family_labels_used_to_select_dimension": False,
        "dif_penalty": args.dif_penalty,
        "purification_rounds": args.purification_rounds,
        "close_pair_gap": args.close_pair_gap,
        "composition_bootstrap_unit_strategy": bootstrap_strategy,
        "composition_bootstrap_units": int(len(np.unique(bootstrap_units))),
        "bootstrap_replicates": args.bootstrap_replicates,
        "matched_random_anchor_replicates": args.negative_control_replicates,
        "raw_benchmark_items": data["raw_benchmark_items"],
        "observed_benchmark_items": len(responses),
        "complete_model_population": data["complete_models"],
        "selected_dimensions": {
            row["audit_half"]: int(row["selected_dimension"])
            for row in dimension_rows
        },
        "seeds": {
            "family_split": args.family_seed,
            "nested_owner_validation": args.nested_owner_seed,
            "svd": args.svd_seed,
            "composition_bootstrap_base": args.bootstrap_seed,
            "matched_random_base": args.negative_control_seed,
            "owner_bootstrap": args.bootstrap_seed + 1000,
        },
        "interpretation_warning": "Residual DIF is measurement non-invariance, not a causal lineage effect; anchor scoring changes the item-composition estimand.",
        "elapsed_seconds": float(time.monotonic() - started),
        "inputs": [public_input_record(path) for path in data["input_paths"]],
    }

    pd.DataFrame(anchor_rows).to_csv(output / "crossfit_anchor_items.csv", index=False)
    pd.DataFrame(purification_rows).to_csv(
        output / "anchor_purification.csv", index=False
    )
    pd.concat(cv_parts, ignore_index=True).to_csv(output / "dimension_cv.csv", index=False)
    dimensions.to_csv(output / "dimension_selection.csv", index=False)
    pd.DataFrame(coordinate_rows).to_csv(
        output / "crossfit_model_coordinates.csv", index=False
    )
    family_models.to_csv(output / "family_models.csv", index=False)
    representatives.to_csv(output / "owner_family_representatives.csv", index=False)
    scored.to_csv(output / "crossfit_model_scores.csv", index=False)
    observed.to_csv(output / "ranking_impact_metrics.csv", index=False)
    bootstrap.to_csv(output / "composition_bootstrap.csv", index=False)
    bootstrap.to_csv(output / "source_block_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(output / "composition_bootstrap_summary.csv", index=False)
    bootstrap_summary.to_csv(output / "source_block_bootstrap_summary.csv", index=False)
    controls.to_csv(output / "matched_random_anchor_controls.csv", index=False)
    control_summary.to_csv(
        output / "matched_random_anchor_control_summary.csv", index=False
    )
    family_shifts.to_csv(output / "family_rank_shifts.csv", index=False)
    owner_bootstrap.to_csv(output / "owner_bootstrap_family_shift_replicates.csv", index=False)
    owner_summary.to_csv(output / "owner_bootstrap_family_shifts.csv", index=False)
    write_json(output / "protocol.json", protocol)
    (output / "report.md").write_text(
        build_report(benchmark, observed, control_summary, dimensions),
        encoding="utf-8",
    )
    print(observed.to_string(index=False), flush=True)
    print("written to", output, flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
