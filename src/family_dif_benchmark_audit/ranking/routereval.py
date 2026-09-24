#!/usr/bin/env python3
"""Run the frozen family-DIF ranking-impact experiment on a RouterEval benchmark.

BBH remains the default for backward compatibility.  Additional benchmarks use
the identical owner-disjoint cross-fit, anchor purification, matched-random
control, and decision gate.  Benchmarks with fewer than five native source
blocks use deterministic contiguous item clusters only for the composition
bootstrap; native sources are still used for anchor stratification.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..core.family_dif import (
    DEFAULT_FAMILIES,
    select_and_split_family_models,
)
from ..ranking.mmlu_pro import (
    build_report,
    dtf_curve_rows,
    family_shift_rows,
    matched_random_anchor_controls,
    negative_control_summary,
    owner_bootstrap_family_shifts,
    pairwise_ranking_metrics,
    percentile_ranks,
    purify_invariant_anchors,
    quantile_interval,
    select_owner_family_representatives,
    source_block_bootstrap,
    weighted_accuracy,
)
from ..discovery.qmirt_minimal import sha256, write_json


from .._paths import PROJECT_ROOT


HERE = PROJECT_ROOT
DEFAULT_MATRIX = HERE / "outputs/routereval_raw_global/all_response_matrix_q_by_m.npy"
DEFAULT_ITEMS = HERE / "outputs/routereval_raw_global/routereval_raw_50265_items.csv"
DEFAULT_MODELS = HERE / "outputs/routereval_raw_global/routereval_raw_8577_models.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--benchmark", type=str, default="bbh")
    parser.add_argument("--benchmark-label", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--item-bootstrap-clusters", type=int, default=100)
    parser.add_argument("--anchor-fractions", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--primary-anchor-fraction", type=float, default=0.5)
    parser.add_argument("--difficulty-strata", type=int, default=4)
    parser.add_argument("--purification-rounds", type=int, default=3)
    parser.add_argument("--dif-penalty", type=float, default=1.0)
    parser.add_argument("--dif-cycles", type=int, default=5)
    parser.add_argument("--close-pair-gap", type=float, default=0.01)
    parser.add_argument("--families", type=str, default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--family-seed", type=int, default=20260821)
    parser.add_argument("--owner-cap", type=int, default=5)
    parser.add_argument("--min-family-models", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--negative-control-replicates", type=int, default=1000)
    parser.add_argument("--negative-control-seed", type=int, default=20260825)
    return parser.parse_args()


def load_benchmark_inputs(
    matrix_path: Path, items_path: Path, models_path: Path, benchmark: str
) -> Dict[str, Any]:
    """Extract one unfiltered benchmark and its complete model population."""
    response_matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    item_frame = pd.read_csv(items_path)
    model_frame = pd.read_csv(models_path)
    if len(item_frame) != response_matrix.shape[0]:
        raise ValueError("item metadata and response matrix have different row counts")
    if len(model_frame) != response_matrix.shape[1]:
        raise ValueError("model metadata and response matrix have different columns")
    if not np.array_equal(
        model_frame["global_model_index"].to_numpy(), np.arange(len(model_frame))
    ):
        raise ValueError("global model metadata is not dense and ordered")
    all_model_ids = model_frame["model_id"].astype(str).to_numpy()
    raw_indices = np.asarray(
        item_frame.index[item_frame["dataset"].astype(str).eq(benchmark)],
        dtype=np.int32,
    )
    if len(raw_indices) < 1000:
        raise ValueError(
            "%s does not satisfy the frozen 1000-item rule" % benchmark
        )
    item_indices = raw_indices
    responses_all_models = np.asarray(response_matrix[item_indices])
    complete_models = (responses_all_models != -1).all(axis=0)
    responses = responses_all_models[:, complete_models].astype(np.int8)
    model_ids = all_model_ids[complete_models]
    sources = np.asarray(
        item_frame.loc[item_indices, "dataset_name"].astype(str).to_numpy()
    )
    if len(model_ids) < 1000:
        raise ValueError(
            "%s does not satisfy the frozen complete-model rule" % benchmark
        )
    if not np.all(np.isin(responses, [0, 1])):
        raise ValueError("%s complete response block is not binary" % benchmark)
    return {
        "responses": responses,
        "model_ids": model_ids,
        "sources": sources,
        "all_benchmarks_item_indices": item_indices,
        "raw_benchmark_items": int(len(raw_indices)),
        "dropped_unobserved_items": 0,
        "complete_models": int(complete_models.sum()),
    }


def composition_bootstrap_units(
    sources: np.ndarray, item_clusters: int
) -> tuple[np.ndarray, str]:
    """Choose native source blocks or deterministic contiguous item clusters."""
    sources = np.asarray(sources).astype(str)
    native_count = len(np.unique(sources))
    if native_count >= 5:
        return sources.copy(), "native_source_block"
    if item_clusters < 20:
        raise ValueError("item bootstrap requires at least 20 clusters")
    cluster_count = min(int(item_clusters), len(sources))
    cluster_index = np.floor(
        np.arange(len(sources), dtype=np.float64) * cluster_count / len(sources)
    ).astype(np.int32)
    units = np.asarray(
        ["item_cluster_%03d" % index for index in cluster_index], dtype=str
    )
    if len(np.unique(units)) != cluster_count:
        raise AssertionError("failed to construct deterministic item clusters")
    return units, "contiguous_item_cluster"


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    benchmark = args.benchmark.strip().lower()
    benchmark_label = args.benchmark_label or benchmark
    if args.output_dir is None:
        args.output_dir = Path("outputs/%s_family_ranking_impact" % benchmark)
    fractions = tuple(
        sorted({float(value.strip()) for value in args.anchor_fractions.split(",")})
    )
    if fractions != (0.3, 0.5, 0.7):
        raise ValueError("external replication freezes anchor fractions at 0.3,0.5,0.7")
    if args.primary_anchor_fraction != 0.5:
        raise ValueError("external replication freezes the primary anchor fraction at 0.5")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_benchmark_inputs(
        args.responses, args.items, args.models, benchmark
    )
    responses = np.asarray(data["responses"], dtype=np.int8)
    model_ids = np.asarray(data["model_ids"]).astype(str)
    sources = np.asarray(data["sources"]).astype(str)
    bootstrap_units, bootstrap_unit_strategy = composition_bootstrap_units(
        sources, args.item_bootstrap_clusters
    )
    requested_families = tuple(
        value.strip().lower()
        for value in args.families.split(",")
        if value.strip()
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
    family_to_code = {family: index for index, family in enumerate(family_names)}
    family_models["family_code"] = family_models["family"].map(family_to_code)
    representatives = select_owner_family_representatives(family_models)

    scored_parts: List[pd.DataFrame] = []
    observed_rows: List[Dict[str, Any]] = []
    bootstrap_parts: List[pd.DataFrame] = []
    control_parts: List[pd.DataFrame] = []
    anchor_rows: List[Dict[str, Any]] = []
    purification_rows: List[Dict[str, Any]] = []
    dtf_rows: List[Dict[str, Any]] = []

    for fraction_index, fraction in enumerate(fractions):
        for audit_half, target_half in (
            ("discovery", "validation"),
            ("validation", "discovery"),
        ):
            audit = family_models[family_models["model_half"] == audit_half]
            audit_indices = audit["dataset_model_index"].to_numpy(dtype=np.int32)
            audit_families = audit["family_code"].to_numpy(dtype=np.int32)
            print(
                "fit %s anchors fraction=%.2f audit=%s models=%d"
                % (benchmark, fraction, audit_half, len(audit)),
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
                    {"anchor_fraction": fraction, "audit_half": audit_half, **row}
                )
            for item_index in range(len(sources)):
                anchor_rows.append(
                    {
                        "anchor_fraction": fraction,
                        "audit_half": audit_half,
                        "dataset_item_index": item_index,
                        "all_benchmarks_item_index": int(
                            data["all_benchmarks_item_indices"][item_index]
                        ),
                        "source": sources[item_index],
                        "difficulty_cell": fit.cells[item_index],
                        "selected_anchor": bool(fit.selected[item_index]),
                        "anchor_weight": float(fit.weights[item_index]),
                        "dif_logit_range": float(fit.dif_magnitude[item_index]),
                        "item_intercept": float(fit.item_intercepts[item_index]),
                    }
                )
            dtf_rows.extend(dtf_curve_rows(fit, family_names, audit_half, fraction))

            target = representatives[
                representatives["model_half"] == target_half
            ].copy()
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
                    bootstrap_units,
                    fit.weights,
                    target["family"].to_numpy(),
                    target["owner"].to_numpy(),
                    args.close_pair_gap,
                    args.bootstrap_replicates,
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

    primary_scored = scored_models[
        np.isclose(scored_models["anchor_fraction"], args.primary_anchor_fraction)
    ]
    owner_summary_rows: List[Dict[str, Any]] = []
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
                "mean_score_shift_points": float(
                    100.0 * group["anchor_minus_full"].mean()
                ),
                "mean_score_shift_ci_low": score_low,
                "mean_score_shift_ci_high": score_high,
            }
        )
    owner_summary = pd.DataFrame(owner_summary_rows)

    protocol: Dict[str, Any] = {
        "experiment": "%s_crossfit_family_dif_ranking_impact_replication"
        % benchmark,
        "benchmark": benchmark_label,
        "benchmark_key": benchmark,
        "selection_rule_frozen_before_results": {
            "minimum_items": 1000,
            "minimum_complete_models": 1000,
            "minimum_owner_capped_models_per_family": args.min_family_models,
        },
        "raw_benchmark_items": data["raw_benchmark_items"],
        "observed_benchmark_items": len(responses),
        "dropped_globally_unobserved_items": data["dropped_unobserved_items"],
        "complete_model_population": data["complete_models"],
        "native_source_blocks": int(len(np.unique(sources))),
        "composition_bootstrap_units": int(len(np.unique(bootstrap_units))),
        "composition_bootstrap_unit_strategy": bootstrap_unit_strategy,
        "primary_anchor_fraction": args.primary_anchor_fraction,
        "sensitivity_anchor_fractions": [0.3, 0.7],
        "families": family_names,
        "owner_cap": args.owner_cap,
        "family_seed": args.family_seed,
        "owner_disjoint_crossfit": True,
        "ranking_population": "one median-accuracy checkpoint per owner-family",
        "anchor_audit_population": "all frozen owner-capped models in opposite half",
        "anchor_selection": "lowest family-DIF logit range within native source x Rasch difficulty stratum",
        "blueprint_weighting": "inverse within-stratum selection; exact original stratum mass",
        "difficulty_strata": args.difficulty_strata,
        "purification_rounds": args.purification_rounds,
        "dif_penalty": args.dif_penalty,
        "dif_cycles": args.dif_cycles,
        "close_pair_gap": args.close_pair_gap,
        "bootstrap": {
            "unit": bootstrap_unit_strategy,
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "paired_across_owner_folds": True,
            "close_pair_set": "frozen from observed full-benchmark score",
        },
        "owner_bootstrap": {
            "unit": "uploader owner",
            "replicates": args.bootstrap_replicates,
        },
        "matched_random_anchor_control": {
            "replicates": args.negative_control_replicates,
            "seed": args.negative_control_seed,
            "matching": "exact anchor count and inverse weight within every native source-by-easiness cell",
        },
        "frozen_go_gate": {
            "mean_primary_tau": "<=0.95",
            "primary_close_cross_family_flip_rate": ">=0.10",
            "source_bootstrap_tau_upper": "<0.98",
            "source_bootstrap_close_flip_lower": ">0.05",
            "sensitivity": "each 30/70% fraction mean tau<=0.97 and close flip rate>=0.075",
            "minimum_close_pairs": 50,
        },
        "cross_benchmark_directional_target": "descriptive only; no universal family direction is assumed",
        "interpretation_warning": "DIF is non-invariance, not automatically item bias; anchor scoring changes the estimand.",
        "elapsed_seconds": float(time.monotonic() - started),
        "inputs": {
            "responses": str(args.responses.resolve()),
            "responses_sha256": sha256(args.responses),
            "items": str(args.items.resolve()),
            "items_sha256": sha256(args.items),
            "models": str(args.models.resolve()),
            "models_sha256": sha256(args.models),
        },
    }

    pd.DataFrame(anchor_rows).to_csv(
        args.output_dir / "crossfit_anchor_items.csv", index=False
    )
    pd.DataFrame(purification_rows).to_csv(
        args.output_dir / "anchor_purification.csv", index=False
    )
    family_models.to_csv(args.output_dir / "family_models.csv", index=False)
    representatives.to_csv(
        args.output_dir / "owner_family_representatives.csv", index=False
    )
    scored_models.to_csv(args.output_dir / "crossfit_model_scores.csv", index=False)
    observed.to_csv(args.output_dir / "ranking_impact_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "composition_bootstrap.csv", index=False)
    bootstrap.to_csv(args.output_dir / "source_block_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(
        args.output_dir / "composition_bootstrap_summary.csv", index=False
    )
    bootstrap_summary.to_csv(
        args.output_dir / "source_block_bootstrap_summary.csv", index=False
    )
    controls.to_csv(args.output_dir / "matched_random_anchor_controls.csv", index=False)
    control_summary.to_csv(
        args.output_dir / "matched_random_anchor_control_summary.csv", index=False
    )
    family_shifts.to_csv(args.output_dir / "family_rank_shifts.csv", index=False)
    owner_summary.to_csv(
        args.output_dir / "owner_bootstrap_family_shifts.csv", index=False
    )
    dtf_curves.to_csv(args.output_dir / "dtf_curves.csv", index=False)
    write_json(args.output_dir / "protocol.json", protocol)
    report = build_report(
        protocol,
        observed,
        bootstrap_summary,
        family_shifts,
        dtf_curves,
        control_summary,
    ).replace(
        "# Family-DIF ranking-impact decision experiment",
        "# %s family-DIF ranking-impact replication" % benchmark_label,
        1,
    ).replace(
        "Source-block bootstrap:",
        "%s bootstrap:" % bootstrap_unit_strategy.replace("_", " ").title(),
        1,
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(observed.to_string(index=False), flush=True)
    print("written to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
