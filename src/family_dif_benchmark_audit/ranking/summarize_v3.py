#!/usr/bin/env python3
"""Synthesize the five spectral-MIRT-primary benchmark audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .._paths import PROJECT_ROOT
from ..discovery.qmirt_minimal import write_json
from .exact_common import pairwise_common_model_check


RUNS = {
    "MMLU-Pro": PROJECT_ROOT / "outputs/mmlu_pro_family_ranking_impact",
    "BBH": PROJECT_ROOT / "outputs/bbh_family_ranking_impact",
    "MMLU": PROJECT_ROOT / "outputs/mmlu_family_ranking_impact",
    "HellaSwag": PROJECT_ROOT / "outputs/hellaswag_family_ranking_impact",
    "WinoGrande": PROJECT_ROOT / "outputs/winogrande_family_ranking_impact",
}
PAIRS = (
    ("MMLU-Pro", "BBH"),
    ("MMLU", "HellaSwag"),
    ("HellaSwag", "WinoGrande"),
)
PRIMARY_FRACTION = 0.5
OUTPUT = PROJECT_ROOT / "outputs/v3_five_benchmark_mirt_primary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def exact_common_family_shift_summary(common: pd.DataFrame) -> pd.DataFrame:
    """Report every family on each frozen exact-common-model benchmark pair."""
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(20260832)
    for pair, frame in common.groupby("pair", sort=False):
        left_name, right_name = pair.split(" vs ", 1)
        owners = np.sort(frame["owner"].astype(str).unique())
        owner_rows = {
            owner: frame.index[frame["owner"].astype(str).eq(owner)].to_numpy()
            for owner in owners
        }
        for family, family_frame in frame.groupby("family", sort=True):
            left = 100.0 * float(family_frame["anchor_minus_full_left"].mean())
            right = 100.0 * float(family_frame["anchor_minus_full_right"].mean())
            boot: List[tuple[float, float, float]] = []
            for _ in range(2000):
                sampled = rng.choice(owners, size=len(owners), replace=True)
                positions = np.concatenate([owner_rows[owner] for owner in sampled])
                replicate = frame.loc[positions]
                replicate = replicate[replicate["family"].eq(family)]
                if replicate.empty:
                    continue
                left_boot = 100.0 * float(
                    replicate["anchor_minus_full_left"].mean()
                )
                right_boot = 100.0 * float(
                    replicate["anchor_minus_full_right"].mean()
                )
                boot.append((left_boot, right_boot, right_boot - left_boot))
            array = np.asarray(boot, dtype=np.float64)
            rows.append(
                {
                    "pair": pair,
                    "family": family,
                    "left_benchmark": left_name,
                    "right_benchmark": right_name,
                    "left_shift_points": left,
                    "left_ci_low": float(np.quantile(array[:, 0], 0.025)),
                    "left_ci_high": float(np.quantile(array[:, 0], 0.975)),
                    "right_shift_points": right,
                    "right_ci_low": float(np.quantile(array[:, 1], 0.025)),
                    "right_ci_high": float(np.quantile(array[:, 1], 0.975)),
                    "right_minus_left_points": right - left,
                    "difference_ci_low": float(np.quantile(array[:, 2], 0.025)),
                    "difference_ci_high": float(np.quantile(array[:, 2], 0.975)),
                    "n_common_models": len(family_frame),
                    "n_common_owners": family_frame["owner"].nunique(),
                    "replicates": len(array),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []
    primary_scores: Dict[str, pd.DataFrame] = {}
    for benchmark, path in RUNS.items():
        metrics = pd.read_csv(path / "ranking_impact_metrics.csv")
        primary = metrics[np.isclose(metrics["anchor_fraction"], PRIMARY_FRACTION)]
        scores = pd.read_csv(path / "crossfit_model_scores.csv")
        scores = scores[np.isclose(scores["anchor_fraction"], PRIMARY_FRACTION)]
        primary_scores[benchmark] = scores
        controls = pd.read_csv(
            path / "matched_random_anchor_control_summary.csv"
        ).set_index("metric")
        control = controls.loc["close_cross_family_flip_rate"]
        bootstrap = pd.read_csv(path / "source_block_bootstrap_summary.csv").iloc[0]
        dimensions = pd.read_csv(path / "dimension_selection.csv")
        anchors = pd.read_csv(
            path / "crossfit_anchor_items.csv",
            usecols=["anchor_fraction", "dataset_item_index", "source"],
        )
        anchors = anchors[np.isclose(anchors["anchor_fraction"], PRIMARY_FRACTION)]
        close_flips = int(primary["close_cross_family_flips"].sum())
        close_pairs = int(primary["close_cross_family_eligible_pairs"].sum())
        close_rate = close_flips / close_pairs
        cross_flips = int(primary["cross_family_flips"].sum())
        cross_pairs = int(primary["cross_family_eligible_pairs"].sum())
        with (path / "protocol.json").open(encoding="utf-8") as handle:
            protocol = json.load(handle)
        rows.append(
            {
                "benchmark": benchmark,
                "items": int(anchors["dataset_item_index"].nunique()),
                "native_source_blocks": int(anchors["source"].nunique()),
                "bootstrap_unit": protocol["composition_bootstrap_unit_strategy"],
                "representatives": len(scores),
                "selected_dimensions": "/".join(
                    str(int(value)) for value in dimensions["selected_dimension"]
                ),
                "mean_kendall_tau_b": float(primary["kendall_tau_b"].mean()),
                "bootstrap_tau_ci_low": float(bootstrap["tau_ci_low"]),
                "bootstrap_tau_ci_high": float(bootstrap["tau_ci_high"]),
                "close_cross_family_flips": close_flips,
                "close_cross_family_pairs": close_pairs,
                "close_cross_family_flip_rate": close_rate,
                "bootstrap_close_ci_low": float(bootstrap["close_flip_ci_low"]),
                "bootstrap_close_ci_high": float(bootstrap["close_flip_ci_high"]),
                "random_control_close_median": float(control["random_control_median"]),
                "excess_close_flip_rate": close_rate
                - float(control["random_control_median"]),
                "random_control_close_p": float(control["one_sided_randomization_p"]),
                "all_cross_family_flip_rate": cross_flips / cross_pairs,
                "family_score_shift_range_points": float(
                    controls.loc["family_score_shift_range_points", "observed_low_dif"]
                ),
                "family_score_shift_random_p": float(
                    controls.loc[
                        "family_score_shift_range_points",
                        "one_sided_randomization_p",
                    ]
                ),
            }
        )
        family = pd.read_csv(path / "owner_bootstrap_family_shifts.csv")
        family.insert(0, "benchmark", benchmark)
        family_rows.extend(family.to_dict("records"))

    summary = pd.DataFrame(rows)
    family_summary = pd.DataFrame(family_rows)
    common_parts: List[pd.DataFrame] = []
    pair_boot_parts: List[pd.DataFrame] = []
    pair_summary_parts: List[pd.DataFrame] = []
    for index, (left, right) in enumerate(PAIRS):
        common, boot, pair_summary = pairwise_common_model_check(
            left,
            right,
            primary_scores[left],
            primary_scores[right],
            20260827 + index,
        )
        common_parts.append(common)
        pair_boot_parts.append(boot)
        pair_summary_parts.append(pair_summary)
    common = pd.concat(common_parts, ignore_index=True)
    pair_boot = pd.concat(pair_boot_parts, ignore_index=True)
    pair_summary = pd.concat(pair_summary_parts, ignore_index=True)
    all_family_pair_summary = exact_common_family_shift_summary(common)

    summary.to_csv(OUTPUT / "benchmark_summary.csv", index=False)
    family_summary.to_csv(OUTPUT / "family_shift_summary.csv", index=False)
    common.to_csv(OUTPUT / "pairwise_exact_common_models.csv", index=False)
    pair_boot.to_csv(OUTPUT / "pairwise_exact_common_owner_bootstrap.csv", index=False)
    pair_summary.to_csv(
        OUTPUT / "pairwise_exact_common_qwen_phi_summary.csv", index=False
    )
    all_family_pair_summary.to_csv(
        OUTPUT / "pairwise_exact_common_family_shift_summary.csv", index=False
    )
    write_json(
        OUTPUT / "protocol.json",
        {
            "analysis_version": "submission",
            "primary_estimator": "family-label-free cross-fitted spectral MIRT",
            "runs": {
                name: "${PROJECT_ROOT}/%s" % path.relative_to(PROJECT_ROOT)
                for name, path in RUNS.items()
            },
            "primary_anchor_fraction": PRIMARY_FRACTION,
            "pairwise_exact_common_checks": [list(pair) for pair in PAIRS],
            "selection_protocol": "${PROJECT_ROOT}/configs/analysis_protocol.json",
        },
    )

    significant = int((summary["random_control_close_p"] <= 0.05).sum())
    large = int((summary["excess_close_flip_rate"] >= 0.05).sum())
    systematic = int((summary["family_score_shift_random_p"] <= 0.05).sum())
    lines = [
        "# Five-benchmark MIRT-primary synthesis",
        "",
        "Residual near-tie excess reversal is significant in %d/5 benchmarks and exceeds 5 pp in %d/5. Systematic family score-shift ranges exceed matched random controls in %d/5."
        % (significant, large, systematic),
        "",
        "| benchmark | K by audit fold | tau-b | close reversal | random | excess | p | family shift range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| %s | %s | %.4f | %.1f%% | %.1f%% | %+.1f pp | %.4f | %.2f pp |"
            % (
                row.benchmark,
                row.selected_dimensions,
                row.mean_kendall_tau_b,
                100.0 * row.close_cross_family_flip_rate,
                100.0 * row.random_control_close_median,
                100.0 * row.excess_close_flip_rate,
                row.random_control_close_p,
                row.family_score_shift_range_points,
            )
        )
    lines.extend(
        [
            "",
            "WinoGrande is the boundary case for the primary near-tie statistic: its large raw reversal rate is explained by matched shorter-test noise, while its family score-shift range remains systematic. The other four benchmarks retain significant excess near-tie sensitivity after multidimensional ability adjustment.",
            "",
            "The result supports local benchmark-dependent ranking resolution, not a universal family ordering or causal architecture effect.",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("written to", OUTPUT)


if __name__ == "__main__":
    main()
