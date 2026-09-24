"""Exact-common-model checks used by the five-benchmark synthesis."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


BOOTSTRAP_REPLICATES = 2000


def pairwise_common_model_check(
    left_name: str,
    right_name: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare benchmark shifts on exactly shared models and owners."""
    keys = ["model_id", "owner", "family"]
    value = "anchor_minus_full"
    merged = left[keys + [value]].merge(
        right[keys + [value]],
        on=keys,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    family_counts = merged["family"].value_counts()
    if not {"qwen", "phi"}.issubset(family_counts.index):
        raise ValueError(
            "pair %s/%s lacks Qwen or Phi exact common models"
            % (left_name, right_name)
        )
    means = 100.0 * merged.groupby("family")[[
        value + "_left",
        value + "_right",
    ]].mean()
    observed_left = float(
        means.loc["qwen", value + "_left"] - means.loc["phi", value + "_left"]
    )
    observed_right = float(
        means.loc["qwen", value + "_right"] - means.loc["phi", value + "_right"]
    )

    owners = np.sort(merged["owner"].astype(str).unique())
    owner_rows = {
        owner: merged.index[merged["owner"].astype(str).eq(owner)].to_numpy()
        for owner in owners
    }
    rng = np.random.default_rng(seed)
    bootstrap_rows: List[Dict[str, Any]] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(owners, size=len(owners), replace=True)
        positions = np.concatenate([owner_rows[owner] for owner in sampled])
        boot = merged.loc[positions]
        boot_means = 100.0 * boot.groupby("family")[[
            value + "_left",
            value + "_right",
        ]].mean()
        if not {"qwen", "phi"}.issubset(boot_means.index):
            continue
        left_contrast = float(
            boot_means.loc["qwen", value + "_left"]
            - boot_means.loc["phi", value + "_left"]
        )
        right_contrast = float(
            boot_means.loc["qwen", value + "_right"]
            - boot_means.loc["phi", value + "_right"]
        )
        bootstrap_rows.append(
            {
                "pair": "%s vs %s" % (left_name, right_name),
                "replicate": replicate,
                "left_qwen_minus_phi_points": left_contrast,
                "right_qwen_minus_phi_points": right_contrast,
                "right_minus_left_points": right_contrast - left_contrast,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary_rows: List[Dict[str, Any]] = []
    for label, estimate, column in (
        (left_name, observed_left, "left_qwen_minus_phi_points"),
        (right_name, observed_right, "right_qwen_minus_phi_points"),
        (
            "%s - %s" % (right_name, left_name),
            observed_right - observed_left,
            "right_minus_left_points",
        ),
    ):
        summary_rows.append(
            {
                "pair": "%s vs %s" % (left_name, right_name),
                "benchmark_or_difference": label,
                "estimate_points": estimate,
                "owner_bootstrap_ci_low": float(bootstrap[column].quantile(0.025)),
                "owner_bootstrap_ci_high": float(bootstrap[column].quantile(0.975)),
                "n_common_models": len(merged),
                "n_common_owners": merged["owner"].nunique(),
                "qwen_common_models": int(family_counts.get("qwen", 0)),
                "phi_common_models": int(family_counts.get("phi", 0)),
                "replicates": len(bootstrap),
            }
        )
    merged = merged.assign(pair="%s vs %s" % (left_name, right_name))
    return merged, bootstrap, pd.DataFrame(summary_rows)
