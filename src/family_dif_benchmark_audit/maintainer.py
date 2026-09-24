"""Audit a maintainer-supplied binary benchmark with explicit model metadata.

This entry point reuses the released spectral estimator and ranking controls.
It accepts arbitrary family names without inferring their meaning from names.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn

from .core.spectral_mirt import (
    CANDIDATE_DIMENSIONS,
    SpectralAnchorFit,
    choose_spectral_dimension,
    nested_owner_split,
    purify_spectral_mirt_anchors,
)
from .ranking.mmlu_pro import (
    matched_random_anchor_controls,
    pairwise_ranking_metrics,
    select_owner_family_representatives,
    weighted_accuracy,
)


@dataclass(frozen=True)
class AuditConfig:
    """Primary-analysis settings; all seeds are recorded in the output."""

    dimensions: tuple[int, ...] = CANDIDATE_DIMENSIONS
    owner_cap: int = 5  # zero disables the cap
    min_family_models: int = 20
    min_family_models_per_half: int = 2
    anchor_fraction: float = 0.5
    difficulty_strata: int = 4
    purification_rounds: int = 3
    dif_penalty: float = 1.0
    dif_cycles: int = 5
    coordinate_ridge: float = 1.0
    gap_pp: float = 1.0
    resamples: int = 1000
    seed: int = 20260821
    nested_owner_seed: int = 20260826
    svd_seed: int = 20260826
    control_seed: int = 20260825
    mode: str = "primary-settings"

    def validate(self) -> None:
        integers = {
            "min_family_models": self.min_family_models,
            "min_family_models_per_half": self.min_family_models_per_half,
            "difficulty_strata": self.difficulty_strata,
            "purification_rounds": self.purification_rounds,
            "dif_cycles": self.dif_cycles,
            "resamples": self.resamples,
        }
        for name, value in integers.items():
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.owner_cap, (int, np.integer)) or self.owner_cap < 0:
            raise ValueError("owner_cap must be a nonnegative integer; zero disables it")
        if not self.dimensions or any(
            not isinstance(k, (int, np.integer)) or k <= 0 for k in self.dimensions
        ):
            raise ValueError("dimensions must contain positive integers")
        if not 0 < self.anchor_fraction < 1:
            raise ValueError("anchor_fraction must be strictly between zero and one")
        for name in ("dif_penalty", "coordinate_ridge", "gap_pp"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("seed", "nested_owner_seed", "svd_seed", "control_seed"):
            if not isinstance(getattr(self, name), (int, np.integer)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass
class BenchmarkData:
    responses: np.ndarray
    model_id: np.ndarray
    family: np.ndarray
    owner: np.ndarray
    item_group: np.ndarray
    provenance: tuple[dict[str, str], ...] = ()

    def validate(self) -> None:
        self.responses = np.asarray(self.responses)
        if self.responses.ndim != 2 or min(self.responses.shape) < 2:
            raise ValueError("responses must be a nonempty item-by-model matrix")
        if not np.isin(self.responses, [0, 1]).all():
            raise ValueError("responses must be complete binary 0/1 data; no missing values")
        self.responses = self.responses.astype(np.int8, copy=False)
        n_items, n_models = self.responses.shape
        for name, expected in (("model_id", n_models), ("family", n_models),
                               ("owner", n_models), ("item_group", n_items)):
            original = np.asarray(getattr(self, name))
            if original.shape != (expected,) or pd.isna(original).any():
                raise ValueError(f"{name} must be a complete vector of length {expected}")
            values = original.astype(str)
            if any(not value.strip() for value in values):
                raise ValueError(f"{name} must not contain blank values")
            setattr(self, name, values)
        if len(np.unique(self.model_id)) != n_models:
            raise ValueError("model_id must uniquely identify every response column")
        if len(np.unique(self.family)) < 2:
            raise ValueError("at least two explicitly supplied model families are required")
        if len(np.unique(self.owner)) < 4:
            raise ValueError("at least four independent owners are required; more are usually needed")


def _input_record(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "sha256": digest.hexdigest()}


def load_benchmark(
    path: str | Path, models: str | Path | None = None,
    items: str | Path | None = None,
) -> BenchmarkData:
    """Load NPZ vectors, or NPY plus position-aligned models/items CSV files."""
    path = Path(path)
    paths = [path]
    if path.suffix.lower() == ".npz":
        if models is not None or items is not None:
            raise ValueError("NPZ input contains its metadata; omit --models and --items")
        required = ("responses", "model_id", "family", "owner", "item_group")
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(required).difference(archive.files))
            if missing:
                raise ValueError(f"NPZ is missing required arrays: {', '.join(missing)}")
            data = BenchmarkData(**{name: archive[name] for name in required})
    elif path.suffix.lower() == ".npy":
        if models is None or items is None:
            raise ValueError("NPY input requires --models and --items CSV metadata")
        paths.extend([Path(models), Path(items)])
        model_frame = pd.read_csv(models, dtype=str, keep_default_na=False)
        item_frame = pd.read_csv(items, dtype=str, keep_default_na=False)
        missing = {"model_id", "family", "owner"}.difference(model_frame.columns)
        if missing or "item_group" not in item_frame.columns:
            raise ValueError("models needs model_id,family,owner; items needs item_group")
        data = BenchmarkData(
            np.load(path, allow_pickle=False),
            model_frame["model_id"].to_numpy(), model_frame["family"].to_numpy(),
            model_frame["owner"].to_numpy(), item_frame["item_group"].to_numpy(),
        )
    else:
        raise ValueError("input must be .npz or .npy")
    data.validate()
    data.provenance = tuple(_input_record(input_path) for input_path in paths)
    return data


def prepare_population(data: BenchmarkData, config: AuditConfig) -> pd.DataFrame:
    """Cap owner-family pairs and balance whole owners using the paper's rule.

    The population split balances counts and full-score sums. Once frozen,
    each fold's target responses never enter its dimension or anchor fitting.
    """
    data.validate()
    config.validate()
    rng = np.random.default_rng(config.seed)
    frame = pd.DataFrame({
        "dataset_model_index": np.arange(data.responses.shape[1]),
        "model_id": data.model_id, "family": data.family, "owner": data.owner,
        "train_accuracy": data.responses.mean(axis=0, dtype=np.float64),
    })
    parts = []
    families = sorted(frame.family.unique())
    for family in families:
        for _, group in frame[frame.family.eq(family)].groupby("owner", sort=True):
            if config.owner_cap and len(group) > config.owner_cap:
                chosen = np.sort(rng.choice(group.index, config.owner_cap, replace=False))
                group = group.loc[chosen]
            parts.append(group)
    selected = pd.concat(parts, ignore_index=True)
    counts = selected.groupby("family").size().reindex(families).to_numpy()
    if np.any(counts < config.min_family_models):
        raise ValueError("a family has fewer capped models than min_family_models")
    means = selected.groupby("family")["train_accuracy"].mean().reindex(families).to_numpy()
    owner_groups = []
    for owner, group in selected.groupby("owner", sort=True):
        owner_groups.append({
            "owner": owner, "n": len(group), "tie": float(rng.uniform()),
            "counts": group.groupby("family").size().reindex(families, fill_value=0).to_numpy(),
            "sums": group.groupby("family")["train_accuracy"].sum().reindex(families, fill_value=0).to_numpy(),
        })
    owner_groups.sort(key=lambda group: (-group["n"], group["tie"]))
    side_counts = np.zeros((2, len(families)))
    side_sums = np.zeros_like(side_counts)
    assignment = {}
    for group in owner_groups:
        costs = []
        for side in (0, 1):
            candidate_counts, candidate_sums = side_counts.copy(), side_sums.copy()
            candidate_counts[side] += group["counts"]
            candidate_sums[side] += group["sums"]
            expected = candidate_counts * means[None, :]
            costs.append(float(np.sum(np.abs(candidate_counts[0] - candidate_counts[1]) / counts)
                               + np.sum(np.abs((candidate_sums[0] - expected[0])
                                               - (candidate_sums[1] - expected[1])) / counts)))
        side = int(rng.integers(0, 2)) if costs[0] == costs[1] else int(np.argmin(costs))
        assignment[group["owner"]] = ("discovery", "validation")[side]
        side_counts[side] += group["counts"]
        side_sums[side] += group["sums"]
    selected["model_half"] = selected.owner.map(assignment)
    selected["family_code"] = selected.family.map({f: i for i, f in enumerate(families)})
    if np.any(side_counts < config.min_family_models_per_half):
        raise ValueError("owner-disjoint split leaves a family with too few models in a half")
    for fold, half in enumerate(("discovery", "validation")):
        try:
            nested_owner_split(selected.loc[selected.model_half.eq(half), "owner"].to_numpy(),
                               config.nested_owner_seed + fold)
        except ValueError as error:
            raise ValueError(f"{half} cannot support nested owner validation: {error}") from error
    return selected.sort_values(["family", "model_half", "owner", "model_id"]).reset_index(drop=True)


def fit_audit_half(
    data: BenchmarkData, population: pd.DataFrame, audit_half: str,
    config: AuditConfig, fold: int,
) -> tuple[SpectralAnchorFit, pd.DataFrame, dict[str, Any]]:
    """Fit only a frozen audit population; opposite-half responses are unused."""
    audit = population[population.model_half.eq(audit_half)]
    truth = data.responses[:, audit.dataset_model_index.to_numpy(dtype=int)]
    dimension, cv, selection = choose_spectral_dimension(
        truth, audit.owner.to_numpy(), data.item_group,
        dimensions=config.dimensions, owner_seed=config.nested_owner_seed + fold,
        svd_seed=config.svd_seed + fold, ridge=config.coordinate_ridge,
    )
    try:
        fit = purify_spectral_mirt_anchors(
            truth, audit.family_code.to_numpy(dtype=int), data.item_group,
            # The paper's primary 0.5 anchor is the second entry of 0.3/0.5/0.7.
            dimension=dimension, svd_seed=config.svd_seed + fold * 1000 + 100,
            difficulty_strata_count=config.difficulty_strata,
            dif_penalty=config.dif_penalty, dif_cycles=config.dif_cycles,
            rounds=config.purification_rounds, anchor_fraction=config.anchor_fraction,
        )
    except ValueError as error:
        raise ValueError(
            f"selected K={dimension} could not be fitted during anchor purification: {error}. "
            "Use more items/models or declare a smaller dimension grid; K is not silently clipped."
        ) from error
    return fit, cv, selection


def summarize_reversals(
    observed: pd.DataFrame, controls: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Pool counts, never fold percentages; withhold inference if undefined."""
    pooled = controls.groupby("replicate", sort=True).agg(
        eligible_pairs=("close_cross_family_eligible_pairs", "sum"),
        reversals=("close_cross_family_flips", "sum"),
    ).reset_index()
    pooled["reversal_rate"] = pooled.reversals / pooled.eligible_pairs.replace(0, np.nan)
    denominator = int(observed.close_cross_family_eligible_pairs.sum())
    flips = int(observed.close_cross_family_flips.sum())
    rate = flips / denominator if denominator else None
    valid = pooled.reversal_rate.notna()
    result: dict[str, Any] = {
        "eligible_pairs": denominator, "reversals": flips,
        "low_dif_reversal_rate": rate,
        "random_control_replicates": len(pooled),
        "undefined_control_replicates": int((~valid).sum()),
        "matched_random_median_rate": None, "excess_pp": None,
        "one_sided_randomization_p": None,
        "minimum_resolvable_p": 1 / (len(pooled) + 1),
        "status": "insufficient_eligible_pairs" if denominator == 0 else "undefined_random_controls",
    }
    if denominator and valid.all():
        random_rates = pooled.reversal_rate.to_numpy()
        median = float(np.median(random_rates))
        p_value = float((1 + np.sum(random_rates >= rate)) / (len(random_rates) + 1))
        excess = float(100 * (rate - median))
        result.update({
            "matched_random_median_rate": median, "excess_pp": excess,
            "one_sided_randomization_p": p_value,
            "status": "excess_sensitivity_detected" if excess > 0 and p_value <= 0.05
                      else "no_significant_excess_detected",
        })
    return result, pooled


def run_audit(data: BenchmarkData, config: AuditConfig | None = None) -> dict[str, Any]:
    """Run both owner-disjoint directions and return summary and audit tables."""
    config = config or AuditConfig()
    population = prepare_population(data, config)
    representatives = select_owner_family_representatives(population)
    observed, controls, scores, anchors, dimensions, rounds = [], [], [], [], [], []
    fold_details = []
    for fold, (audit_half, target_half) in enumerate(
        (("discovery", "validation"), ("validation", "discovery"))
    ):
        fit, cv, selection = fit_audit_half(data, population, audit_half, config, fold)
        audit = population[population.model_half.eq(audit_half)]
        target = representatives[representatives.model_half.eq(target_half)].copy()
        if set(audit.owner).intersection(target.owner):
            raise AssertionError("an owner crossed the frozen audit/target split")
        target_truth = data.responses[:, target.dataset_model_index.to_numpy(dtype=int)]
        full = target_truth.mean(axis=0, dtype=np.float64)
        low = weighted_accuracy(target_truth, fit.weights)
        metrics = pairwise_ranking_metrics(full, low, target.family.to_numpy(),
                                          target.owner.to_numpy(), config.gap_pp / 100)
        metrics.pop("close_pair_mask")
        observed.append({"audit_half": audit_half, "target_half": target_half, **metrics})
        control = matched_random_anchor_controls(
            target_truth, fit, target.family.to_numpy(), target.owner.to_numpy(),
            config.gap_pp / 100, config.resamples, config.control_seed + fold,
        )
        control.insert(0, "audit_half", audit_half)
        controls.append(control)
        target["full_score"] = full
        target["low_dif_score"] = low
        scores.append(target)
        dimensions.append(cv.assign(audit_half=audit_half))
        group_codes = pd.factorize(data.item_group, sort=True)[0]
        cell_codes = pd.factorize(fit.cells, sort=True)[0]
        anchors.append(pd.DataFrame({
            "audit_half": audit_half, "item_index": np.arange(len(data.item_group)),
            "item_group_code": group_codes, "blueprint_cell_code": cell_codes,
            "selected": fit.selected, "weight": fit.weights,
            "final_refit_dif_range": fit.dif_magnitude,
        }))
        rounds.extend({"audit_half": audit_half, **row} for row in fit.rounds)
        fold_details.append({
            "audit_half": audit_half, "target_half": target_half,
            "audit_models": len(audit), "audit_owners": audit.owner.nunique(),
            "target_representatives": len(target), "target_owners": target.owner.nunique(),
            "selected_items": int(fit.selected.sum()), "blueprint_cells": len(np.unique(fit.cells)),
            **selection,
        })
    observed_frame = pd.DataFrame(observed)
    controls_frame = pd.concat(controls, ignore_index=True)
    summary, pooled_controls = summarize_reversals(observed_frame, controls_frame)
    warnings = []
    if config.resamples < 1000:
        warnings.append("Fewer than 1,000 controls: this is a lower-resolution exploratory run.")
    if any(row["close_cross_family_eligible_pairs"] == 0 for row in observed):
        warnings.append("At least one target half has no eligible near-tie pairs.")
    if any(row["best_dimension"] == max(row["feasible_dimensions"]) for row in fold_details):
        warnings.append("At least one raw CV loss minimum reaches the feasible dimension ceiling.")
    group_sizes = pd.Series(data.item_group).value_counts()
    if group_sizes.min() < 2 * config.difficulty_strata:
        warnings.append("Some small item groups create tiny cells with few possible replacements.")
    recorded_config = asdict(config)
    defaults = asdict(AuditConfig())
    if config.mode == "primary-settings" and any(
        value != defaults[key] for key, value in recorded_config.items() if key != "mode"
    ):
        recorded_config["mode"] = "custom-settings"
    summary.update({
        "config": recorded_config, "input_provenance": list(data.provenance),
        "items": data.responses.shape[0], "input_models": data.responses.shape[1],
        "capped_models": len(population), "owners": population.owner.nunique(),
        "families": population.family.nunique(), "item_groups": len(group_sizes),
        "scored_representatives": len(representatives),
        "mean_fold_kendall_tau_b": float(observed_frame.kendall_tau_b.mean()),
        "folds": fold_details, "warnings": warnings,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__},
        "interpretation": (
            "This diagnostic tests cross-family near-tie sensitivity to low-DIF item reweighting. "
            "A nonsignificant result is not an equivalence test or a certificate of robustness. "
            "It does not establish item unfairness, a causal family mechanism, or superior scores."
        ),
    })
    # Reports retain positional indices for traceability without publishing labels.
    score_frame = pd.concat(scores, ignore_index=True)
    population_public = population.copy()
    for column in ("owner", "family"):
        aliases = {value: f"{column}_{index:04d}" for index, value in enumerate(sorted(population[column].unique()))}
        population_public[column] = population_public[column].map(aliases)
        score_frame[column] = score_frame[column].map(aliases)
    public_columns = ["dataset_model_index", "family", "owner", "model_half"]
    return {
        "summary": summary,
        "observed_by_fold": observed_frame,
        "random_controls_by_fold": controls_frame,
        "random_controls_pooled": pooled_controls,
        "scored_models": score_frame[public_columns + ["full_score", "low_dif_score"]],
        "population": population_public[public_columns],
        "anchors": pd.concat(anchors, ignore_index=True),
        "dimension_cv": pd.concat(dimensions, ignore_index=True),
        "purification": pd.DataFrame(rounds),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_audit(result: dict[str, Any], output_dir: str | Path) -> None:
    """Write a new output directory; never overwrite a previous audit."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory is not empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    summary = _json_safe(result["summary"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(output / f"{name}.csv", index=False)
    def number(value: Any, scale: float = 1) -> str:
        return "not estimable" if value is None else f"{value * scale:.3f}"
    text = [
        "# Benchmark near-tie audit", "",
        f"Status: **{summary['status']}**", "",
        f"Items: {summary['items']}; capped models: {summary['capped_models']}; "
        f"owners: {summary['owners']}; families: {summary['families']}.",
        f"Full-score gap: <= {summary['config']['gap_pp']} percentage points. "
        "Only different-owner, different-family comparisons within each target half are scored.", "",
        "| Quantity | Result |", "|---|---:|",
        f"| Eligible low-DIF pairs (strict ties excluded) | {summary['eligible_pairs']} |",
        f"| Low-DIF reversals | {summary['reversals']} |",
        f"| Low-DIF reversal rate (%) | {number(summary['low_dif_reversal_rate'], 100)} |",
        f"| Matched-random median (%) | {number(summary['matched_random_median_rate'], 100)} |",
        f"| Excess reversal (percentage points) | {number(summary['excess_pp'])} |",
        f"| One-sided randomization p | {number(summary['one_sided_randomization_p'])} |",
        f"| Random controls | {summary['random_control_replicates']} |", "",
        "Selected K by audit half: " + ", ".join(
            f"{fold['audit_half']}={fold['selected_dimension']}" for fold in summary["folds"]
        ) + ".", "", summary["interpretation"], "",
        "The full-score near-tie mask is fixed before control scoring. "
        "Each scoring rule excludes its own exact ties; rates pool counts across "
        "folds, and no between-half pairs are added. See the exported denominators.", "",
        "Population splitting balances owner-level counts and full-score sums; "
        "dimension and anchor fitting then use only the frozen audit half. "
        "This report covers the primary spectral diagnostic, not all paper validation experiments.", "",
        "Output model, owner, family, and item-group identities use indices or aliases. "
        "Input provenance records only file basenames and SHA-256 digests.",
    ]
    if summary["warnings"]:
        text += ["", "## Interpretation notes", ""] + [f"- {warning}" for warning in summary["warnings"]]
    (output / "report.md").write_text("\n".join(text) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ, or NPY with --models and --items")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--items", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="exploratory: K up to 32 and 199 random controls")
    parser.add_argument("--dimensions", help="comma-separated K candidates; overrides --quick grid")
    parser.add_argument("--resamples", type=int, help="random controls; defaults to 1000 (199 with --quick)")
    parser.add_argument("--gap-pp", type=float, default=1.0)
    parser.add_argument("--owner-cap", type=int, default=5, help="per owner-family cap; 0 disables")
    parser.add_argument("--min-family-models", type=int, default=20)
    parser.add_argument("--seed", type=int, help="master seed; otherwise use the paper's separate seeds")
    parser.add_argument("--anchor-fraction", type=float, default=0.5)
    parser.add_argument("--difficulty-strata", type=int, default=4)
    parser.add_argument("--purification-rounds", type=int, default=3)
    parser.add_argument("--dif-penalty", type=float, default=1.0)
    parser.add_argument("--coordinate-ridge", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("output directory is not empty; choose a new directory")
        settings = {
            "dimensions": tuple(int(k.strip()) for k in args.dimensions.split(",")) if args.dimensions
                          else ((1, 2, 4, 8, 16, 32) if args.quick else CANDIDATE_DIMENSIONS),
            "resamples": args.resamples if args.resamples is not None else (199 if args.quick else 1000),
            "gap_pp": args.gap_pp, "owner_cap": args.owner_cap,
            "min_family_models": args.min_family_models,
            "anchor_fraction": args.anchor_fraction, "difficulty_strata": args.difficulty_strata,
            "purification_rounds": args.purification_rounds, "dif_penalty": args.dif_penalty,
            "coordinate_ridge": args.coordinate_ridge,
            "mode": "quick-exploratory" if args.quick else "primary-settings",
        }
        if args.seed is not None:
            settings.update(seed=args.seed, nested_owner_seed=args.seed + 5,
                            svd_seed=args.seed + 5, control_seed=args.seed + 4)
        config = AuditConfig(**settings)
        data = load_benchmark(args.input, args.models, args.items)
        print(f"Auditing {data.responses.shape[0]} items and {data.responses.shape[1]} models.", flush=True)
        result = run_audit(data, config)
        write_audit(result, args.output)
        print(f"Audit complete: {result['summary']['status']}. See report.md in the requested output directory.")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Audit could not be completed: {error}\n")


if __name__ == "__main__":
    main()
