"""Family-blind spectral MIRT adjustment and residual family-DIF fitting.

This module contains the estimator used by the released analysis.  It learns a
low-rank item--model response representation without family labels, selects
its dimension by nested owner- and item-held-out prediction, and estimates
residual item-by-family effects conditional on the resulting offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.utils.extmath import randomized_svd

from ..ranking.mmlu_pro import difficulty_cells, stratified_anchor_weights


CANDIDATE_DIMENSIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PRIMARY_FRACTION = 0.50


def _validate_offsets(responses: np.ndarray, offsets: np.ndarray) -> None:
    if responses.ndim != 2 or offsets.shape != responses.shape:
        raise ValueError("responses and ability offsets must have the same 2D shape")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("ability offsets contain non-finite values")


def fit_item_intercepts_with_offsets(
    responses: np.ndarray,
    ability_offsets: np.ndarray,
    family_codes: np.ndarray | None = None,
    family_effects: np.ndarray | None = None,
    initial: np.ndarray | None = None,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> np.ndarray:
    """Fit one item intercept given fixed item-by-model ability offsets."""
    truth = np.asarray(responses, dtype=np.float64)
    offsets = np.asarray(ability_offsets, dtype=np.float64)
    _validate_offsets(truth, offsets)
    if initial is None:
        mean = np.clip(truth.mean(axis=1), 1e-4, 1.0 - 1e-4)
        intercept = np.log(mean / (1.0 - mean)) - offsets.mean(axis=1)
    else:
        intercept = np.asarray(initial, dtype=np.float64).copy()
    extra: float | np.ndarray = 0.0
    if family_codes is not None and family_effects is not None:
        codes = np.asarray(family_codes, dtype=np.int32)
        extra = np.asarray(family_effects, dtype=np.float64)[:, codes]
    for _ in range(max_iter):
        probability = expit(intercept[:, None] + offsets + extra)
        gradient = (truth - probability).sum(axis=1)
        information = (probability * (1.0 - probability)).sum(axis=1) + 1e-9
        step = np.clip(gradient / information, -2.0, 2.0)
        intercept += step
        if float(np.max(np.abs(step))) < tol:
            break
    return intercept


def fit_family_dif_with_offsets(
    responses: np.ndarray,
    ability_offsets: np.ndarray,
    family_codes: np.ndarray,
    n_families: int,
    penalty: float,
    cycles: int = 5,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate centered residual item-by-family effects after fixed offsets."""
    if penalty <= 0.0:
        raise ValueError("DIF penalty must be positive")
    truth = np.asarray(responses, dtype=np.float64)
    offsets = np.asarray(ability_offsets, dtype=np.float64)
    codes = np.asarray(family_codes, dtype=np.int32)
    _validate_offsets(truth, offsets)
    if codes.shape != (truth.shape[1],):
        raise ValueError("family codes do not align with response columns")
    item = fit_item_intercepts_with_offsets(truth, offsets)
    effects = np.zeros((truth.shape[0], n_families), dtype=np.float64)
    for _ in range(cycles):
        item = fit_item_intercepts_with_offsets(
            truth,
            offsets,
            family_codes=codes,
            family_effects=effects,
            initial=item,
            max_iter=max_iter,
            tol=tol,
        )
        largest = 0.0
        for family in range(n_families):
            model_mask = codes == family
            family_truth = truth[:, model_mask]
            base = item[:, None] + offsets[:, model_mask]
            delta = effects[:, family]
            for _ in range(max_iter):
                probability = expit(base + delta[:, None])
                gradient = (family_truth - probability).sum(axis=1) - penalty * delta
                information = (
                    (probability * (1.0 - probability)).sum(axis=1) + penalty
                )
                step = np.clip(gradient / information, -2.0, 2.0)
                delta += step
                if float(np.max(np.abs(step))) < tol:
                    break
            effects[:, family] = delta
            largest = max(largest, float(np.max(np.abs(step))))
        center = effects.mean(axis=1)
        effects -= center[:, None]
        item += center
        if largest < tol:
            break
    return item, effects


def smoothed_item_parameters(responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(responses, dtype=np.float64)
    successes = truth.sum(axis=1)
    probability = (successes + 0.5) / (truth.shape[1] + 1.0)
    probability = np.clip(probability, 1e-4, 1.0 - 1e-4)
    scale = np.sqrt(probability * (1.0 - probability))
    return probability, scale


def working_responses(
    responses: np.ndarray, probability: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    truth = np.asarray(responses, dtype=np.float64)
    return (truth - probability[:, None]) / scale[:, None]


def fit_spectral_mirt_offsets(
    responses: np.ndarray,
    selected: np.ndarray,
    dimension: int,
    seed: int,
    maximum_offset: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learn model axes on anchors, then project every item onto those axes."""
    truth = np.asarray(responses, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool)
    if truth.ndim != 2 or selected.shape != (truth.shape[0],):
        raise ValueError("anchor mask does not align with responses")
    if dimension <= 0 or dimension >= min(int(selected.sum()), truth.shape[1]):
        raise ValueError("invalid spectral MIRT dimension")
    probability, scale = smoothed_item_parameters(truth)
    working = working_responses(truth, probability, scale)
    _, _, right = randomized_svd(
        working[selected],
        n_components=dimension,
        n_iter=5,
        random_state=seed,
        flip_sign=True,
    )
    model_axes = right.T
    item_loadings = working @ model_axes
    reconstructed = item_loadings @ model_axes.T
    offsets = np.clip(
        reconstructed / scale[:, None], -maximum_offset, maximum_offset
    )
    return offsets, model_axes, item_loadings


def stratified_calibration_mask(
    blueprint_sources: np.ndarray, seed: int
) -> np.ndarray:
    sources = np.asarray(blueprint_sources).astype(str)
    rng = np.random.default_rng(seed)
    calibration = np.zeros(len(sources), dtype=bool)
    for source in np.unique(sources):
        positions = np.flatnonzero(sources == source)
        order = rng.permutation(positions)
        count = max(1, min(len(positions) - 1, int(round(0.5 * len(positions)))))
        if len(positions) == 1:
            calibration[positions] = True
        else:
            calibration[order[:count]] = True
    if calibration.all() or (~calibration).all():
        raise ValueError("calibration/evaluation item split is degenerate")
    return calibration


def nested_owner_split(owners: np.ndarray, seed: int) -> np.ndarray:
    owners = np.asarray(owners).astype(str)
    unique = np.unique(owners)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    validation_count = max(1, min(len(unique) - 1, int(round(0.25 * len(unique)))))
    validation_owners = set(shuffled[:validation_count])
    mask = np.asarray([owner in validation_owners for owner in owners], dtype=bool)
    if mask.sum() < 4 or (~mask).sum() < 8:
        raise ValueError("nested owner split has too few models")
    return mask


def binary_log_loss_by_model(truth: np.ndarray, logits: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    return np.mean(np.logaddexp(0.0, logits) - truth * logits, axis=0)


def choose_spectral_dimension(
    responses: np.ndarray,
    owners: np.ndarray,
    blueprint_sources: np.ndarray,
    dimensions: Sequence[int] = CANDIDATE_DIMENSIONS,
    owner_seed: int = 20260826,
    svd_seed: int = 20260826,
    ridge: float = 1.0,
) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
    """Choose K using nested owner-held-out and item-held-out prediction."""
    truth = np.asarray(responses, dtype=np.float64)
    validation_models = nested_owner_split(owners, owner_seed)
    fit_truth = truth[:, ~validation_models]
    validation_truth = truth[:, validation_models]
    probability, scale = smoothed_item_parameters(fit_truth)
    working_fit = working_responses(fit_truth, probability, scale)
    requested_dimensions = tuple(sorted({int(value) for value in dimensions}))
    if not requested_dimensions or requested_dimensions[0] <= 0:
        raise ValueError("candidate dimensions must be positive")
    # A truncated SVD cannot request more components than either side of the
    # fitting matrix.  Filtering here implements the paper's stated policy of
    # omitting rank-infeasible candidates while preserving one common grid.
    fitting_matrix_rank_bound = min(working_fit.shape)
    feasible_dimensions = tuple(
        value for value in requested_dimensions if value <= fitting_matrix_rank_bound
    )
    if not feasible_dimensions:
        raise ValueError("no candidate dimension is feasible for the fitting matrix")
    maximum = max(feasible_dimensions)
    left, singular, _ = randomized_svd(
        working_fit,
        n_components=maximum,
        n_iter=5,
        random_state=svd_seed,
        flip_sign=True,
    )
    item_factors = left * singular[None, :]
    working_validation = working_responses(validation_truth, probability, scale)
    calibration = stratified_calibration_mask(blueprint_sources, svd_seed)
    evaluation = ~calibration
    rows: List[Dict[str, Any]] = []
    for dimension in feasible_dimensions:
        design = item_factors[calibration, :dimension]
        gram = design.T @ design + ridge * np.eye(dimension)
        coordinates = np.linalg.solve(
            gram, design.T @ working_validation[calibration]
        )
        standardized_prediction = item_factors[evaluation, :dimension] @ coordinates
        offset = np.clip(
            standardized_prediction / scale[evaluation, None], -6.0, 6.0
        )
        predicted_logits = logit(probability[evaluation])[:, None] + offset
        losses = binary_log_loss_by_model(validation_truth[evaluation], predicted_logits)
        rows.append(
            {
                "dimension": int(dimension),
                "mean_log_loss": float(losses.mean()),
                "se_across_validation_models": float(
                    losses.std(ddof=1) / np.sqrt(len(losses))
                ),
                "n_fit_models": int((~validation_models).sum()),
                "n_validation_models": int(validation_models.sum()),
                "n_calibration_items": int(calibration.sum()),
                "n_evaluation_items": int(evaluation.sum()),
            }
        )
    frame = pd.DataFrame(rows).sort_values("dimension").reset_index(drop=True)
    best_row = frame.loc[frame["mean_log_loss"].idxmin()]
    threshold = float(
        best_row["mean_log_loss"] + best_row["se_across_validation_models"]
    )
    selected = int(frame.loc[frame["mean_log_loss"] <= threshold, "dimension"].min())
    frame["one_se_threshold"] = threshold
    frame["selected"] = frame["dimension"].eq(selected)
    audit = {
        "requested_dimensions": list(requested_dimensions),
        "feasible_dimensions": list(feasible_dimensions),
        "fitting_matrix_rank_bound": int(fitting_matrix_rank_bound),
        "best_dimension": int(best_row["dimension"]),
        "selected_dimension": selected,
        "one_se_threshold": threshold,
        "validation_owner_count": int(
            len(np.unique(np.asarray(owners).astype(str)[validation_models]))
        ),
    }
    return selected, frame, audit


@dataclass
class SpectralAnchorFit:
    selected: np.ndarray
    weights: np.ndarray
    effects: np.ndarray
    item_intercepts: np.ndarray
    dif_magnitude: np.ndarray
    cells: np.ndarray
    model_axes: np.ndarray
    item_loadings: np.ndarray
    dimension: int
    rounds: List[Dict[str, Any]]


def purify_spectral_mirt_anchors(
    responses: np.ndarray,
    family_codes: np.ndarray,
    blueprint_sources: np.ndarray,
    dimension: int,
    svd_seed: int,
    difficulty_strata_count: int = 4,
    dif_penalty: float = 1.0,
    dif_cycles: int = 5,
    rounds: int = 3,
    anchor_fraction: float = PRIMARY_FRACTION,
) -> SpectralAnchorFit:
    """Alternate spectral adjustment, residual DIF, and low-DIF selection."""
    truth = np.asarray(responses, dtype=np.float64)
    codes = np.asarray(family_codes, dtype=np.int32)
    blueprint = np.asarray(blueprint_sources).astype(str)
    selected = np.ones(truth.shape[0], dtype=bool)
    weights = np.ones(truth.shape[0], dtype=np.float64)
    previous: np.ndarray | None = None
    history: List[Dict[str, Any]] = []
    cells = np.empty(truth.shape[0], dtype=str)
    for round_index in range(rounds):
        offsets, axes, loadings = fit_spectral_mirt_offsets(
            truth, selected, dimension, svd_seed + round_index
        )
        item, effects = fit_family_dif_with_offsets(
            truth,
            offsets,
            codes,
            int(codes.max()) + 1,
            dif_penalty,
            cycles=dif_cycles,
        )
        magnitude = np.ptp(effects, axis=1)
        cells = difficulty_cells(blueprint, item, difficulty_strata_count)
        new_selected, new_weights = stratified_anchor_weights(
            magnitude, cells, anchor_fraction
        )
        jaccard = (
            1.0
            if previous is None
            else float(
                np.sum(previous & new_selected) / np.sum(previous | new_selected)
            )
        )
        history.append(
            {
                "round": round_index + 1,
                "dimension": dimension,
                "n_anchors": int(new_selected.sum()),
                "anchor_jaccard_with_previous": jaccard,
                "median_dif_range_all": float(np.median(magnitude)),
                "median_dif_range_anchor": float(np.median(magnitude[new_selected])),
            }
        )
        previous = new_selected.copy()
        selected = new_selected
        weights = new_weights
    offsets, axes, loadings = fit_spectral_mirt_offsets(
        truth, selected, dimension, svd_seed + rounds
    )
    item, effects = fit_family_dif_with_offsets(
        truth,
        offsets,
        codes,
        int(codes.max()) + 1,
        dif_penalty,
        cycles=dif_cycles,
    )
    magnitude = np.ptp(effects, axis=1)
    return SpectralAnchorFit(
        selected=selected,
        weights=weights,
        effects=effects,
        item_intercepts=item,
        dif_magnitude=magnitude,
        cells=cells,
        model_axes=axes,
        item_loadings=loadings,
        dimension=dimension,
        rounds=history,
    )
