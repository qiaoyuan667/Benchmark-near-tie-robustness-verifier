#!/usr/bin/env python3
"""Minimal fixed-Q bifactor experiment on the audited MMLU-Pro matrix.

The experiment asks one deliberately small question: after controlling for a
general model ability and a cold-start estimate of item easiness, does a model's
category-specific ability improve prediction on unseen items?

The fixed Q-matrix is the official MMLU-Pro category one-hot matrix.  All item
discriminations are fixed to one, so this is a bifactor Rasch/MIRT-lite model:

    logit p(y_mi = 1) = item_logit_i + alpha_m + theta_m,category(i)

``alpha`` is the general ability. ``theta`` is fitted with an L2 penalty, which
is the MAP estimate under a zero-mean Gaussian prior. Test-item item logits are
estimated from training-item metadata alone (category baseline) or from
question/options text plus metadata (strong baseline). No test response is used
for Q construction, hyperparameter selection, or threshold selection.

The primary evaluation is stricter than a cell-wise random split:

* 70/30 category-stratified item split, seed 20260811 (the frozen SAE split);
* 80/20 random population/evaluation model split, seed 20260812;
* item-difficulty targets and regularization selection use population models;
* evaluation-model alpha/theta use only their training-item responses;
* final metrics use evaluation-model responses on held-out items.

This first version intentionally uses categories rather than LLM-generated skill
annotations. It is a cheap gate: if category-specific ability cannot beat strong
additive controls, finer semantic Q annotation needs a stronger justification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from scipy.stats import wilcoxon
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


DEFAULT_SEED = 20260811
DEFAULT_MODEL_SEED = 20260812
CLIP = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("outputs/routereval_raw_matrices/mmlu_pro_response_matrix.npz"),
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=Path("external_cache/mmlu_pro_test.parquet"),
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=Path("sae_slices/mmlu_pro_categories.json"),
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("sae_slices/mmlu_pro_meta.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mmlu_pro_qmirt_minimal"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--eval-model-fraction", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--text-max-features", type=int, default=12000)
    parser.add_argument("--text-alpha", type=float, default=100.0)
    parser.add_argument("--category-prior-items", type=float, default=20.0)
    parser.add_argument(
        "--lambda-grid", type=str, default="1,3,10,30,100,300"
    )
    parser.add_argument(
        "--tuning-models",
        type=int,
        default=256,
        help="Deterministic population-model subset used only to tune Q shrinkage.",
    )
    parser.add_argument(
        "--threshold-cells-per-fold",
        type=int,
        default=200000,
        help="Deterministic OOF cell sample used to select decision thresholds.",
    )
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def clipped_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=np.float64), CLIP, 1.0 - CLIP))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _source_family(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("ori"):
        return "ori"
    if text.startswith("stemez"):
        return "stemez"
    if text.startswith("theoremqa"):
        return "theoremQA"
    return "unknown"


def _item_text(question: Any, options: Any) -> str:
    if isinstance(options, np.ndarray):
        choices = options.tolist()
    elif isinstance(options, (list, tuple)):
        choices = list(options)
    else:
        choices = []
    rendered = [str(question).strip()]
    rendered.extend(str(choice).strip() for choice in choices if str(choice).strip())
    return "\n".join(rendered)


def load_inputs(
    responses_path: Path,
    items_path: Path,
    categories_path: Path,
    meta_path: Path,
) -> Dict[str, Any]:
    for path in (responses_path, items_path, categories_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(responses_path, allow_pickle=False) as store:
        required = {
            "responses_q_by_m",
            "model_ids",
            "global_item_ids",
            "dataset_item_indices",
        }
        missing = required - set(store.files)
        if missing:
            raise ValueError("response NPZ missing keys: %s" % sorted(missing))
        responses = np.asarray(store["responses_q_by_m"], dtype=np.int8)
        model_ids = np.asarray(store["model_ids"]).astype(str)
        global_item_ids = np.asarray(store["global_item_ids"], dtype=np.int64)
        dataset_indices = np.asarray(store["dataset_item_indices"], dtype=np.int64)

    if responses.shape != (12032, 1823):
        raise ValueError("expected response shape (12032, 1823), got %r" % (responses.shape,))
    if not np.array_equal(np.unique(responses), np.array([0, 1], dtype=np.int8)):
        raise ValueError("minimal experiment requires a complete binary response matrix")
    if len(model_ids) != responses.shape[1]:
        raise ValueError("model ID count does not match response columns")
    if not np.array_equal(dataset_indices, np.arange(responses.shape[0])):
        raise ValueError("dataset item indices are not dense and ordered")
    if not np.array_equal(global_item_ids, np.arange(38233, 50265)):
        raise ValueError("unexpected RouterEval global MMLU-Pro interval")

    items = pd.read_parquet(items_path)
    needed = {"question", "options", "category", "src"}
    missing_columns = needed - set(items.columns)
    if missing_columns:
        raise ValueError("item parquet missing columns: %s" % sorted(missing_columns))
    if len(items) != responses.shape[0]:
        raise ValueError("item parquet row count does not match response rows")

    category_map = json.loads(categories_path.read_text(encoding="utf-8"))
    meta_map = json.loads(meta_path.read_text(encoding="utf-8"))
    categories = np.asarray(
        [str(category_map.get(str(i), "unknown")) for i in dataset_indices]
    )
    sources: List[str] = []
    subdomains: List[str] = []
    n_options: List[int] = []
    texts: List[str] = []
    for i, row in items.iterrows():
        options = row["options"]
        option_count = len(options) if isinstance(options, (list, tuple, np.ndarray)) else 0
        meta = meta_map.get(str(i), {})
        sources.append(_source_family(meta.get("src", row.get("src", "unknown"))))
        subdomains.append(str(row.get("src", "unknown")).strip() or "unknown")
        n_options.append(int(meta.get("n_options", option_count)))
        texts.append(_item_text(row["question"], options))
    if any(not text.strip() for text in texts):
        raise ValueError("empty question/options text found")

    category_names, category_codes = np.unique(categories, return_inverse=True)
    subdomain_names, subdomain_codes = np.unique(
        np.asarray(subdomains), return_inverse=True
    )
    return {
        "responses": np.ascontiguousarray(responses),
        "model_ids": model_ids,
        "global_item_ids": global_item_ids,
        "dataset_indices": dataset_indices,
        "texts": np.asarray(texts, dtype=object),
        "categories": categories,
        "category_names": category_names,
        "category_codes": category_codes.astype(np.int32),
        "sources": np.asarray(sources),
        "subdomains": np.asarray(subdomains),
        "subdomain_names": subdomain_names,
        "subdomain_codes": subdomain_codes.astype(np.int32),
        "n_options": np.asarray(n_options, dtype=np.int16),
        "question_ids": np.asarray(items.get("question_id", dataset_indices)),
    }


def make_item_split(categories: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(categories), dtype=np.int32)
    train, test = train_test_split(
        positions,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=categories,
    )
    train = np.sort(train.astype(np.int32))
    test = np.sort(test.astype(np.int32))
    if len(categories) == 12032 and (len(train), len(test)) != (8422, 3610):
        raise ValueError("frozen split did not produce 8422/3610 items")
    return train, test


def make_model_split(
    n_models: int, eval_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < eval_fraction < 0.5:
        raise ValueError("eval model fraction must be in (0, 0.5)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_models)
    n_eval = int(round(n_models * eval_fraction))
    evaluation = np.sort(order[:n_eval].astype(np.int32))
    population = np.sort(order[n_eval:].astype(np.int32))
    return population, evaluation


def category_item_predictions(
    item_easiness: np.ndarray,
    category_codes: np.ndarray,
    fit_items: np.ndarray,
    pred_items: np.ndarray,
    prior_items: float,
) -> Tuple[np.ndarray, np.ndarray]:
    overall = float(np.mean(item_easiness[fit_items]))
    n_categories = int(category_codes.max()) + 1
    means = np.full(n_categories, overall, dtype=np.float64)
    for category in range(n_categories):
        mask = fit_items[category_codes[fit_items] == category]
        if len(mask):
            means[category] = (
                float(item_easiness[mask].sum()) + prior_items * overall
            ) / (len(mask) + prior_items)
    return means[category_codes[fit_items]], means[category_codes[pred_items]]


def _meta_records(
    categories: np.ndarray,
    sources: np.ndarray,
    n_options: np.ndarray,
    indices: Iterable[int],
) -> List[Dict[str, float]]:
    return [
        {
            "category=" + str(categories[i]): 1.0,
            "source=" + str(sources[i]): 1.0,
            "n_options=" + str(int(n_options[i])): 1.0,
        }
        for i in indices
    ]


def text_item_predictions(
    texts: np.ndarray,
    categories: np.ndarray,
    sources: np.ndarray,
    n_options: np.ndarray,
    item_easiness: np.ndarray,
    fit_items: np.ndarray,
    pred_items: np.ndarray,
    max_features: int,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_text_fit = vectorizer.fit_transform(texts[fit_items].tolist())
    x_text_pred = vectorizer.transform(texts[pred_items].tolist())
    meta_vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
    x_meta_fit = meta_vectorizer.fit_transform(
        _meta_records(categories, sources, n_options, fit_items)
    )
    x_meta_pred = meta_vectorizer.transform(
        _meta_records(categories, sources, n_options, pred_items)
    )
    x_fit = sparse.hstack((x_text_fit, x_meta_fit), format="csr")
    x_pred = sparse.hstack((x_text_pred, x_meta_pred), format="csr")
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(x_fit, item_easiness[fit_items])
    pred_fit = np.clip(model.predict(x_fit), CLIP, 1.0 - CLIP)
    pred_other = np.clip(model.predict(x_pred), CLIP, 1.0 - CLIP)
    return pred_fit, pred_other


def fit_general_ability(
    item_logits: np.ndarray,
    responses: np.ndarray,
    initial: Optional[np.ndarray] = None,
    max_iter: int = 30,
    tol: float = 1e-8,
    model_chunk: int = 256,
) -> np.ndarray:
    item_logits = np.asarray(item_logits, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)
    n_models = responses.shape[1]
    if initial is None:
        alpha = clipped_logit(responses.mean(axis=0)) - float(item_logits.mean())
    else:
        alpha = np.asarray(initial, dtype=np.float64).copy()
    for _ in range(max_iter):
        largest = 0.0
        for start in range(0, n_models, model_chunk):
            stop = min(start + model_chunk, n_models)
            eta = item_logits[:, None] + alpha[None, start:stop]
            prob = expit(eta)
            gradient = (responses[:, start:stop] - prob).sum(axis=0)
            information = (prob * (1.0 - prob)).sum(axis=0) + 1e-9
            step = np.clip(gradient / information, -2.0, 2.0)
            alpha[start:stop] += step
            largest = max(largest, float(np.max(np.abs(step))))
        if largest < tol:
            break
    return alpha


def fit_bifactor_offsets(
    item_logits: np.ndarray,
    category_codes: np.ndarray,
    responses: np.ndarray,
    penalty: float,
    general_ability: Optional[np.ndarray] = None,
    max_iter: int = 30,
    tol: float = 1e-8,
    model_chunk: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit shrunk category offsets around a separately fitted general factor.

    Conditional on the general ability, every category offset is a convex
    one-dimensional penalized logistic fit. Centering theta afterwards and
    shifting alpha by the same amount preserves every fitted logit while giving
    the semantic factors a per-model sum-to-zero interpretation.
    """

    if penalty <= 0:
        raise ValueError("bifactor penalty must be positive")
    item_logits = np.asarray(item_logits, dtype=np.float64)
    category_codes = np.asarray(category_codes, dtype=np.int32)
    responses = np.asarray(responses, dtype=np.float64)
    if general_ability is None:
        alpha = fit_general_ability(item_logits, responses, model_chunk=model_chunk)
    else:
        alpha = np.asarray(general_ability, dtype=np.float64).copy()
    n_models = responses.shape[1]
    n_categories = int(category_codes.max()) + 1
    theta = np.zeros((n_models, n_categories), dtype=np.float64)

    for category in range(n_categories):
        item_mask = category_codes == category
        if not np.any(item_mask):
            continue
        base = item_logits[item_mask]
        truth = responses[item_mask]
        for start in range(0, n_models, model_chunk):
            stop = min(start + model_chunk, n_models)
            delta = theta[start:stop, category]
            for _ in range(max_iter):
                eta = base[:, None] + alpha[None, start:stop] + delta[None, :]
                prob = expit(eta)
                gradient = (truth[:, start:stop] - prob).sum(axis=0) - penalty * delta
                information = (prob * (1.0 - prob)).sum(axis=0) + penalty
                step = np.clip(gradient / information, -2.0, 2.0)
                delta += step
                if float(np.max(np.abs(step))) < tol:
                    break
            theta[start:stop, category] = delta

    center = theta.mean(axis=1)
    alpha = alpha + center
    theta = theta - center[:, None]
    return alpha, theta


def bifactor_logits(
    item_logits: np.ndarray,
    category_codes: np.ndarray,
    alpha: np.ndarray,
    theta: Optional[np.ndarray] = None,
) -> np.ndarray:
    scores = item_logits[:, None] + alpha[None, :]
    if theta is not None:
        scores = scores + theta[:, category_codes].T
    return np.asarray(scores, dtype=np.float64)


def _fit_general_with_effects(
    item_logits: np.ndarray,
    factor_codes: Sequence[np.ndarray],
    effects: Sequence[np.ndarray],
    responses: np.ndarray,
    initial: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-8,
    model_chunk: int = 256,
) -> Tuple[np.ndarray, float]:
    alpha = np.asarray(initial, dtype=np.float64).copy()
    n_models = responses.shape[1]
    last_largest = float("inf")
    for _ in range(max_iter):
        largest = 0.0
        for start in range(0, n_models, model_chunk):
            stop = min(start + model_chunk, n_models)
            eta = item_logits[:, None] + alpha[None, start:stop]
            for codes, effect in zip(factor_codes, effects):
                eta = eta + effect[start:stop][:, codes].T
            prob = expit(eta)
            gradient = (responses[:, start:stop] - prob).sum(axis=0)
            information = (prob * (1.0 - prob)).sum(axis=0) + 1e-9
            step = np.clip(gradient / information, -2.0, 2.0)
            alpha[start:stop] += step
            largest = max(largest, float(np.max(np.abs(step))))
        last_largest = largest
        if largest < tol:
            break
    return alpha, last_largest


def fit_multilevel_offsets(
    item_logits: np.ndarray,
    factor_codes: Sequence[np.ndarray],
    n_levels: Sequence[int],
    responses: np.ndarray,
    penalties: Sequence[float],
    general_ability: Optional[np.ndarray] = None,
    cycles: int = 4,
    max_iter: int = 20,
    tol: float = 1e-7,
    model_chunk: int = 256,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Coordinate-descent MAP fit for multiple overlapping fixed-Q factors."""

    if not (len(factor_codes) == len(n_levels) == len(penalties)):
        raise ValueError("factor codes, level counts, and penalties must align")
    if any(value <= 0 for value in penalties):
        raise ValueError("all multilevel penalties must be positive")
    item_logits = np.asarray(item_logits, dtype=np.float64)
    codes_list = [np.asarray(value, dtype=np.int32) for value in factor_codes]
    responses = np.asarray(responses, dtype=np.float64)
    if any(len(value) != len(item_logits) for value in codes_list):
        raise ValueError("every factor-code vector must have one value per item")
    if general_ability is None:
        alpha = fit_general_ability(item_logits, responses, model_chunk=model_chunk)
    else:
        alpha = np.asarray(general_ability, dtype=np.float64).copy()
    n_models = responses.shape[1]
    effects = [
        np.zeros((n_models, int(level_count)), dtype=np.float64)
        for level_count in n_levels
    ]

    for _ in range(cycles):
        largest = 0.0
        alpha, alpha_step = _fit_general_with_effects(
            item_logits,
            codes_list,
            effects,
            responses,
            alpha,
            max_iter=max_iter,
            tol=tol,
            model_chunk=model_chunk,
        )
        largest = max(largest, alpha_step)
        for factor_index, (codes, level_count, penalty) in enumerate(
            zip(codes_list, n_levels, penalties)
        ):
            for level in range(int(level_count)):
                item_mask = codes == level
                if not np.any(item_mask):
                    continue
                base_item = item_logits[item_mask]
                truth = responses[item_mask]
                for start in range(0, n_models, model_chunk):
                    stop = min(start + model_chunk, n_models)
                    delta = effects[factor_index][start:stop, level]
                    other = base_item[:, None] + alpha[None, start:stop]
                    for other_index, (other_codes, other_effect) in enumerate(
                        zip(codes_list, effects)
                    ):
                        if other_index == factor_index:
                            continue
                        other = other + other_effect[start:stop][
                            :, other_codes[item_mask]
                        ].T
                    for _ in range(max_iter):
                        probability = expit(other + delta[None, :])
                        gradient = (
                            (truth[:, start:stop] - probability).sum(axis=0)
                            - penalty * delta
                        )
                        information = (
                            (probability * (1.0 - probability)).sum(axis=0)
                            + penalty
                        )
                        step = np.clip(gradient / information, -2.0, 2.0)
                        delta += step
                        if float(np.max(np.abs(step))) < tol:
                            break
                    effects[factor_index][start:stop, level] = delta
                    largest = max(largest, float(np.max(np.abs(step))))
            center = effects[factor_index].mean(axis=1)
            effects[factor_index] -= center[:, None]
            alpha += center
        if largest < tol:
            break
    return alpha, effects


def multilevel_logits(
    item_logits: np.ndarray,
    factor_codes: Sequence[np.ndarray],
    alpha: np.ndarray,
    effects: Sequence[np.ndarray],
) -> np.ndarray:
    scores = np.asarray(item_logits, dtype=np.float64)[:, None] + alpha[None, :]
    for codes, effect in zip(factor_codes, effects):
        scores = scores + effect[:, np.asarray(codes, dtype=np.int32)].T
    return np.asarray(scores, dtype=np.float64)


def binary_nll(truth: np.ndarray, logits: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - truth * logits))


def mean_within_axis_auc(
    truth: np.ndarray, scores: np.ndarray, axis: int
) -> Tuple[float, float, int, np.ndarray]:
    n_groups = truth.shape[axis]
    per_group = np.full(n_groups, np.nan, dtype=np.float64)
    for index in range(n_groups):
        y = truth[:, index] if axis == 1 else truth[index, :]
        s = scores[:, index] if axis == 1 else scores[index, :]
        if np.min(y) == np.max(y):
            continue
        per_group[index] = float(roc_auc_score(y, s))
    valid = per_group[~np.isnan(per_group)]
    if not len(valid):
        return float("nan"), float("nan"), 0, per_group
    se = float(valid.std(ddof=1) / math.sqrt(len(valid))) if len(valid) > 1 else float("nan")
    return float(valid.mean()), se, len(valid), per_group


def sample_cells(
    truth: np.ndarray,
    score_map: Mapping[str, np.ndarray],
    max_cells: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n_cells = truth.size
    rng = np.random.default_rng(seed)
    if n_cells <= max_cells:
        chosen = np.arange(n_cells)
    else:
        chosen = np.sort(rng.choice(n_cells, size=max_cells, replace=False))
    sampled_truth = truth.ravel()[chosen].astype(np.int8)
    sampled_scores = {name: value.ravel()[chosen] for name, value in score_map.items()}
    return sampled_truth, sampled_scores


def choose_threshold(truth: np.ndarray, logits: np.ndarray) -> Tuple[float, float]:
    quantiles = np.linspace(0.01, 0.99, 99)
    candidates = np.unique(np.quantile(logits, quantiles))
    best_threshold = 0.0
    best_ba = -1.0
    positive = truth == 1
    negative = ~positive
    for threshold in candidates:
        prediction = logits >= threshold
        tpr = float(prediction[positive].mean())
        tnr = float((~prediction[negative]).mean())
        score = 0.5 * (tpr + tnr)
        if score > best_ba:
            best_threshold = float(threshold)
            best_ba = score
    return best_threshold, best_ba


def evaluate(
    name: str,
    truth: np.ndarray,
    logits: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, Any], np.ndarray]:
    probability = expit(logits)
    prediction = logits >= threshold
    within_model, within_model_se, n_models, model_aucs = mean_within_axis_auc(
        truth, logits, axis=1
    )
    within_item, within_item_se, n_items, _ = mean_within_axis_auc(
        truth, logits, axis=0
    )
    row = {
        "predictor": name,
        "threshold_logit": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(truth.ravel(), prediction.ravel())),
        "accuracy": float(np.mean(prediction == truth)),
        "pooled_auroc": float(roc_auc_score(truth.ravel(), logits.ravel())),
        "within_model_auroc": within_model,
        "within_model_auroc_se": within_model_se,
        "within_item_auroc": within_item,
        "within_item_auroc_se": within_item_se,
        "log_loss": binary_nll(truth, logits),
        "brier": float(np.mean((probability - truth) ** 2)),
        "n_test_cells": int(truth.size),
        "n_auc_models": int(n_models),
        "n_auc_items": int(n_items),
    }
    return row, model_aucs


def prepare_fold_bases(
    fit_items: np.ndarray,
    validation_items: np.ndarray,
    item_easiness: np.ndarray,
    data: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    q_fit, q_validation = category_item_predictions(
        item_easiness,
        data["category_codes"],
        fit_items,
        validation_items,
        args.category_prior_items,
    )
    text_fit, text_validation = text_item_predictions(
        data["texts"],
        data["categories"],
        data["sources"],
        data["n_options"],
        item_easiness,
        fit_items,
        validation_items,
        args.text_max_features,
        args.text_alpha,
    )
    return {
        "general": (np.zeros(len(fit_items)), np.zeros(len(validation_items))),
        "q": (clipped_logit(q_fit), clipped_logit(q_validation)),
        "text_meta": (clipped_logit(text_fit), clipped_logit(text_validation)),
        "oracle": (
            clipped_logit(item_easiness[fit_items]),
            clipped_logit(item_easiness[validation_items]),
        ),
    }


def markdown_report(
    metrics: pd.DataFrame,
    cv: pd.DataFrame,
    item_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> str:
    metric_lines = [
        "| predictor | bal. acc. | pooled AUC | within-model AUC | log loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        metric_lines.append(
            "| %s | %.4f | %.4f | %.4f | %.4f |"
            % (
                row.predictor,
                row.balanced_accuracy,
                row.pooled_auroc,
                row.within_model_auroc,
                row.log_loss,
            )
        )
    selected = protocol["selected_penalties"]
    metric_lookup = metrics.set_index("predictor")
    delta_q = (
        metric_lookup.loc["bifactor_q", "within_model_auroc"]
        - metric_lookup.loc["additive_q", "within_model_auroc"]
    )
    delta_text = (
        metric_lookup.loc["bifactor_text_meta_q", "within_model_auroc"]
        - metric_lookup.loc["additive_text_meta", "within_model_auroc"]
    )
    comparison_lookup = comparisons.set_index("comparison")
    q_comparison = comparison_lookup.loc["bifactor_q_minus_additive_q"]
    text_comparison = comparison_lookup.loc[
        "bifactor_text_meta_q_minus_additive_text_meta"
    ]
    decision = "GO" if delta_text >= 0.01 else "NO-GO for finer Q annotation yet"
    return "\n".join(
        [
            "# Minimal fixed-Q bifactor MMLU-Pro experiment",
            "",
            "## Result",
            "",
            *metric_lines,
            "",
            "Primary increment over the strong additive text+metadata baseline: "
            "within-model AUC Δ = %+.4f." % delta_text,
            "Across evaluation models its median Δ is %+.4f, %.1f%% are positive, "
            "and the paired Wilcoxon p-value is %.3g."
            % (
                text_comparison["median_delta"],
                100.0 * text_comparison["fraction_positive"],
                text_comparison["wilcoxon_p"],
            ),
            "Category-only Q increment: within-model AUC Δ = %+.4f." % delta_q,
            "Its paired median Δ is %+.4f and Wilcoxon p=%.3g; this is detectable "
            "but far below the practical gate."
            % (q_comparison["median_delta"], q_comparison["wilcoxon_p"]),
            "Preliminary gate: **%s**." % decision,
            "",
            "## Protocol",
            "",
            "- Frozen category-stratified item split: %d train / %d test."
            % (protocol["n_train_items"], protocol["n_test_items"]),
            "- Random response-population split: %d population / %d evaluation models."
            % (protocol["n_population_models"], protocol["n_evaluation_models"]),
            "- Fixed Q: %d official category dimensions; unit discrimination."
            % protocol["n_q_dimensions"],
            "- Selected shrinkage: q=%g, text+metadata=%g."
            % (selected["q"], selected["text_meta"]),
            "- Hyperparameters and thresholds were selected using training items and "
            "population models only.",
            "- Evaluation-model abilities were calibrated only on training-item responses.",
            "- The population-item-rate oracle opens population-model test responses only "
            "after all model selection is frozen.",
            "",
            "The model is a fixed-Q bifactor Rasch/MIRT-lite diagnostic, not a full free-"
            "discrimination 2PL MIRT fit. The random model holdout is not yet a lineage-family "
            "holdout; that is the next robustness test if this gate is positive.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.set_printoptions(precision=5, suppress=True)

    data = load_inputs(args.responses, args.items, args.categories, args.meta)
    responses = data["responses"]
    train_items, test_items = make_item_split(data["categories"], args.seed)
    population_models, evaluation_models = make_model_split(
        responses.shape[1], args.eval_model_fraction, args.model_seed
    )
    if args.tuning_models <= 0 or args.tuning_models > len(population_models):
        raise ValueError("--tuning-models must be within the population model count")
    rng = np.random.default_rng(args.model_seed + 1)
    tuning_models = np.sort(
        rng.choice(population_models, size=args.tuning_models, replace=False)
    )
    penalties = [float(value) for value in args.lambda_grid.split(",")]
    if not penalties or any(value <= 0 for value in penalties):
        raise ValueError("lambda grid must contain positive values")

    item_easiness = responses[:, population_models].mean(axis=1, dtype=np.float64)
    splitter = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed + 1
    )
    fold_specs = list(splitter.split(train_items, data["categories"][train_items]))
    fold_cache: List[Dict[str, Any]] = []
    cv_rows: List[Dict[str, Any]] = []

    print(
        "items %d/%d, models %d/%d, Q dimensions %d"
        % (
            len(train_items),
            len(test_items),
            len(population_models),
            len(evaluation_models),
            len(data["category_names"]),
        ),
        flush=True,
    )
    print("building strict item-fold representations ...", flush=True)
    for fold, (fit_pos, validation_pos) in enumerate(fold_specs):
        fit_items = train_items[fit_pos]
        validation_items = train_items[validation_pos]
        fold_bases = prepare_fold_bases(
            fit_items, validation_items, item_easiness, data, args
        )
        fold_cache.append(
            {
                "fold": fold,
                "fit_items": fit_items,
                "validation_items": validation_items,
                "bases": fold_bases,
            }
        )
        y_fit = responses[np.ix_(fit_items, tuning_models)]
        y_validation = responses[np.ix_(validation_items, tuning_models)]
        category_fit = data["category_codes"][fit_items]
        category_validation = data["category_codes"][validation_items]
        for mode in ("q", "text_meta"):
            base_fit, base_validation = fold_bases[mode]
            alpha = fit_general_ability(base_fit, y_fit)
            additive_validation = bifactor_logits(base_validation, category_validation, alpha)
            additive_nll = binary_nll(y_validation, additive_validation)
            for penalty in penalties:
                alpha_q, theta = fit_bifactor_offsets(
                    base_fit,
                    category_fit,
                    y_fit,
                    penalty,
                    general_ability=alpha,
                )
                score = bifactor_logits(
                    base_validation, category_validation, alpha_q, theta
                )
                cv_rows.append(
                    {
                        "fold": fold,
                        "mode": mode,
                        "penalty": penalty,
                        "additive_log_loss": additive_nll,
                        "bifactor_log_loss": binary_nll(y_validation, score),
                    }
                )
        print("  fold %d/%d ready" % (fold + 1, args.cv_folds), flush=True)

    cv = pd.DataFrame(cv_rows)
    cv["delta_log_loss"] = cv["bifactor_log_loss"] - cv["additive_log_loss"]
    selected_penalties: Dict[str, float] = {}
    for mode in ("q", "text_meta"):
        summary = (
            cv[cv["mode"] == mode]
            .groupby("penalty", as_index=False)["bifactor_log_loss"]
            .mean()
            .sort_values(["bifactor_log_loss", "penalty"])
        )
        selected_penalties[mode] = float(summary.iloc[0]["penalty"])
        print(
            "selected %s penalty=%g (CV log loss %.6f)"
            % (mode, selected_penalties[mode], summary.iloc[0]["bifactor_log_loss"]),
            flush=True,
        )

    print("building OOF threshold samples ...", flush=True)
    threshold_truth: List[np.ndarray] = []
    threshold_scores: Dict[str, List[np.ndarray]] = {
        name: []
        for name in (
            "general_only",
            "additive_q",
            "bifactor_q",
            "additive_text_meta",
            "bifactor_text_meta_q",
            "oracle_population_item_rate",
        )
    }
    for cached in fold_cache:
        fold = cached["fold"]
        fit_items = cached["fit_items"]
        validation_items = cached["validation_items"]
        bases = cached["bases"]
        y_fit = responses[np.ix_(fit_items, population_models)]
        y_validation = responses[np.ix_(validation_items, population_models)]
        category_fit = data["category_codes"][fit_items]
        category_validation = data["category_codes"][validation_items]
        scores: Dict[str, np.ndarray] = {}

        general_alpha = fit_general_ability(bases["general"][0], y_fit)
        scores["general_only"] = bifactor_logits(
            bases["general"][1], category_validation, general_alpha
        )
        for mode, additive_name, bifactor_name in (
            ("q", "additive_q", "bifactor_q"),
            ("text_meta", "additive_text_meta", "bifactor_text_meta_q"),
        ):
            base_fit, base_validation = bases[mode]
            alpha = fit_general_ability(base_fit, y_fit)
            scores[additive_name] = bifactor_logits(
                base_validation, category_validation, alpha
            )
            alpha_q, theta = fit_bifactor_offsets(
                base_fit,
                category_fit,
                y_fit,
                selected_penalties[mode],
                general_ability=alpha,
            )
            scores[bifactor_name] = bifactor_logits(
                base_validation, category_validation, alpha_q, theta
            )
        oracle_alpha = fit_general_ability(bases["oracle"][0], y_fit)
        scores["oracle_population_item_rate"] = bifactor_logits(
            bases["oracle"][1], category_validation, oracle_alpha
        )
        sampled_truth, sampled_scores = sample_cells(
            y_validation,
            scores,
            args.threshold_cells_per_fold,
            args.seed + 100 + fold,
        )
        threshold_truth.append(sampled_truth)
        for name, value in sampled_scores.items():
            threshold_scores[name].append(value)

    threshold_y = np.concatenate(threshold_truth)
    thresholds: Dict[str, float] = {}
    threshold_ba: Dict[str, float] = {}
    for name, parts in threshold_scores.items():
        thresholds[name], threshold_ba[name] = choose_threshold(
            threshold_y, np.concatenate(parts)
        )

    print("fitting final train-item models ...", flush=True)
    q_train_p, q_test_p = category_item_predictions(
        item_easiness,
        data["category_codes"],
        train_items,
        test_items,
        args.category_prior_items,
    )
    text_train_p, text_test_p = text_item_predictions(
        data["texts"],
        data["categories"],
        data["sources"],
        data["n_options"],
        item_easiness,
        train_items,
        test_items,
        args.text_max_features,
        args.text_alpha,
    )
    final_bases = {
        "general": (np.zeros(len(train_items)), np.zeros(len(test_items))),
        "q": (clipped_logit(q_train_p), clipped_logit(q_test_p)),
        "text_meta": (clipped_logit(text_train_p), clipped_logit(text_test_p)),
    }
    y_eval_train = responses[np.ix_(train_items, evaluation_models)]
    category_train = data["category_codes"][train_items]
    category_test = data["category_codes"][test_items]
    final_scores: Dict[str, np.ndarray] = {}

    alpha = fit_general_ability(final_bases["general"][0], y_eval_train)
    final_scores["general_only"] = bifactor_logits(
        final_bases["general"][1], category_test, alpha
    )
    theta_outputs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for mode, additive_name, bifactor_name in (
        ("q", "additive_q", "bifactor_q"),
        ("text_meta", "additive_text_meta", "bifactor_text_meta_q"),
    ):
        base_train, base_test = final_bases[mode]
        alpha = fit_general_ability(base_train, y_eval_train)
        final_scores[additive_name] = bifactor_logits(base_test, category_test, alpha)
        alpha_q, theta = fit_bifactor_offsets(
            base_train,
            category_train,
            y_eval_train,
            selected_penalties[mode],
            general_ability=alpha,
        )
        final_scores[bifactor_name] = bifactor_logits(
            base_test, category_test, alpha_q, theta
        )
        theta_outputs[mode] = (alpha_q, theta)

    # This is the first point at which held-out-item responses are used.
    y_eval_test = responses[np.ix_(test_items, evaluation_models)]
    population_test_easiness = responses[np.ix_(test_items, population_models)].mean(
        axis=1, dtype=np.float64
    )
    oracle_train_logits = clipped_logit(item_easiness[train_items])
    oracle_test_logits = clipped_logit(population_test_easiness)
    oracle_alpha = fit_general_ability(oracle_train_logits, y_eval_train)
    final_scores["oracle_population_item_rate"] = bifactor_logits(
        oracle_test_logits, category_test, oracle_alpha
    )

    metric_rows: List[Dict[str, Any]] = []
    model_auc_rows: List[Dict[str, Any]] = []
    prediction_order = list(threshold_scores)
    for name in prediction_order:
        row, model_aucs = evaluate(
            name, y_eval_test, final_scores[name], thresholds[name]
        )
        row["oof_threshold_balanced_accuracy"] = threshold_ba[name]
        metric_rows.append(row)
        for local_model, auc in enumerate(model_aucs):
            if np.isnan(auc):
                continue
            model_auc_rows.append(
                {
                    "predictor": name,
                    "dataset_model_index": int(evaluation_models[local_model]),
                    "model_id": str(data["model_ids"][evaluation_models[local_model]]),
                    "within_model_auroc": float(auc),
                }
            )
        print(
            "%26s bal=%.4f pooled=%.4f within=%.4f nll=%.4f"
            % (
                name,
                row["balanced_accuracy"],
                row["pooled_auroc"],
                row["within_model_auroc"],
                row["log_loss"],
            ),
            flush=True,
        )
    metrics = pd.DataFrame(metric_rows)
    model_auc_table = pd.DataFrame(model_auc_rows)
    model_auc_wide = model_auc_table.pivot(
        index="dataset_model_index", columns="predictor", values="within_model_auroc"
    )
    comparison_rows: List[Dict[str, Any]] = []
    for comparison, left, right in (
        ("bifactor_q_minus_additive_q", "bifactor_q", "additive_q"),
        (
            "bifactor_text_meta_q_minus_additive_text_meta",
            "bifactor_text_meta_q",
            "additive_text_meta",
        ),
    ):
        delta = (model_auc_wide[left] - model_auc_wide[right]).dropna().to_numpy()
        test = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        comparison_rows.append(
            {
                "comparison": comparison,
                "metric": "within_model_auroc",
                "n_models": len(delta),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "standard_error": float(delta.std(ddof=1) / math.sqrt(len(delta))),
                "fraction_positive": float(np.mean(delta > 0)),
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_p": float(test.pvalue),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)

    item_truth = y_eval_test.mean(axis=1, dtype=np.float64)
    item_rows = []
    for name, prediction in (
        ("category_q", q_test_p),
        ("text_meta", text_test_p),
        ("oracle_population_item_rate", population_test_easiness),
    ):
        residual = item_truth - prediction
        denominator = float(np.sum((item_truth - item_truth.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residual**2)) / denominator
        correlation = float(np.corrcoef(item_truth, prediction)[0, 1])
        item_rows.append(
            {
                "predictor": name,
                "pearson_r": correlation,
                "r2": r2,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "n_test_items": len(test_items),
            }
        )
    item_metrics = pd.DataFrame(item_rows)

    theta_rows: List[Dict[str, Any]] = []
    for mode, (alpha, theta) in theta_outputs.items():
        for local_model, model_index in enumerate(evaluation_models):
            row: Dict[str, Any] = {
                "mode": mode,
                "dataset_model_index": int(model_index),
                "model_id": str(data["model_ids"][model_index]),
                "general_ability": float(alpha[local_model]),
            }
            for category_index, category in enumerate(data["category_names"]):
                row["theta_" + re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")] = float(
                    theta[local_model, category_index]
                )
            theta_rows.append(row)

    model_split_rows = []
    population_set = set(int(value) for value in population_models)
    tuning_set = set(int(value) for value in tuning_models)
    for index, model_id in enumerate(data["model_ids"]):
        model_split_rows.append(
            {
                "dataset_model_index": index,
                "model_id": str(model_id),
                "split": "population" if index in population_set else "evaluation",
                "used_for_lambda_tuning": index in tuning_set,
                "train_item_accuracy": float(responses[train_items, index].mean()),
            }
        )

    protocol: Dict[str, Any] = {
        "experiment": "mmlu_pro_minimal_fixed_q_bifactor",
        "model": "unit-discrimination fixed-Q bifactor Rasch/MIRT-lite",
        "q_definition": "official MMLU-Pro category one-hot",
        "q_categories": data["category_names"].tolist(),
        "n_q_dimensions": len(data["category_names"]),
        "item_split": "category-stratified 70/30",
        "item_split_seed": args.seed,
        "n_train_items": len(train_items),
        "n_test_items": len(test_items),
        "model_split": "random population/evaluation; evaluation models calibrate on train items",
        "model_split_seed": args.model_seed,
        "n_population_models": len(population_models),
        "n_evaluation_models": len(evaluation_models),
        "n_tuning_models": len(tuning_models),
        "cv_folds": args.cv_folds,
        "lambda_grid": penalties,
        "selected_penalties": selected_penalties,
        "text_model": {
            "representation": "question and options TF-IDF + category/source/n_options one-hot",
            "max_features": args.text_max_features,
            "ridge_alpha": args.text_alpha,
            "answer_and_cot_excluded": True,
        },
        "category_prior_items": args.category_prior_items,
        "threshold_selection": {
            "source": "strict outer-fold OOF population-model cells",
            "cells_per_fold": args.threshold_cells_per_fold,
            "thresholds": thresholds,
            "oof_balanced_accuracy": threshold_ba,
        },
        "test_response_opening": "after penalties, thresholds, text models, and Q models were frozen",
        "elapsed_seconds": float(time.monotonic() - started),
        "inputs": {
            "responses": str(args.responses.resolve()),
            "responses_sha256": sha256(args.responses),
            "items": str(args.items.resolve()),
            "items_sha256": sha256(args.items),
            "categories": str(args.categories.resolve()),
            "categories_sha256": sha256(args.categories),
            "meta": str(args.meta.resolve()),
            "meta_sha256": sha256(args.meta),
        },
    }

    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    cv.to_csv(args.output_dir / "lambda_cv.csv", index=False)
    item_metrics.to_csv(args.output_dir / "item_metrics.csv", index=False)
    model_auc_table.to_csv(args.output_dir / "within_model_auroc.csv", index=False)
    comparisons.to_csv(args.output_dir / "comparisons.csv", index=False)
    pd.DataFrame(theta_rows).to_csv(args.output_dir / "theta_category.csv", index=False)
    pd.DataFrame(model_split_rows).to_csv(args.output_dir / "model_split.csv", index=False)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        truth=y_eval_test.astype(np.int8),
        test_item_indices=test_items,
        evaluation_model_indices=evaluation_models,
        **{
            "probability_" + name: expit(final_scores[name]).astype(np.float32)
            for name in prediction_order
        },
    )
    write_json(args.output_dir / "protocol.json", protocol)
    (args.output_dir / "report.md").write_text(
        markdown_report(metrics, cv, item_metrics, comparisons, protocol), encoding="utf-8"
    )
    print("written to %s" % args.output_dir, flush=True)


if __name__ == "__main__":
    main()
