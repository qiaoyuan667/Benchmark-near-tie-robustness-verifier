#!/usr/bin/env python3
"""Minimal owner-disjoint family-DIF experiment on MMLU-Pro.

The experiment estimates item-level relative strengths for high-confidence Qwen,
Llama, Gemma, Mistral/Mixtral, and Phi model groups after Rasch
ability/item-difficulty matching.
Models are capped per uploader and uploader groups are assigned wholly to either
discovery or validation. The item split is the frozen 70/30 MMLU-Pro split.

Training-item responses are used to select DIF shrinkage and representation ridge
penalties. Held-out-item validation responses are opened only after every choice
is frozen. Metadata, TF-IDF, and Gemma-2 layer-15 SAE representations then try to
predict discovery-half train-item DIF on validation-half held-out-item DIF.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

from ..discovery.qmirt_minimal import (
    clipped_logit,
    fit_general_ability,
    load_inputs,
    make_item_split,
    sha256,
    write_json,
)


FAMILY_PATTERNS = {
    "qwen": r"qwen",
    "llama": r"llama",
    "gemma": r"gemma",
    "mistral": r"mistral|mixtral",
    "deepseek": r"deepseek",
    "phi": r"(^|[^a-z])phi[-_ .0-9]",
}
DEFAULT_FAMILIES = ("qwen", "llama", "gemma", "mistral", "phi")


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
        "--sae-features",
        type=Path,
        default=Path("sae_slices/features/mmlu_pro_layer15.npz"),
    )
    parser.add_argument(
        "--sae-item-ids",
        type=Path,
        default=Path("sae_slices/features/mmlu_pro_item_ids.npy"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/mmlu_pro_family_dif_minimal")
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--family-seed", type=int, default=20260821)
    parser.add_argument("--families", type=str, default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--owner-cap", type=int, default=5)
    parser.add_argument("--min-family-models", type=int, default=20)
    parser.add_argument("--dif-penalty-grid", type=str, default="0.3,1,3,10,30")
    parser.add_argument("--dif-cycles", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--text-max-features", type=int, default=12000)
    parser.add_argument("--sae-svd-dim", type=int, default=400)
    parser.add_argument(
        "--sae-svd-cache",
        type=Path,
        default=Path("outputs/mmlu_pro_family_dif_sae_l15_svd400.npy"),
    )
    return parser.parse_args()


def classify_family(model_id: str) -> Tuple[Optional[str], str]:
    text = str(model_id).lower()
    matches = [
        family for family, pattern in FAMILY_PATTERNS.items() if re.search(pattern, text)
    ]
    if len(matches) == 1:
        return matches[0], "unique_regex_match"
    if len(matches) > 1:
        return None, "ambiguous:" + ",".join(matches)
    return None, "unmatched"


def owner_of(model_id: str) -> str:
    text = str(model_id)
    return text.split("__", 1)[0].strip().lower() if "__" in text else text.lower()


def select_and_split_family_models(
    model_ids: np.ndarray,
    train_accuracy: np.ndarray,
    requested_families: Sequence[str],
    owner_cap: int,
    min_family_models: int,
    seed: int,
) -> pd.DataFrame:
    if owner_cap <= 0:
        raise ValueError("owner cap must be positive")
    rng = np.random.default_rng(seed)
    candidates: List[Dict[str, Any]] = []
    requested = set(requested_families)
    for index, model_id in enumerate(model_ids):
        family, reason = classify_family(str(model_id))
        if family not in requested:
            continue
        candidates.append(
            {
                "dataset_model_index": index,
                "model_id": str(model_id),
                "family": family,
                "owner": owner_of(str(model_id)),
                "train_accuracy": float(train_accuracy[index]),
                "classification": reason,
            }
        )
    frame = pd.DataFrame(candidates)
    capped_parts: List[pd.DataFrame] = []
    for family in requested_families:
        family_frame = frame[frame["family"] == family].copy()
        family_capped_parts: List[pd.DataFrame] = []
        for _, owner_frame in family_frame.groupby("owner", sort=True):
            if len(owner_frame) > owner_cap:
                chosen = np.sort(
                    rng.choice(owner_frame.index.to_numpy(), size=owner_cap, replace=False)
                )
                owner_frame = owner_frame.loc[chosen]
            family_capped_parts.append(owner_frame)
        family_capped = pd.concat(family_capped_parts, ignore_index=True)
        if len(family_capped) < min_family_models:
            raise ValueError(
                "family %s has only %d capped models (<%d)"
                % (family, len(family_capped), min_family_models)
            )
        capped_parts.append(family_capped)

    selected = pd.concat(capped_parts, ignore_index=True)
    family_order = list(requested_families)
    family_position = {family: index for index, family in enumerate(family_order)}
    totals = np.asarray(
        [int((selected["family"] == family).sum()) for family in family_order],
        dtype=np.float64,
    )
    family_means = np.asarray(
        [
            float(selected.loc[selected["family"] == family, "train_accuracy"].mean())
            for family in family_order
        ],
        dtype=np.float64,
    )
    owner_groups = []
    for owner, owner_frame in selected.groupby("owner", sort=True):
        counts = np.zeros(len(family_order), dtype=np.float64)
        accuracy_sums = np.zeros(len(family_order), dtype=np.float64)
        for family, part in owner_frame.groupby("family"):
            position = family_position[family]
            counts[position] = len(part)
            accuracy_sums[position] = float(part["train_accuracy"].sum())
        owner_groups.append(
            {
                "owner": owner,
                "counts": counts,
                "accuracy_sums": accuracy_sums,
                "n": len(owner_frame),
                "tie": float(rng.uniform()),
            }
        )
    owner_groups.sort(key=lambda row: (-row["n"], row["tie"]))
    side_count = np.zeros((2, len(family_order)), dtype=np.float64)
    side_accuracy = np.zeros((2, len(family_order)), dtype=np.float64)
    assignment: Dict[str, int] = {}
    for group in owner_groups:
        costs = []
        for side in (0, 1):
            counts = side_count.copy()
            accuracies = side_accuracy.copy()
            counts[side] += group["counts"]
            accuracies[side] += group["accuracy_sums"]
            count_cost = np.sum(np.abs(counts[0] - counts[1]) / np.maximum(totals, 1.0))
            expected = counts * family_means[None, :]
            accuracy_cost = np.sum(
                np.abs((accuracies[0] - expected[0]) - (accuracies[1] - expected[1]))
                / np.maximum(totals, 1.0)
            )
            costs.append(float(count_cost + accuracy_cost))
        side = int(np.argmin(costs))
        if costs[0] == costs[1]:
            side = int(rng.integers(0, 2))
        assignment[group["owner"]] = side
        side_count[side] += group["counts"]
        side_accuracy[side] += group["accuracy_sums"]
    selected["model_half"] = selected["owner"].map(
        {owner: "discovery" if side == 0 else "validation" for owner, side in assignment.items()}
    )
    if selected.groupby("owner")["model_half"].nunique().max() != 1:
        raise AssertionError("an owner crossed model halves")
    for family in requested_families:
        halves = set(selected.loc[selected["family"] == family, "model_half"])
        if halves != {"discovery", "validation"}:
            raise ValueError("family %s was not represented in both halves" % family)
    return selected.sort_values(["family", "model_half", "owner", "model_id"]).reset_index(
        drop=True
    )


def fit_item_intercepts(
    responses: np.ndarray,
    abilities: np.ndarray,
    family_codes: Optional[np.ndarray] = None,
    family_effects: Optional[np.ndarray] = None,
    initial: Optional[np.ndarray] = None,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> np.ndarray:
    responses = np.asarray(responses, dtype=np.float64)
    abilities = np.asarray(abilities, dtype=np.float64)
    if initial is None:
        intercept = clipped_logit(responses.mean(axis=1)) - float(abilities.mean())
    else:
        intercept = np.asarray(initial, dtype=np.float64).copy()
    extra = 0.0
    if family_codes is not None and family_effects is not None:
        extra = family_effects[:, np.asarray(family_codes, dtype=np.int32)]
    for _ in range(max_iter):
        probability = expit(intercept[:, None] + abilities[None, :] + extra)
        gradient = (responses - probability).sum(axis=1)
        information = (probability * (1.0 - probability)).sum(axis=1) + 1e-9
        step = np.clip(gradient / information, -2.0, 2.0)
        intercept += step
        if float(np.max(np.abs(step))) < tol:
            break
    return intercept


def fit_rasch_abilities(
    responses: np.ndarray, cycles: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    responses = np.asarray(responses, dtype=np.float64)
    item = clipped_logit(responses.mean(axis=1))
    ability = clipped_logit(responses.mean(axis=0))
    for _ in range(cycles):
        ability = fit_general_ability(item, responses, initial=ability)
        item = fit_item_intercepts(responses, ability, initial=item)
        center = float(ability.mean())
        ability -= center
        item += center
    return ability, item


def fit_family_dif(
    responses: np.ndarray,
    abilities: np.ndarray,
    family_codes: np.ndarray,
    n_families: int,
    penalty: float,
    cycles: int = 5,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    if penalty <= 0:
        raise ValueError("DIF penalty must be positive")
    responses = np.asarray(responses, dtype=np.float64)
    family_codes = np.asarray(family_codes, dtype=np.int32)
    item = fit_item_intercepts(responses, abilities)
    effects = np.zeros((responses.shape[0], n_families), dtype=np.float64)
    for _ in range(cycles):
        item = fit_item_intercepts(
            responses,
            abilities,
            family_codes=family_codes,
            family_effects=effects,
            initial=item,
            max_iter=max_iter,
            tol=tol,
        )
        largest = 0.0
        for family in range(n_families):
            model_mask = family_codes == family
            truth = responses[:, model_mask]
            base = item[:, None] + abilities[model_mask][None, :]
            delta = effects[:, family]
            for _ in range(max_iter):
                probability = expit(base + delta[:, None])
                gradient = (truth - probability).sum(axis=1) - penalty * delta
                information = (probability * (1.0 - probability)).sum(axis=1) + penalty
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


def family_effect_logits(
    item: np.ndarray,
    abilities: np.ndarray,
    family_codes: np.ndarray,
    effects: np.ndarray,
) -> np.ndarray:
    return item[:, None] + abilities[None, :] + effects[:, family_codes]


def safe_correlations(truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    if np.std(truth) <= 1e-12 or np.std(prediction) <= 1e-12:
        return {"pearson_r": float("nan"), "spearman_r": float("nan")}
    return {
        "pearson_r": float(pearsonr(truth, prediction)[0]),
        "spearman_r": float(spearmanr(truth, prediction)[0]),
    }


def dif_nll(
    responses: np.ndarray,
    abilities: np.ndarray,
    family_codes: np.ndarray,
    item: np.ndarray,
    effects: np.ndarray,
) -> float:
    logits = family_effect_logits(item, abilities, family_codes, effects)
    return float(np.mean(np.logaddexp(0.0, logits) - responses * logits))


def meta_matrix(
    data: Mapping[str, Any], train_items: np.ndarray
) -> sparse.csr_matrix:
    records = [
        {
            "category=" + str(data["categories"][i]): 1.0,
            "subdomain=" + str(data["subdomains"][i]): 1.0,
            "n_options=" + str(int(data["n_options"][i])): 1.0,
        }
        for i in range(len(data["categories"]))
    ]
    vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
    vectorizer.fit([records[int(i)] for i in train_items])
    return vectorizer.transform(records).tocsr()


def build_representations(
    data: Mapping[str, Any],
    train_items: np.ndarray,
    sae_path: Path,
    sae_ids_path: Path,
    text_max_features: int,
    sae_svd_dim: int,
    seed: int,
    sae_svd_cache: Optional[Path] = None,
) -> Dict[str, Tuple[sparse.csr_matrix, Sequence[float]]]:
    meta = meta_matrix(data, train_items)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=text_max_features,
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf.fit(data["texts"][train_items].tolist())
    text = tfidf.transform(data["texts"].tolist()).tocsr()
    extracted_ids = np.load(sae_ids_path).astype(np.int32)
    if not np.array_equal(extracted_ids, np.arange(len(data["texts"]), dtype=np.int32)):
        raise ValueError("SAE rows do not align to MMLU-Pro dataset indices")
    sae = sparse.load_npz(sae_path).tocsr().astype(np.float32)
    if sae.shape[0] != len(data["texts"]):
        raise ValueError("SAE row count mismatch")
    sae.data = np.log1p(sae.data)
    if sae_svd_dim <= 0 or sae_svd_dim >= min(sae[train_items].shape):
        raise ValueError("invalid SAE SVD dimension")
    if sae_svd_cache is not None and sae_svd_cache.is_file():
        sae_svd = np.load(sae_svd_cache).astype(np.float32)
        if sae_svd.shape != (len(data["texts"]), sae_svd_dim):
            raise ValueError("cached SAE SVD shape mismatch")
    else:
        svd = TruncatedSVD(n_components=sae_svd_dim, n_iter=5, random_state=seed)
        svd.fit(sae[train_items])
        sae_svd = svd.transform(sae).astype(np.float32)
        if sae_svd_cache is not None:
            sae_svd_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(sae_svd_cache, sae_svd)
    sae_svd_sparse = sparse.csr_matrix(sae_svd)
    sae_label = "sae_l15_svd%d" % sae_svd_dim
    return {
        "metadata": (meta, (1.0, 10.0, 100.0)),
        "tfidf_metadata": (
            sparse.hstack((text, meta), format="csr"),
            (10.0, 100.0, 1000.0),
        ),
        sae_label: (sae_svd_sparse, (1.0, 10.0, 100.0, 1000.0)),
        sae_label + "_metadata": (
            sparse.hstack((sae_svd_sparse, meta), format="csr"),
            (1.0, 10.0, 100.0, 1000.0),
        ),
        "tfidf_" + sae_label + "_metadata": (
            sparse.hstack((text, sae_svd_sparse, meta), format="csr"),
            (10.0, 100.0, 1000.0),
        ),
    }


def representation_cv(
    name: str,
    matrix: sparse.csr_matrix,
    penalties: Sequence[float],
    train_items: np.ndarray,
    categories: np.ndarray,
    discovery_target: np.ndarray,
    validation_target: np.ndarray,
    folds: int,
    seed: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: List[Dict[str, Any]] = []
    for penalty in penalties:
        fold_scores = []
        for fold, (fit_pos, validation_pos) in enumerate(
            splitter.split(train_items, categories[train_items])
        ):
            fit_items = train_items[fit_pos]
            held_items = train_items[validation_pos]
            model = Ridge(alpha=penalty, solver="lsqr", tol=1e-4)
            model.fit(matrix[fit_items], discovery_target[fit_pos])
            prediction = model.predict(matrix[held_items])
            per_family = [
                safe_correlations(validation_target[validation_pos, family], prediction[:, family])[
                    "pearson_r"
                ]
                for family in range(discovery_target.shape[1])
            ]
            score = float(np.nanmean(per_family))
            fold_scores.append(score)
            rows.append(
                {
                    "representation": name,
                    "penalty": penalty,
                    "fold": fold,
                    "macro_validation_pearson": score,
                }
            )
        rows.append(
            {
                "representation": name,
                "penalty": penalty,
                "fold": "mean",
                "macro_validation_pearson": float(np.mean(fold_scores)),
            }
        )
    summary = [row for row in rows if row["fold"] == "mean"]
    summary.sort(key=lambda row: (-row["macro_validation_pearson"], row["penalty"]))
    return float(summary[0]["penalty"]), rows


def evaluate_effect_predictions(
    representation: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    family_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family_index, family in enumerate(family_names):
        y = truth[:, family_index]
        p = prediction[:, family_index]
        correlations = safe_correlations(y, p)
        denominator = float(np.sum((y - y.mean()) ** 2))
        rows.append(
            {
                "representation": representation,
                "family": family,
                **correlations,
                "r2": 1.0 - float(np.sum((y - p) ** 2)) / denominator,
                "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                "sign_accuracy": float(np.mean((p >= 0) == (y >= 0))),
                "n_items": len(y),
            }
        )
    macro = {
        "representation": representation,
        "family": "MACRO",
        "pearson_r": float(np.nanmean([row["pearson_r"] for row in rows])),
        "spearman_r": float(np.nanmean([row["spearman_r"] for row in rows])),
        "r2": float(np.nanmean([row["r2"] for row in rows])),
        "rmse": float(np.nanmean([row["rmse"] for row in rows])),
        "sign_accuracy": float(np.nanmean([row["sign_accuracy"] for row in rows])),
        "n_items": len(truth),
    }
    rows.append(macro)
    return rows


def make_report(
    family_models: pd.DataFrame,
    penalty_cv: pd.DataFrame,
    metrics: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> str:
    counts = (
        family_models.groupby(["family", "model_half"]).size().unstack(fill_value=0)
    )
    reliability = metrics[metrics["representation"] == "oracle_discovery_test"].set_index(
        "family"
    )
    macro = metrics[metrics["family"] == "MACRO"].sort_values("pearson_r", ascending=False)
    deployable = macro[macro["representation"] != "oracle_discovery_test"]
    best = deployable.iloc[0]
    reliable_families = int(
        (reliability.drop(index="MACRO", errors="ignore")["pearson_r"] >= 0.2).sum()
    )
    go = best["pearson_r"] >= 0.2 and reliable_families >= 3
    lines = [
        "# Minimal owner-disjoint family-DIF experiment",
        "",
        "## Family sample",
        "",
        "| family | discovery | validation |",
        "|---|---:|---:|",
    ]
    for family, row in counts.iterrows():
        lines.append(
            "| %s | %d | %d |"
            % (family, row.get("discovery", 0), row.get("validation", 0))
        )
    lines.extend(
        [
            "",
            "## Held-out-item results",
            "",
            "| representation | macro Pearson r | macro Spearman r | sign accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in macro.itertuples(index=False):
        lines.append(
            "| %s | %+.4f | %+.4f | %.3f |"
            % (row.representation, row.pearson_r, row.spearman_r, row.sign_accuracy)
        )
    lines.extend(
        [
            "",
            "Cross-half behavioral reliability reaches r>=0.2 for %d/%d families."
            % (reliable_families, len(protocol["families"])),
            "Best representation is `%s` with macro r=%+.4f."
            % (best["representation"], best["pearson_r"]),
            "Practical gate: **%s**."
            % ("GO to feature/slice interpretation" if go else "NO-GO"),
            "",
            "All owners are disjoint across model halves and capped at %d models."
            % protocol["owner_cap"],
            "DIF shrinkage and representation penalties use training items only; test-item "
            "responses are opened after all selections are frozen.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.sae_features, args.sae_item_ids):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = load_inputs(args.responses, args.items, args.categories, args.meta)
    responses = data["responses"]
    train_items, test_items = make_item_split(data["categories"], args.seed)
    train_accuracy = responses[train_items].mean(axis=0, dtype=np.float64)
    requested_families = tuple(value.strip() for value in args.families.split(",") if value.strip())
    family_models = select_and_split_family_models(
        data["model_ids"],
        train_accuracy,
        requested_families,
        args.owner_cap,
        args.min_family_models,
        args.family_seed,
    )
    family_names = sorted(family_models["family"].unique())
    family_to_code = {family: index for index, family in enumerate(family_names)}
    family_models["family_code"] = family_models["family"].map(family_to_code)
    print(
        family_models.groupby(["family", "model_half"]).size().unstack(fill_value=0),
        flush=True,
    )

    half_data: Dict[str, Dict[str, Any]] = {}
    for half in ("discovery", "validation"):
        selected = family_models[family_models["model_half"] == half]
        model_indices = selected["dataset_model_index"].to_numpy(dtype=np.int32)
        family_codes = selected["family_code"].to_numpy(dtype=np.int32)
        train_response = responses[np.ix_(train_items, model_indices)]
        ability, _ = fit_rasch_abilities(train_response)
        half_data[half] = {
            "models": model_indices,
            "families": family_codes,
            "abilities": ability,
            "train_response": train_response,
        }

    penalties = [float(value) for value in args.dif_penalty_grid.split(",")]
    penalty_rows: List[Dict[str, Any]] = []
    for penalty in penalties:
        train_effects: Dict[str, np.ndarray] = {}
        train_items_fit: Dict[str, np.ndarray] = {}
        for half in ("discovery", "validation"):
            item, effects = fit_family_dif(
                half_data[half]["train_response"],
                half_data[half]["abilities"],
                half_data[half]["families"],
                len(family_names),
                penalty,
                cycles=args.dif_cycles,
            )
            train_effects[half] = effects
            train_items_fit[half] = item
        validation_nll = dif_nll(
            half_data["validation"]["train_response"],
            half_data["validation"]["abilities"],
            half_data["validation"]["families"],
            train_items_fit["validation"],
            train_effects["discovery"],
        )
        correlations = [
            safe_correlations(train_effects["validation"][:, f], train_effects["discovery"][:, f])[
                "pearson_r"
            ]
            for f in range(len(family_names))
        ]
        penalty_rows.append(
            {
                "penalty": penalty,
                "validation_log_loss": validation_nll,
                "macro_cross_half_pearson": float(np.mean(correlations)),
                **{
                    "pearson_" + family_names[f]: correlations[f]
                    for f in range(len(family_names))
                },
            }
        )
    penalty_cv = pd.DataFrame(penalty_rows).sort_values(
        ["validation_log_loss", "penalty"]
    )
    selected_dif_penalty = float(penalty_cv.iloc[0]["penalty"])
    print("selected DIF penalty", selected_dif_penalty, flush=True)

    train_effects = {}
    for half in ("discovery", "validation"):
        _, effects = fit_family_dif(
            half_data[half]["train_response"],
            half_data[half]["abilities"],
            half_data[half]["families"],
            len(family_names),
            selected_dif_penalty,
            cycles=args.dif_cycles,
        )
        train_effects[half] = effects

    representations = build_representations(
        data,
        train_items,
        args.sae_features,
        args.sae_item_ids,
        args.text_max_features,
        args.sae_svd_dim,
        args.seed,
        args.sae_svd_cache,
    )
    representation_cv_rows: List[Dict[str, Any]] = []
    selected_representation_penalties: Dict[str, float] = {}
    for name, (matrix, ridge_grid) in representations.items():
        selected, rows = representation_cv(
            name,
            matrix,
            ridge_grid,
            train_items,
            data["categories"],
            train_effects["discovery"],
            train_effects["validation"],
            args.cv_folds,
            args.seed + 2,
        )
        selected_representation_penalties[name] = selected
        representation_cv_rows.extend(rows)
        print("selected", name, "alpha", selected, flush=True)

    final_predictions: Dict[str, np.ndarray] = {}
    for name, (matrix, _) in representations.items():
        model = Ridge(
            alpha=selected_representation_penalties[name], solver="lsqr", tol=1e-4
        )
        model.fit(matrix[train_items], train_effects["discovery"])
        final_predictions[name] = model.predict(matrix[test_items])

    # Final held-out-item response opening starts here.
    test_effects: Dict[str, np.ndarray] = {}
    for half in ("discovery", "validation"):
        test_response = responses[np.ix_(test_items, half_data[half]["models"])]
        _, effects = fit_family_dif(
            test_response,
            half_data[half]["abilities"],
            half_data[half]["families"],
            len(family_names),
            selected_dif_penalty,
            cycles=args.dif_cycles,
        )
        test_effects[half] = effects

    metric_rows: List[Dict[str, Any]] = []
    for name, prediction in final_predictions.items():
        metric_rows.extend(
            evaluate_effect_predictions(
                name, test_effects["validation"], prediction, family_names
            )
        )
    metric_rows.extend(
        evaluate_effect_predictions(
            "oracle_discovery_test",
            test_effects["validation"],
            test_effects["discovery"],
            family_names,
        )
    )
    metrics = pd.DataFrame(metric_rows)
    print(
        metrics[metrics["family"] == "MACRO"][["representation", "pearson_r", "spearman_r"]]
        .sort_values("pearson_r", ascending=False)
        .to_string(index=False),
        flush=True,
    )

    best_name = str(
        metrics[(metrics["family"] == "MACRO") & (metrics["representation"] != "oracle_discovery_test")]
        .sort_values("pearson_r", ascending=False)
        .iloc[0]["representation"]
    )
    best_prediction = final_predictions[best_name]
    items_frame = pd.read_parquet(args.items, columns=["question_id", "question", "category", "src"])
    top_rows: List[Dict[str, Any]] = []
    for family_index, family in enumerate(family_names):
        order = np.argsort(test_effects["validation"][:, family_index])
        for direction, local_indices in (
            ("disadvantage", order[:10]),
            ("advantage", order[-10:][::-1]),
        ):
            for rank, local_index in enumerate(local_indices, start=1):
                item_index = int(test_items[local_index])
                top_rows.append(
                    {
                        "family": family,
                        "direction": direction,
                        "rank": rank,
                        "dataset_item_index": item_index,
                        "question_id": items_frame.iloc[item_index]["question_id"],
                        "category": items_frame.iloc[item_index]["category"],
                        "subdomain": items_frame.iloc[item_index]["src"],
                        "validation_dif": float(test_effects["validation"][local_index, family_index]),
                        "discovery_dif": float(test_effects["discovery"][local_index, family_index]),
                        "predicted_dif": float(best_prediction[local_index, family_index]),
                        "question": str(items_frame.iloc[item_index]["question"]),
                    }
                )

    protocol: Dict[str, Any] = {
        "experiment": "mmlu_pro_owner_disjoint_family_dif_minimal",
        "families": family_names,
        "family_classifier": FAMILY_PATTERNS,
        "ambiguous_matches_excluded": True,
        "owner_cap": args.owner_cap,
        "owner_disjoint_halves": True,
        "family_seed": args.family_seed,
        "item_split_seed": args.seed,
        "n_train_items": len(train_items),
        "n_test_items": len(test_items),
        "selected_dif_penalty": selected_dif_penalty,
        "selected_representation_penalties": selected_representation_penalties,
        "sae_representation": {
            "layer": 15,
            "svd_dimensions": args.sae_svd_dim,
            "svd_fit_items": "training items only",
            "svd_cache": str(args.sae_svd_cache.resolve()),
        },
        "primary_metric": "macro family Pearson correlation on validation-half held-out-item DIF",
        "go_gate": "best representation macro r>=0.2 and cross-half reliability r>=0.2 in >=3 families",
        "test_response_opening": "after DIF and representation penalties were frozen",
        "elapsed_seconds": float(time.monotonic() - started),
        "inputs": {
            "responses": str(args.responses.resolve()),
            "responses_sha256": sha256(args.responses),
            "items": str(args.items.resolve()),
            "items_sha256": sha256(args.items),
            "sae_features": str(args.sae_features.resolve()),
            "sae_features_sha256": sha256(args.sae_features),
        },
    }

    family_models.to_csv(args.output_dir / "family_models.csv", index=False)
    penalty_cv.to_csv(args.output_dir / "dif_penalty_cv.csv", index=False)
    pd.DataFrame(representation_cv_rows).to_csv(
        args.output_dir / "representation_cv.csv", index=False
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(top_rows).to_csv(args.output_dir / "top_family_items.csv", index=False)
    np.savez_compressed(
        args.output_dir / "dif_and_predictions.npz",
        train_item_indices=train_items,
        test_item_indices=test_items,
        family_names=np.asarray(family_names),
        discovery_train_dif=train_effects["discovery"],
        validation_train_dif=train_effects["validation"],
        discovery_test_dif=test_effects["discovery"],
        validation_test_dif=test_effects["validation"],
        **{
            "prediction_" + name: prediction.astype(np.float32)
            for name, prediction in final_predictions.items()
        },
    )
    write_json(args.output_dir / "protocol.json", protocol)
    (args.output_dir / "report.md").write_text(
        make_report(family_models, penalty_cv, metrics, protocol), encoding="utf-8"
    )
    print("written to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
