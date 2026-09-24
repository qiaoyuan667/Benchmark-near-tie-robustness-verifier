#!/usr/bin/env python3
"""Isolated, audit-only full spectral-profile matched ranking diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .._paths import PROJECT_ROOT
from . import require_primary_files
from . import discrimination as previous
from . import within_family as within
BENCHMARKS = previous.BENCHMARKS
FOLDS = previous.FOLDS
SCENARIOS = {"random_pairing": None, "optimal_unrestricted": None,
             "caliper_1.00": 1.0, "caliper_0.50": 0.5, "caliper_0.25": 0.25}
REPLICATES = 1000
SEED = 20260924


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def solve_assignment(profile_cost, b_cost, caliper=None):
    """Maximum-cardinality then minimum-cost; optional hard component calipers."""
    n, m = profile_cost.shape
    if not n or not m:
        return np.array([], dtype=int), np.array([], dtype=int)
    total = np.maximum(profile_cost, 0.0) + np.maximum(b_cost, 0.0)
    if caliper is None:
        return linear_sum_assignment(total)
    allowed = (profile_cost <= caliper ** 2 + 1e-12) & (b_cost <= caliper ** 2 + 1e-12)
    if not allowed.any():
        return np.array([], dtype=int), np.array([], dtype=int)
    penalty = (n + 1) * (1.0 + float(total[allowed].max()))
    augmented = np.full((n, m + n), penalty, dtype=np.float64)
    augmented[:, :m] = np.where(allowed, total, 2.0 * penalty)
    rows, columns = linear_sum_assignment(augmented)
    keep = columns < m
    rows, columns = rows[keep], columns[keep]
    if not np.all(allowed[rows, columns]):
        raise AssertionError("forbidden assignment selected")
    return rows, columns


def construct_pairs(cells, selected, embedding, intercept, seed):
    """This function receives no target response, ranking, or family inputs."""
    buffers = {key: [] for key in SCENARIOS}
    rng = np.random.default_rng(seed + 10000)
    for cell in np.unique(cells):
        left = np.flatnonzero((cells == cell) & selected)
        right = np.flatnonzero((cells == cell) & ~selected)
        if not len(left) or not len(right):
            continue
        profile = cdist(embedding[left], embedding[right], metric="sqeuclidean")
        bcost = (intercept[left, None] - intercept[right][None, :]) ** 2
        for scenario, caliper in SCENARIOS.items():
            if scenario == "random_pairing":
                n = min(len(left), len(right))
                a, b = rng.permutation(len(left))[:n], rng.permutation(len(right))[:n]
            else:
                a, b = solve_assignment(profile, bcost, caliper)
            if len(a):
                buffers[scenario].append(np.column_stack([left[a], right[b]]))
    return {key: np.concatenate(parts) if parts else np.empty((0, 2), dtype=int)
            for key, parts in buffers.items()}


def make_weights(weights, pairs, switches):
    result = np.broadcast_to(weights, (len(switches), len(weights))).copy()
    if len(pairs):
        a, b = pairs.T
        result[:, a] = (1.0 - switches) * weights[a]
        result[:, b] = switches * weights[a]
    return result


def assert_weights(w, weights, cells):
    if not np.all((w >= 0) & np.isfinite(w)):
        raise AssertionError("invalid weights")
    for cell in np.unique(cells):
        mask = cells == cell
        if not np.all((w[:, mask] > 0).sum(axis=1) == (weights[mask] > 0).sum()):
            raise AssertionError("control length changed in a cell")
        if not np.allclose(w[:, mask].sum(axis=1), mask.sum(), atol=1e-9, rtol=0):
            raise AssertionError("control cell mass changed")
    if not np.allclose(w.sum(axis=1), len(weights), atol=1e-8, rtol=0):
        raise AssertionError("control total mass changed")


def pair_diagnostics(pairs, frozen, fit, embedding, audit_responses):
    columns = ["anchor_item", "control_item", "blueprint_cell", "profile_rms_logit",
               "raw_loading_rms", "profile_cosine", "abs_b_difference", "b_difference",
               "clipped_profile_rms_logit", "predicted_probability_rms",
               "anchor_residual_dif", "control_residual_dif", "dif_difference", "anchor_weight"]
    if not len(pairs):
        return pd.DataFrame(columns=columns)
    a, b = pairs.T
    ea, eb = embedding[a], embedding[b]
    norm = np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1)
    cosine = np.divide((ea * eb).sum(axis=1), norm,
                       out=np.full(len(a), np.nan), where=norm > 1e-15)
    intercept = frozen.item_intercept.to_numpy()
    dif = frozen.residual_dif_logit_range.to_numpy()
    clipped = fit.offsets[a] - fit.offsets[b]
    probability_diff = expit(intercept[a, None] + fit.offsets[a]) - expit(intercept[b, None] + fit.offsets[b])
    return pd.DataFrame({
        "anchor_item": a, "control_item": b, "blueprint_cell": fit.cells[a],
        "profile_rms_logit": np.linalg.norm(ea - eb, axis=1),
        "raw_loading_rms": np.linalg.norm(fit.item_loadings[a] - fit.item_loadings[b], axis=1) / np.sqrt(audit_responses.shape[1]),
        "profile_cosine": cosine, "abs_b_difference": np.abs(intercept[a] - intercept[b]),
        "b_difference": intercept[b] - intercept[a],
        "clipped_profile_rms_logit": np.sqrt(np.mean(clipped ** 2, axis=1)),
        "predicted_probability_rms": np.sqrt(np.mean(probability_diff ** 2, axis=1)),
        "anchor_residual_dif": dif[a], "control_residual_dif": dif[b],
        "dif_difference": dif[b] - dif[a], "anchor_weight": fit.weights[a],
    })


def matching_summary(table, weights, scenario, fold):
    n = len(table)
    row = {"audit_half": fold, "scenario": scenario, "n_pairs": n,
           "n_anchors": int((weights > 0).sum()),
           "matched_anchor_fraction": n / float((weights > 0).sum()),
           "swappable_weight_fraction": float(table.anchor_weight.sum() / weights.sum()) if n else 0.0}
    for column in ["profile_rms_logit", "raw_loading_rms", "profile_cosine", "abs_b_difference",
                   "clipped_profile_rms_logit", "predicted_probability_rms", "dif_difference"]:
        row[column + "_median"] = float(table[column].median()) if n else np.nan
        row[column + "_p90"] = float(table[column].quantile(.9)) if n else np.nan
    row["higher_final_dif_fraction"] = float((table.dif_difference > 0).mean()) if n else np.nan
    return row


def score_weights(w, y, full, families, owners, stats):
    """Batch scores, with exact original matvec for potentially tied near pairs."""
    totals = w @ y
    scores = totals / w.sum(axis=1, keepdims=True)
    left, right = np.triu_indices(len(full), 1)
    near = (owners[left] != owners[right]) & (np.abs(full[left] - full[right]) <= .01)
    left, right = left[near], right[near]
    possibly_tied = np.any(np.abs(scores[:, left] - scores[:, right]) < 1e-12, axis=1)
    for index in np.flatnonzero(possibly_tied):
        totals[index] = w[index] @ y
        scores[index] = totals[index] / w[index].sum()
    rows = []
    for index, score in enumerate(scores):
        row = within.pairwise_metrics(full, score, families, owners)
        wi = w[index]
        row.update({
            "mean_audit_residual_dif": float(wi @ stats["dif"] / wi.sum()),
            "mean_audit_b": float(wi @ stats["b"] / wi.sum()),
            "mean_audit_discrimination": float(wi @ stats["g"] / wi.sum()),
            "target_citc": float(wi @ stats["citc"] / wi.sum()),
            "target_independent_item_sem": previous.independent_item_score_sem(stats["variance"], wi),
            "target_weighted_alpha": previous.weighted_cronbach_alpha(totals[index], stats["variance"], wi),
            "effective_item_count": previous.effective_item_count(wi),
            "anchor_overlap_fraction": float(((wi > 0) & stats["selected"]).sum() / stats["selected"].sum()),
            "profile_mean_rms_difference": float(np.linalg.norm(wi @ stats["embedding"] / wi.sum() - stats["low_mean"])),
            "max_abs_profile_axis_smd": float(np.max(np.abs((wi @ stats["embedding"] / wi.sum() - stats["low_mean"]) / stats["embedding_sd"]))),
        })
        rows.append(row)
    return rows


def freeze_protocol(out, keys):
    package_root = Path(__file__).resolve().parents[1]
    files = [package_root / name for name in [
        "__init__.py", "_paths.py", "diagnostics/__init__.py",
        "diagnostics/discrimination.py", "diagnostics/within_family.py",
        "diagnostics/capability_profile.py", "core/family_dif.py",
        "core/spectral_mirt.py", "discovery/qmirt_minimal.py",
        "ranking/mirt_primary.py", "ranking/mmlu_pro.py", "ranking/routereval.py",
    ]]
    write_json(out / "protocol.json", {
        "experiment": "full_latent_profile_matched_diagnostic",
        "status": "post-hoc diagnostic; all matching scenarios fixed before scoring",
        "benchmarks": keys, "replicates": REPLICATES, "seed_base": SEED,
        "scenarios": SCENARIOS,
        "distance": "||L_i/(s_i sqrt(M))-L_j/(s_j sqrt(M))||^2 + (b_i-b_j)^2",
        "calipers": "both profile RMS and absolute b difference, in logits",
        "dimension": "frozen full K, no additional dimension reduction",
        "matching": "maximum cardinality, then minimum total cost; no replacement; same frozen blueprint cell",
        "unmatched_policy": "keep original low-DIF anchor fixed; report restricted coverage",
        "controls": "one uniformly random endpoint per pair, 1000 configurations; unchanged per-cell counts/weights",
        "interpretation": "conditional diagnostic reference distribution, not a causal identification claim",
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "code_sha256": {str(p.relative_to(package_root)): sha256(p) for p in files},
    })


def load_frozen(key, fold_index, data, primary_root):
    run = primary_root / (key + "_family_ranking_impact")
    fit, audit, target, audit_y, y = previous.reconstruct_fold(key, fold_index, run, data)
    audit_half, target_half = FOLDS[fold_index]
    frozen = pd.read_csv(run / "crossfit_anchor_items.csv")
    frozen = frozen[np.isclose(frozen.anchor_fraction, .5) & frozen.audit_half.eq(audit_half)].sort_values("dataset_item_index")
    if set(audit.owner) & set(target.owner):
        raise AssertionError("audit/target owner leakage")
    if not np.array_equal(np.asarray(data["sources"]).astype(str), frozen.source.to_numpy(dtype=str)):
        raise AssertionError("item-group identity mismatch")
    if not np.allclose(fit.model_axes.T @ fit.model_axes, np.eye(fit.model_axes.shape[1]), atol=1e-10):
        raise AssertionError("model axes not orthonormal")
    return run, fit, audit, target, audit_y, np.asarray(y, dtype=float), frozen


def run_benchmark(key, out, primary_root):
    started = time.monotonic()
    bench_dir = out / key
    bench_dir.mkdir()
    data = previous.mirt_primary.load_data(key)
    write_json(bench_dir / "input_sha256.json", {p.name: sha256(p) for p in data["input_paths"]})
    control_rows, observed_rows, swap_rows, diagnostics_rows, tests = [], [], [], [], []
    for fold_index, (audit_half, target_half) in enumerate(FOLDS):
        print(f"{key} {audit_half}: reconstructing frozen K/axes", flush=True)
        run, fit, audit, target, audit_y, y, frozen = load_frozen(key, fold_index, data, primary_root)
        _, scale = previous.smoothed_item_parameters(audit_y)
        embedding = fit.item_loadings / scale[:, None] / np.sqrt(audit_y.shape[1])
        intercept = frozen.item_intercept.to_numpy()
        seed = SEED + 100 * list(BENCHMARKS).index(key) + fold_index
        print(f"{key} {audit_half}: matching in all {embedding.shape[1]} dimensions", flush=True)
        pairings = construct_pairs(fit.cells, fit.selected, embedding, intercept, seed)
        # Persist audit-only representations and explicit pairs before target scoring.
        np.savez_compressed(bench_dir / f"{audit_half}_representation.npz",
                            item_loadings=fit.item_loadings, effective_embedding=embedding,
                            item_intercept=intercept, scale=scale, model_axes=fit.model_axes,
                            item_group=frozen.source.to_numpy(dtype=str), blueprint_cell=fit.cells,
                            selected=fit.selected, weights=fit.weights,
                            audit_model_index=audit.dataset_model_index.to_numpy(),
                            target_model_index=target.dataset_model_index.to_numpy())
        for scenario, pairs in pairings.items():
            if len(pairs):
                a, b = pairs.T
                assert len(np.unique(pairs)) == 2 * len(pairs)
                assert fit.selected[a].all() and (~fit.selected[b]).all()
                assert np.array_equal(fit.cells[a], fit.cells[b])
                assert np.array_equal(frozen.source.to_numpy()[a], frozen.source.to_numpy()[b])
            pair_table = pair_diagnostics(pairs, frozen, fit, embedding, audit_y)
            pair_table.to_csv(bench_dir / f"{audit_half}_{scenario}_pairs.csv", index=False)
            diagnostic = matching_summary(pair_table, fit.weights, scenario, audit_half)
            diagnostic.update(benchmark=BENCHMARKS[key], selected_dimension=embedding.shape[1])
            diagnostics_rows.append(diagnostic)
            print(f"{key} {audit_half} {scenario}: {len(pairs)} pairs, "
                  f"swappable mass {diagnostic['swappable_weight_fraction']:.1%}, "
                  f"median profile RMS {diagnostic['profile_rms_logit_median']:.3f}", flush=True)
        pd.DataFrame(diagnostics_rows).to_csv(bench_dir / "matching_quality.csv", index=False)
        families, owners = target.family.to_numpy(dtype=str), target.owner.to_numpy(dtype=str)
        full = y.mean(axis=0)
        stats = {"dif": frozen.residual_dif_logit_range.to_numpy(), "b": intercept,
                 "g": fit.offsets.std(axis=1), "citc": previous.corrected_item_total_correlation(y),
                 "variance": y.var(axis=1, ddof=1), "selected": fit.selected,
                 "embedding": embedding, "low_mean": fit.weights @ embedding / fit.weights.sum(),
                 "embedding_sd": np.maximum(embedding.std(axis=0, ddof=1), 1e-12)}
        context = {"benchmark": BENCHMARKS[key], "benchmark_key": key,
                   "audit_half": audit_half, "target_half": target_half}
        observed = score_weights(fit.weights[None, :], y, full, families, owners, stats)[0]
        saved = pd.read_csv(run / "ranking_impact_metrics.csv")
        saved = saved[np.isclose(saved.anchor_fraction, .5) & saved.audit_half.eq(audit_half)].iloc[0]
        for column in ["close_cross_family_flips", "close_cross_family_eligible_pairs"]:
            if observed[column] != int(saved[column]):
                raise AssertionError(f"frozen low-DIF result not reproduced: {column}")
        # Reproduce original null samples, not just its final summary.
        original = pd.read_csv(run / "matched_random_anchor_controls.csv")
        original = original[original.audit_half.eq(audit_half)].sort_values("replicate")
        rng = np.random.default_rng(previous.ORIGINAL_SEED + fold_index)
        for index in range(10):
            w = previous.matched_random_anchor_weights(fit.cells, fit.selected, rng)
            got = within.pairwise_metrics(full, w @ y / w.sum(), families, owners)
            for column in ["close_cross_family_flips", "close_cross_family_eligible_pairs"]:
                if got[column] != int(original.iloc[index][column]):
                    raise AssertionError("original random control reproduction failed")
        observed_rows.append({**context, **observed})
        tests.append({**context, "axes_agree_to_1e_12": True, "owners_disjoint": True,
                      "low_dif_counts_reproduced": True, "original_controls_reproduced": 10})
        for scenario, pairs in pairings.items():
            rng = np.random.default_rng(seed)
            switches = rng.integers(0, 2, size=(REPLICATES, len(pairs)), dtype=np.int8)
            np.savez_compressed(bench_dir / f"{audit_half}_{scenario}_switches.npz",
                                pairs=pairs, switches=switches)
            swapped_w = make_weights(fit.weights, pairs, np.ones((1, len(pairs))))
            assert_weights(swapped_w, fit.weights, fit.cells)
            swapped = score_weights(swapped_w, y, full, families, owners, stats)[0]
            swap_rows.append({**context, "scenario": scenario, **swapped})
            for start in range(0, REPLICATES, 100):
                s = switches[start:start + 100]
                weights = make_weights(fit.weights, pairs, s)
                opposite = make_weights(fit.weights, pairs, 1 - s)
                assert_weights(weights, fit.weights, fit.cells)
                assert_weights(opposite, fit.weights, fit.cells)
                rows = score_weights(weights, y, full, families, owners, stats)
                complements = score_weights(opposite, y, full, families, owners, stats)
                for offset, (row, comp) in enumerate(zip(rows, complements)):
                    extra = {"complement_" + k: v for k, v in comp.items()
                             if k.startswith("close_")}
                    control_rows.append({**context, "scenario": scenario,
                                         "replicate": start + offset, **row, **extra})
            print(f"{key} {audit_half} {scenario}: scored {REPLICATES} paired controls", flush=True)
        pd.DataFrame(control_rows).to_csv(bench_dir / "controls.csv", index=False)
        pd.DataFrame(observed_rows).to_csv(bench_dir / "observed.csv", index=False)
        pd.DataFrame(swap_rows).to_csv(bench_dir / "all_swapped.csv", index=False)
        write_json(bench_dir / "verification.json", tests)
    print(f"{key}: finished in {time.monotonic() - started:.1f}s", flush=True)
    return pd.DataFrame(control_rows), pd.DataFrame(observed_rows), pd.DataFrame(swap_rows), pd.DataFrame(diagnostics_rows)


def pooled_rate(frame, prefix="close_cross_family", random=False):
    if random:
        x = frame.groupby("replicate")[[prefix + "_flips", prefix + "_eligible_pairs"]].sum()
        return x[prefix + "_flips"].to_numpy() / x[prefix + "_eligible_pairs"].to_numpy()
    return float(frame[prefix + "_flips"].sum() / frame[prefix + "_eligible_pairs"].sum())


def tail(values, observed, two_sided=False):
    if two_sided:
        values, observed = np.abs(values), abs(observed)
    return float((1 + np.sum(values >= observed)) / (1 + len(values)))


def summarize(key, controls, observed, swapped, diagnostics, primary_root):
    run = primary_root / (key + "_family_ranking_impact")
    original = pd.read_csv(run / "matched_random_anchor_controls.csv")
    original_median = float(np.median(pooled_rate(original, random=True)))
    low = pooled_rate(observed)
    within_low = pooled_rate(observed, "close_same_family")
    rows = []
    for scenario in SCENARIOS:
        c = controls[controls.scenario.eq(scenario)]
        s = swapped[swapped.scenario.eq(scenario)]
        d = diagnostics[diagnostics.scenario.eq(scenario)]
        null = pooled_rate(c, random=True)
        within_null = pooled_rate(c, "close_same_family", random=True)
        complement = pooled_rate(c, "complement_close_cross_family", random=True)
        high = pooled_rate(s)
        observed_contrast = low - high
        row = {"benchmark": BENCHMARKS[key], "scenario": scenario,
               "low_dif_rate": low, "original_random_median": original_median,
               "paired_random_median": float(np.median(null)),
               "excess_pp": 100 * (low - np.median(null)),
               "empirical_upper_p": tail(null, low),
               "paired_random_q025": float(np.quantile(null, .025)),
               "paired_random_q975": float(np.quantile(null, .975)),
               "all_swapped_rate": high,
               "low_minus_all_swapped_pp": 100 * observed_contrast,
               "paired_contrast_two_sided_p": tail(null - complement, observed_contrast, True),
               "within_low_dif_rate": within_low,
               "within_paired_median": float(np.median(within_null)),
               "within_excess_pp": 100 * (within_low - np.median(within_null)),
               "specificity_excess_pp": 100 * ((low - within_low) - np.median(null - within_null)),
               "specificity_upper_p": tail(null - within_null, low - within_low),
               "matched_anchor_fraction": float(d.n_pairs.sum() / d.n_anchors.sum()),
               "swappable_weight_fraction": float(d.swappable_weight_fraction.mean()),
               "median_control_overlap": float(c.anchor_overlap_fraction.median()),
               "n_pairs_total": int(d.n_pairs.sum())}
        for col in ["profile_rms_logit_median", "profile_rms_logit_p90", "profile_cosine_median",
                    "abs_b_difference_median", "predicted_probability_rms_median",
                    "higher_final_dif_fraction", "dif_difference_median"]:
            row[col + "_fold_mean"] = float(d[col].mean())
        for col in ["mean_audit_residual_dif", "mean_audit_b", "mean_audit_discrimination",
                    "target_citc", "target_independent_item_sem", "target_weighted_alpha", "effective_item_count"]:
            row["low_" + col] = float(observed[col].mean())
            row["control_" + col] = float(c.groupby("replicate")[col].mean().median())
        row["profile_mean_rms_difference_median"] = float(c.profile_mean_rms_difference.median())
        row["max_axis_smd_median"] = float(c.max_abs_profile_axis_smd.median())
        rows.append(row)
    return rows


def report(summary):
    lines = ["# Latent-capability-profile matched diagnostic", "",
             "Post-hoc analysis using frozen primary anchors and all selected spectral dimensions.", "",
             "## Unrestricted optimal matching: ranking results", "",
             "|Benchmark|Low-DIF %|Original random %|Profile-paired random %|Excess pp|Empirical p|All-swapped %|Swappable mass %|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    main = summary[summary.scenario.eq("optimal_unrestricted")]
    for r in main.itertuples():
        lines.append(f"|{r.benchmark}|{100*r.low_dif_rate:.1f}|{100*r.original_random_median:.1f}|"
                     f"{100*r.paired_random_median:.1f}|{r.excess_pp:+.1f}|{r.empirical_upper_p:.3f}|"
                     f"{100*r.all_swapped_rate:.1f}|{100*r.swappable_weight_fraction:.1f}|")
    lines += ["", "## Unrestricted pair quality (mean of fold medians unless marked)", "",
              "|Benchmark|Profile RMS logits|Profile RMS p90|Cosine|Absolute b difference|Probability RMS|Higher final DIF %|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in main.itertuples():
        lines.append(f"|{r.benchmark}|{r.profile_rms_logit_median_fold_mean:.3f}|{r.profile_rms_logit_p90_fold_mean:.3f}|"
                     f"{r.profile_cosine_median_fold_mean:.3f}|{r.abs_b_difference_median_fold_mean:.3f}|"
                     f"{r.predicted_probability_rms_median_fold_mean:.3f}|{100*r.higher_final_dif_fraction_fold_mean:.1f}|")
    lines += ["", "## All scenarios, including caliper coverage", "",
              "|Benchmark|Scenario|Swappable mass %|Control/anchor overlap %|Random reversal %|Excess pp|Empirical p|",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"|{r.benchmark}|{r.scenario}|{100*r.swappable_weight_fraction:.1f}|{100*r.median_control_overlap:.1f}|"
                     f"{100*r.paired_random_median:.1f}|{r.excess_pp:+.1f}|{r.empirical_upper_p:.3f}|")
    lines += ["", "## Interpretation", "",
              "A significant tail result is insufficient without close matching over substantial swappable mass.",
              "When calipers retain few pairs, fixed low-DIF items dominate both forms; this is a restricted perturbation, not a fully capability-balanced replacement test.",
              "Unrestricted optimal matching minimizes distance but does not guarantee that distances are small.",
              "Controls randomly choose either pair member, so overlap with low-DIF anchors is expected and is explicitly reported.",
              "The all-swapped form changes every feasible pair and is reported separately, not treated as 1,000 independent controls.",
              "Tail probabilities assume conditional endpoint exchangeability for this diagnostic; they do not establish causal identification.",
              "Profiles use the final purified spectral basis and all its selected dimensions. Unmodeled capability, nonlinear effects, estimation error, and anchor-dependent basis fitting remain possible.",
              "No threshold/scenario was selected after inspecting target ranking outcomes. All scenarios are reported.", ""]
    return "\n".join(lines)


def reference_counts(full, subset, families, owners):
    """Independent pair enumeration for verification of the vectorized scorer."""
    counts = {"close_cross_family_flips": 0, "close_cross_family_eligible_pairs": 0,
              "close_same_family_flips": 0, "close_same_family_eligible_pairs": 0}
    for i in range(len(full)):
        for j in range(i + 1, len(full)):
            d, a = full[i] - full[j], subset[i] - subset[j]
            if owners[i] == owners[j] or abs(d) > .01 or d == 0 or a == 0:
                continue
            prefix = "close_cross_family" if families[i] != families[j] else "close_same_family"
            counts[prefix + "_eligible_pairs"] += 1
            counts[prefix + "_flips"] += int(d * a < 0)
    return counts


def verify_outputs(out, primary_root):
    """Read-only checks of output hashes, all matching scenarios and sampled scores."""
    completion = json.loads((out / "completion.json").read_text())
    if not completion["complete"] or not completion["all_checks_passed"]:
        raise AssertionError("run did not complete")
    protocol = json.loads((out / "protocol.json").read_text())
    keys = protocol["benchmarks"]
    if protocol["scenarios"] != SCENARIOS or protocol["replicates"] != REPLICATES:
        raise AssertionError("protocol does not match the full diagnostic")
    manifest = json.loads((out / "output_sha256.json").read_text())
    for relative, digest in manifest.items():
        target = (out / relative).resolve()
        if not target.is_relative_to(out.resolve()) or sha256(target) != digest:
            raise AssertionError("output checksum mismatch: " + relative)
    package_root = Path(__file__).resolve().parents[1]
    for relative, digest in protocol["code_sha256"].items():
        target = (package_root / relative).resolve()
        if not target.is_relative_to(package_root) or sha256(target) != digest:
            raise AssertionError("code checksum mismatch: " + relative)
    summary = pd.read_csv(out / "summary.csv")
    checked = 0
    with threadpool_limits(limits=4):
        for key in keys:
            data = previous.mirt_primary.load_data(key)
            bench = out / key
            input_manifest = json.loads((bench / "input_sha256.json").read_text())
            for path in data["input_paths"]:
                if input_manifest[path.name] != sha256(path):
                    raise AssertionError("input checksum mismatch: " + path.name)
            controls = pd.read_csv(bench / "controls.csv")
            if len(controls) != 2 * len(SCENARIOS) * REPLICATES:
                raise AssertionError("incomplete control table")
            if controls.duplicated(["audit_half", "scenario", "replicate"]).any():
                raise AssertionError("duplicate control replicate")
            for fold, target_half in FOLDS:
                saved = np.load(bench / f"{fold}_representation.npz", allow_pickle=False)
                selected, baseline = saved["selected"], saved["weights"]
                cells, embedding = saved["blueprint_cell"], saved["effective_embedding"]
                y = np.asarray(data["responses"][:, saved["target_model_index"]], dtype=float)
                models = pd.read_csv(primary_root / (key + "_family_ranking_impact") /
                                     "owner_family_representatives.csv")
                target = models[models.model_half.eq(target_half)]
                np.testing.assert_array_equal(target.dataset_model_index, saved["target_model_index"])
                full = y.mean(axis=0)
                families, owners = target.family.to_numpy(), target.owner.to_numpy()
                for scenario, caliper in SCENARIOS.items():
                    archive = np.load(bench / f"{fold}_{scenario}_switches.npz", allow_pickle=False)
                    pairs, switches = archive["pairs"], archive["switches"]
                    if switches.shape != (REPLICATES, len(pairs)) or not np.isin(switches, [0, 1]).all():
                        raise AssertionError("invalid paired endpoint switches")
                    if len(np.unique(pairs)) != 2 * len(pairs):
                        raise AssertionError("matching reused an item")
                    pair_table = pd.read_csv(bench / f"{fold}_{scenario}_pairs.csv")
                    if len(pairs):
                        a, b = pairs.T
                        if not selected[a].all() or selected[b].any():
                            raise AssertionError("pair endpoint membership mismatch")
                        np.testing.assert_array_equal(cells[a], cells[b])
                        distances = np.linalg.norm(embedding[a] - embedding[b], axis=1)
                        bdiff = np.abs(saved["item_intercept"][a] - saved["item_intercept"][b])
                        np.testing.assert_allclose(distances, pair_table.profile_rms_logit, atol=1e-12)
                        np.testing.assert_allclose(bdiff, pair_table.abs_b_difference, atol=1e-12)
                        if caliper is not None and not ((distances <= caliper + 1e-10).all() and
                                                       (bdiff <= caliper + 1e-10).all()):
                            raise AssertionError("hard caliper violated")
                        aa, bb = pairs[:10].T
                        offset_difference = ((saved["item_loadings"][aa] / saved["scale"][aa, None]
                                              - saved["item_loadings"][bb] / saved["scale"][bb, None])
                                             @ saved["model_axes"].T)
                        np.testing.assert_allclose(np.sqrt(np.mean(offset_difference ** 2, axis=1)),
                                                   distances[:10], atol=1e-10)
                    for replicate in [0, 137, 499, 999]:
                        weights = baseline.copy()
                        if len(pairs):
                            chosen = pairs[switches[replicate].astype(bool)]
                            weights[chosen[:, 1]] = weights[chosen[:, 0]]
                            weights[chosen[:, 0]] = 0
                        assert_weights(weights[None, :], baseline, cells)
                        score = weights @ y / weights.sum()
                        reference = reference_counts(full, score, families, owners)
                        got = controls[(controls.audit_half == fold) & (controls.scenario == scenario)
                                       & (controls.replicate == replicate)].iloc[0]
                        for column, value in reference.items():
                            if got[column] != value:
                                raise AssertionError(f"score check failed: {key}/{fold}/{scenario}/{replicate}/{column}")
                        checked += 1
            observed = pd.read_csv(bench / "observed.csv")
            observed_rate = observed.close_cross_family_flips.sum() / observed.close_cross_family_eligible_pairs.sum()
            for scenario in SCENARIOS:
                c = controls[controls.scenario.eq(scenario)]
                summed = c.groupby("replicate")[["close_cross_family_flips", "close_cross_family_eligible_pairs"]].sum()
                rates = summed.close_cross_family_flips / summed.close_cross_family_eligible_pairs
                row = summary[(summary.benchmark == BENCHMARKS[key]) & (summary.scenario == scenario)].iloc[0]
                np.testing.assert_allclose([row.low_dif_rate, row.paired_random_median, row.empirical_upper_p],
                                           [observed_rate, rates.median(), (1 + (rates >= observed_rate).sum()) / 1001],
                                           atol=1e-15, rtol=0)
            print(key, "independent checks passed", flush=True)
    print(f"All checks passed: {checked} independently scored configurations; all scenarios and pooled tails checked.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    parser.add_argument("--primary-root", type=Path, default=PROJECT_ROOT / "outputs",
                        help="Parent of the reconstructed primary benchmark directories.")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/capability_profile_diagnostic")
    parser.add_argument("--verify-only", action="store_true",
                        help="Independently verify existing outputs, calipers, scoring, and empirical tails.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    keys = [key.strip() for key in args.benchmarks.split(",") if key.strip()]
    if not keys or len(keys) != len(set(keys)) or set(keys) - set(BENCHMARKS):
        raise ValueError("unknown or repeated benchmark")
    out = args.output_dir.resolve()
    if args.verify_only:
        verify_outputs(out, args.primary_root)
        return
    require_primary_files(args.primary_root, keys, reconstruct=True)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("output is not empty; preserve it and choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    freeze_protocol(out, keys)
    summaries = []
    with threadpool_limits(limits=4):
        for key in keys:
            c, o, s, d = run_benchmark(key, out, args.primary_root)
            summaries.extend(summarize(key, c, o, s, d, args.primary_root))
            frame = pd.DataFrame(summaries)
            frame.to_csv(out / "summary.csv", index=False)
            (out / "report.md").write_text(report(frame))
    write_json(out / "completion.json", {"complete": True, "benchmarks": keys,
               "elapsed_seconds": time.monotonic() - started, "all_checks_passed": True})
    checksums = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "output_sha256.json", checksums)
    print(frame[["benchmark", "scenario", "excess_pp", "empirical_upper_p", "swappable_weight_fraction"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
