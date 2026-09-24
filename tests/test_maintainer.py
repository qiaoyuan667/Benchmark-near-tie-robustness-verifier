from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from family_dif_benchmark_audit.maintainer import (
    AuditConfig, BenchmarkData, fit_audit_half, load_benchmark, main,
    prepare_population, run_audit, summarize_reversals, write_audit,
)
from family_dif_benchmark_audit.ranking.mmlu_pro import (
    matched_random_anchor_weights, pairwise_ranking_metrics,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(39)
    n_items, n_owners, n_families = 160, 32, 3
    n_models = n_owners * n_families
    logits = rng.normal(size=(n_items, 2)) @ rng.normal(size=(2, n_models))
    return BenchmarkData(
        rng.binomial(1, expit(logits)).astype(np.int8),
        np.asarray([f"checkpoint_{m:04d}" for m in range(n_models)]),
        np.asarray([f"private_family_{m % n_families}" for m in range(n_models)]),
        np.asarray([f"private_owner_{m // n_families}" for m in range(n_models)]),
        np.asarray([f"private_group_{i % 4}" for i in range(n_items)]),
    )


@pytest.fixture
def config():
    return AuditConfig(dimensions=(1, 2), purification_rounds=1,
                       dif_cycles=2, resamples=7, gap_pp=5)


def test_frozen_audit_fit_ignores_all_opposite_owner_responses(data, config):
    population = prepare_population(data, config)
    assert population.groupby("owner").model_half.nunique().max() == 1
    original, cv, selection = fit_audit_half(data, population, "discovery", config, 0)
    changed = data.responses.copy()
    target_columns = population.loc[population.model_half.eq("validation"), "dataset_model_index"]
    changed[:, target_columns] = 1 - changed[:, target_columns]
    other = replace(data, responses=changed)
    second, cv2, selection2 = fit_audit_half(other, population, "discovery", config, 0)
    np.testing.assert_array_equal(original.weights, second.weights)
    np.testing.assert_array_equal(original.selected, second.selected)
    np.testing.assert_allclose(original.effects, second.effects, rtol=0, atol=0)
    pd.testing.assert_frame_equal(cv, cv2)
    assert selection == selection2


def test_selected_and_random_weights_preserve_each_blueprint_cell(data, config):
    population = prepare_population(data, config)
    fit, _, _ = fit_audit_half(data, population, "discovery", config, 0)
    random_weights = matched_random_anchor_weights(fit.cells, fit.selected, np.random.default_rng(12))
    for cell in np.unique(fit.cells):
        mask = fit.cells == cell
        assert fit.weights[mask].sum() == pytest.approx(mask.sum())
        assert random_weights[mask].sum() == pytest.approx(mask.sum())
        assert np.count_nonzero(random_weights[mask]) == fit.selected[mask].sum()


def test_cross_family_pairs_exclude_same_owner_and_either_exact_tie():
    metrics = pairwise_ranking_metrics(
        np.array([.500, .505, .505, .510]),
        np.array([.510, .490, .510, .490]),
        np.array(["A", "B", "A", "B"]),
        np.array(["shared", "shared", "owner_2", "owner_3"]), .02,
    )
    # (0,1) same owner; (1,2) full tie; (1,3) weighted tie and same family.
    # Exactly (0,3) and (2,3) are eligible cross-family reversals.
    assert metrics["close_cross_family_eligible_pairs"] == 2
    assert metrics["close_cross_family_flips"] == 2


def test_pooled_counts_not_mean_of_fold_rates_and_correct_randomization_tail():
    observed = pd.DataFrame({"close_cross_family_eligible_pairs": [2, 8],
                             "close_cross_family_flips": [2, 0]})
    controls = pd.DataFrame({
        "replicate": [0, 0, 1, 1, 2, 2],
        "close_cross_family_eligible_pairs": [2, 8, 2, 8, 2, 8],
        "close_cross_family_flips": [0, 1, 0, 3, 0, 0],
    })
    summary, pooled = summarize_reversals(observed, controls)
    assert summary["low_dif_reversal_rate"] == .2  # not .5
    assert summary["matched_random_median_rate"] == .1
    assert summary["excess_pp"] == 10
    assert summary["one_sided_randomization_p"] == .5
    assert pooled.eligible_pairs.tolist() == [10, 10, 10]


def test_no_eligible_pairs_produces_no_false_robustness_claim(data, config):
    tied = replace(data, responses=np.ones_like(data.responses))
    result = run_audit(tied, config)
    assert result["summary"]["eligible_pairs"] == 0
    assert result["summary"]["status"] == "insufficient_eligible_pairs"
    assert result["summary"]["one_sided_randomization_p"] is None
    assert result["summary"]["excess_pp"] is None


def test_undefined_random_replicate_withholds_inference():
    observed = pd.DataFrame({"close_cross_family_eligible_pairs": [2, 8],
                             "close_cross_family_flips": [2, 0]})
    controls = pd.DataFrame({"replicate": [0, 0, 1, 1],
        "close_cross_family_eligible_pairs": [2, 8, 0, 0],
        "close_cross_family_flips": [0, 1, 0, 0]})
    summary, _ = summarize_reversals(observed, controls)
    assert summary["status"] == "undefined_random_controls"
    assert summary["one_sided_randomization_p"] is None


def test_reproducible_pipeline_and_reports_suppress_original_identifiers(data, config, tmp_path):
    one, two = run_audit(data, config), run_audit(data, config)
    assert one["summary"] == two["summary"]
    for key, frame in one.items():
        if isinstance(frame, pd.DataFrame):
            pd.testing.assert_frame_equal(frame, two[key])
    output = tmp_path / "audit"
    write_audit(one, output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["config"]["gap_pp"] == 5
    for path in output.iterdir():
        text = path.read_text()
        assert "private_owner" not in text
        assert "private_family" not in text
        assert "private_group" not in text
        assert "checkpoint_" not in text
        assert str(tmp_path) not in text
    with pytest.raises(ValueError, match="not empty"):
        write_audit(one, output)


def test_npz_and_npy_csv_loaders_align_and_record_only_basename(data, tmp_path):
    payload = {name: getattr(data, name) for name in
               ("responses", "model_id", "family", "owner", "item_group")}
    npz = tmp_path / "benchmark.npz"
    np.savez_compressed(npz, **payload)
    loaded = load_benchmark(npz)
    np.testing.assert_array_equal(loaded.responses, data.responses)
    assert loaded.provenance[0]["file"] == "benchmark.npz"
    assert len(loaded.provenance[0]["sha256"]) == 64
    matrix, models, items = [tmp_path / name for name in ("responses.npy", "models.csv", "items.csv")]
    np.save(matrix, data.responses)
    pd.DataFrame({name: getattr(data, name) for name in ("model_id", "family", "owner")}).to_csv(models, index=False)
    pd.DataFrame({"item_group": data.item_group}).to_csv(items, index=False)
    alternate = load_benchmark(matrix, models, items)
    for name in payload:
        np.testing.assert_array_equal(getattr(alternate, name), payload[name])
    with pytest.raises(ValueError, match="requires"):
        load_benchmark(matrix)


@pytest.mark.parametrize("field,value,match", [
    ("responses", np.zeros((3, 4, 5)), "item-by-model"),
    ("family", np.array(["A"]), "length"),
    ("owner", np.array([""] * 96), "blank"),
    ("model_id", np.array(["same"] * 96), "uniquely"),
])
def test_invalid_metadata_rejected(data, field, value, match):
    bad = replace(data, **{field: value})
    with pytest.raises(ValueError, match=match):
        bad.validate()


def test_nonbinary_and_missing_responses_rejected(data):
    for invalid in (-1, np.nan, .5):
        truth = data.responses.astype(float)
        truth[0, 0] = invalid
        with pytest.raises(ValueError, match="binary"):
            replace(data, responses=truth).validate()


def test_cap_applies_per_owner_family_and_nested_population_checks(data, config):
    expanded = BenchmarkData(
        np.repeat(data.responses, 3, axis=1),
        np.array([f"m_{i}" for i in range(data.responses.shape[1] * 3)]),
        np.repeat(data.family, 3), np.repeat(data.owner, 3), data.item_group,
    )
    population = prepare_population(expanded, replace(config, owner_cap=2))
    assert population.groupby(["owner", "family"]).size().max() == 2
    with pytest.raises(ValueError, match="fewer capped"):
        prepare_population(data, replace(config, min_family_models=100))
    with pytest.raises(ValueError, match="resamples"):
        run_audit(data, replace(config, resamples=0))


def test_cli_end_to_end(data, tmp_path):
    archive = tmp_path / "sample.npz"
    np.savez_compressed(archive, **{name: getattr(data, name) for name in
                       ("responses", "model_id", "family", "owner", "item_group")})
    output = tmp_path / "result"
    main([str(archive), "--output", str(output), "--quick", "--resamples", "3"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["config"]["mode"] == "quick-exploratory"
    assert summary["config"]["dimensions"] == [1, 2, 4, 8, 16, 32]
    assert summary["config"]["purification_rounds"] == 3
    assert summary["random_control_replicates"] == 3
    assert len(summary["folds"]) == 2
