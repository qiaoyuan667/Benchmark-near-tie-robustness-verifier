import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("release_contract_audit", ROOT / "tools/audit_release.py")
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
BENCHMARKS = ["MMLU-Pro", "BBH", "MMLU", "HellaSwag", "WinoGrande"]
BENCHMARK_KEYS = ["mmlu_pro", "bbh", "mmlu", "hellaswag", "winogrande"]
AXES = {
    "quantitative_symbolic",
    "formal_rule_reasoning",
    "factual_domain_knowledge",
    "contextual_reading",
    "commonsense_narrative",
    "spatial_temporal",
    "linguistic_wordplay",
    "negation_exception",
    "code_structured_representation",
    "distractor_discrimination",
}


def test_submission_protocol_has_five_retained_benchmarks_and_no_math():
    protocol = json.loads((ROOT / "configs/analysis_protocol.json").read_text())
    assert protocol["version"] == "submission"
    assert protocol["benchmarks"] == BENCHMARK_KEYS
    assert "math_lvl5" not in protocol["benchmarks"]
    assert protocol["primary_ability_model"]["candidate_dimensions"] == [
        1, 2, 4, 8, 16, 32, 64, 128, 256
    ]


def test_primary_summary_matches_frozen_headline_values():
    frame = pd.read_csv(ROOT / "results/primary/benchmark_summary.csv").set_index(
        "benchmark"
    )
    expected = {
        "MMLU-Pro": (0.470779, 0.185065, 0.285714, 0.000999),
        "BBH": (0.421384, 0.252471, 0.168913, 0.000999),
        "MMLU": (0.406801, 0.163215, 0.243586, 0.000999),
        "HellaSwag": (0.309287, 0.114090, 0.195197, 0.000999),
        "WinoGrande": (0.338177, 0.346926, -0.008749, 0.689311),
    }
    assert list(frame.index) == BENCHMARKS
    for benchmark, values in expected.items():
        observed = frame.loc[benchmark]
        np.testing.assert_allclose(
            [
                observed.close_cross_family_flip_rate,
                observed.random_control_close_median,
                observed.excess_close_flip_rate,
                observed.random_control_close_p,
            ],
            values,
            atol=5e-6,
            rtol=0,
        )


def test_exact_common_release_is_the_promised_15_row_matrix():
    frame = pd.read_csv(
        ROOT / "results/primary/pairwise_exact_common_family_shift_summary.csv"
    )
    assert len(frame) == 15
    assert frame["pair"].nunique() == 3
    assert frame.groupby("pair")["family"].nunique().eq(5).all()
    assert not {"model_id", "owner", "uploader"}.intersection(frame.columns)


def test_every_primary_run_protocol_has_seeds_and_input_hashes():
    expected_dimensions = {
        "mmlu_pro": {"discovery": 64, "validation": 32},
        "bbh": {"discovery": 128, "validation": 128},
        "mmlu": {"discovery": 128, "validation": 128},
        "hellaswag": {"discovery": 16, "validation": 64},
        "winogrande": {"discovery": 8, "validation": 8},
    }
    expected_candidates = {
        "mmlu_pro": [1, 2, 4, 8, 16, 32, 64, 128],
        "bbh": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "mmlu": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "hellaswag": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "winogrande": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    }
    for benchmark in BENCHMARK_KEYS:
        path = ROOT / "results/primary/per_benchmark" / benchmark / "protocol.json"
        protocol = json.loads(path.read_text())
        assert protocol["selected_dimensions"] == expected_dimensions[benchmark]
        assert protocol["candidate_dimensions"] == expected_candidates[benchmark]
        assert set(protocol["seeds"]) == {
            "family_split",
            "nested_owner_validation",
            "svd",
            "composition_bootstrap_base",
            "matched_random_base",
            "owner_bootstrap",
        }
        assert protocol["inputs"]
        for record in protocol["inputs"]:
            assert record["path"].startswith("${PROJECT_ROOT}/")
            assert len(record["sha256"]) == 64


def test_gap_primary_rows_equal_main_summary():
    main = pd.read_csv(ROOT / "results/primary/benchmark_summary.csv").set_index(
        "benchmark"
    )
    gap = pd.read_csv(
        ROOT / "results/gap_sensitivity/gap_sensitivity_summary.csv"
    )
    primary = gap[np.isclose(gap["gap_points"], 1.0)].set_index("benchmark")
    assert set(primary.index) == set(main.index)
    np.testing.assert_allclose(
        primary.loc[main.index, "observed_flip_rate"],
        main["close_cross_family_flip_rate"],
    )


def test_population_robustness_is_complete_and_primary_positives_survive():
    frame = pd.read_csv(
        ROOT / "results/population_robustness/robustness_summary.csv"
    )
    assert len(frame) == 50
    assert frame.groupby("benchmark")["variant"].nunique().eq(10).all()
    positive = frame[frame["benchmark"].isin(BENCHMARKS[:4])]
    assert positive["significant_positive_excess"].all()


def test_winogrande_is_significant_in_only_one_population_spec():
    frame = pd.read_csv(
        ROOT / "results/population_robustness/robustness_summary.csv"
    )
    wino = frame[frame["benchmark"].eq("WinoGrande")]
    assert int(wino["significant_positive_excess"].sum()) == 1


def test_all_item_signature_gates_pass():
    frame = pd.read_csv(ROOT / "results/item_stability/stability_summary.csv")
    assert set(frame["benchmark"]) == set(BENCHMARKS)
    assert frame["passes_all_stability_gates"].all()
    np.testing.assert_allclose(frame["family_within_cell_permutation_p"], 1 / 501)


def test_source_attribution_counts_match_the_paper():
    source = pd.read_csv(
        ROOT / "results/item_stability/source_validation_summary.csv"
    ).set_index("benchmark")
    direct = pd.read_csv(
        ROOT / "results/item_stability/source_shift_driver_validation_summary.csv"
    ).set_index("benchmark")
    assert int(source.loc["Pooled source-rich main", "validation_sign_replicated"]) == 58
    assert int(direct.loc["Pooled source-rich main", "validation_sign_replicated"]) == 43


def test_content_audit_remains_a_confirmatory_null():
    binary = pd.read_csv(ROOT / "results/content_audit/paired_binary_enrichment.csv")
    protocol = json.loads((ROOT / "results/content_audit/protocol.json").read_text())
    post = json.loads(
        (ROOT / "results/content_audit/post_annotation_audit.json").read_text()
    )
    assert not binary["replicated_confirmatory_axis"].any()
    assert np.isclose(binary["paired_difference"].abs().max(), 0.096)
    assert np.isclose(
        binary.loc[
            binary["annotator"].eq("annotator_a")
            & binary["axis"].eq("spatial_temporal"),
            "bh_q",
        ].iloc[0],
        0.1063748788,
    )
    assert post["exploratory_family_direction"]["stable_direction_items"] == 158
    assert protocol["semantic_gate_passed"] is False


def test_annotation_rubric_and_schema_cover_all_frozen_axes():
    schema = json.loads(
        (ROOT / "configs/blinded_content_annotation_schema.json").read_text()
    )
    properties = schema["properties"]["annotations"]["items"]["properties"]
    assert AXES.issubset(properties)
    rubric = (ROOT / "configs/blinded_content_rubric.md").read_text()
    assert all(axis in rubric for axis in AXES)


def test_public_tree_contains_no_forbidden_binary_or_sensitive_result_columns():
    forbidden = {".pkl", ".npy", ".npz", ".parquet", ".pt", ".safetensors"}
    assert not [path for path in _audit.public_paths(ROOT) if path.suffix.lower() in forbidden]
    sensitive = {"question", "question_text", "problem", "rationale", "prompt"}
    for path in (ROOT / "results").rglob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        assert not columns.intersection(sensitive), path
