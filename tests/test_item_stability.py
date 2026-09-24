import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from family_dif_benchmark_audit.interpretation.item_stability_v3 import (
    build_summary_svg,
    center_within_cells,
    selected_replication_summary,
    source_shift_attribution,
    stratified_high_mask,
)


class ItemDifSignatureStabilityTests(unittest.TestCase):
    def test_center_within_cells_removes_each_cell_mean(self):
        values = np.asarray([[1.0, 4.0], [3.0, 8.0], [10.0, 2.0], [14.0, 6.0]])
        cells = np.asarray(["a", "a", "b", "b"])
        centered = center_within_cells(values, cells)
        for cell in np.unique(cells):
            np.testing.assert_allclose(centered[cells == cell].mean(axis=0), 0.0)

    def test_stratified_high_mask_selects_each_cell_tail(self):
        values = np.asarray([1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0])
        cells = np.asarray(["a"] * 4 + ["b"] * 4)
        selected = stratified_high_mask(values, cells, 0.5)
        np.testing.assert_array_equal(np.flatnonzero(selected), np.asarray([1, 3, 4, 6]))

    def test_source_shift_decomposition_is_exact(self):
        responses = np.asarray(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.int8,
        )
        sources = np.asarray(["a", "a", "b", "b"])
        representatives = pd.DataFrame(
            {
                "dataset_model_index": [0, 1, 2, 3],
                "family": ["gemma", "qwen", "gemma", "qwen"],
                "model_half": ["validation", "validation", "discovery", "discovery"],
            }
        )
        fits = {
            "discovery": SimpleNamespace(weights=np.asarray([2.0, 0.0, 2.0, 0.0])),
            "validation": SimpleNamespace(weights=np.asarray([0.0, 2.0, 0.0, 2.0])),
        }
        _, selected, audit = source_shift_attribution(
            "toy", responses, sources, representatives, fits
        )
        self.assertEqual(len(selected), 8)
        self.assertLess(float(audit["absolute_decomposition_error_points"].max()), 1e-10)

    def test_selected_replication_summary_pools_only_main_benchmarks(self):
        selected = pd.DataFrame(
            {
                "benchmark": ["MMLU-Pro"] * 4 + ["BBH"] * 4 + ["WinoGrande"] * 4,
                "validation_sign_replicated": [True] * 8 + [False] * 4,
            }
        )
        summary = selected_replication_summary(
            selected,
            ["MMLU-Pro", "BBH", "WinoGrande"],
            "Pooled source-rich main",
        )
        pooled = summary[summary["benchmark"].eq("Pooled source-rich main")].iloc[0]
        self.assertEqual(int(pooled.selected_extremes), 8)
        self.assertEqual(float(pooled.validation_sign_replication_rate), 1.0)

    def test_summary_svg_contains_both_panels(self):
        stability = pd.DataFrame(
            [
                {
                    "benchmark": benchmark,
                    "median_family_within_cell_spearman": 0.5,
                    "top20_overlap_precision": 0.4,
                    "stable_top20_advantaged_family_agreement": 0.7,
                }
                for benchmark in ("MMLU-Pro", "BBH", "MMLU", "HellaSwag", "WinoGrande")
            ]
        )
        attribution = pd.DataFrame(
            [
                {"benchmark": benchmark, "validation_sign_replication_rate": 0.8}
                for benchmark in ("MMLU-Pro", "BBH", "MMLU", "WinoGrande")
            ]
        )
        svg = build_summary_svg(stability, attribution)
        self.assertIn("Item-level signature stability", svg)
        self.assertIn("Direct source attribution", svg)
        self.assertIn("WinoGrande", svg)


if __name__ == "__main__":
    unittest.main()
