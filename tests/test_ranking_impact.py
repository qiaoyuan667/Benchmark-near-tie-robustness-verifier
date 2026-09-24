import unittest

import numpy as np
import pandas as pd

from family_dif_benchmark_audit.ranking.routereval import composition_bootstrap_units
from family_dif_benchmark_audit.ranking.mmlu_pro import (
    difficulty_cells,
    matched_random_anchor_weights,
    pairwise_ranking_metrics,
    select_owner_family_representatives,
    stratified_anchor_weights,
)


class FamilyRankingImpactTests(unittest.TestCase):
    def test_anchor_weights_restore_every_cell_mass(self):
        magnitude = np.asarray([0.9, 0.1, 0.4, 0.2, 0.7, 0.3])
        cells = np.asarray(["a", "a", "a", "b", "b", "b"])
        selected, weights = stratified_anchor_weights(magnitude, cells, 2.0 / 3.0)
        self.assertEqual(int(selected.sum()), 4)
        self.assertAlmostEqual(float(weights[cells == "a"].sum()), 3.0)
        self.assertAlmostEqual(float(weights[cells == "b"].sum()), 3.0)
        self.assertTrue(selected[1])
        self.assertTrue(selected[3])

    def test_difficulty_cells_are_nested_within_source(self):
        sources = np.asarray(["x"] * 8 + ["y"] * 8)
        intercepts = np.asarray([8, 1, 7, 2, 6, 3, 5, 4] * 2, dtype=float)
        cells = difficulty_cells(sources, intercepts, 4)
        self.assertEqual(len(np.unique(cells)), 8)
        for cell in np.unique(cells):
            positions = np.flatnonzero(cells == cell)
            self.assertEqual(len(np.unique(sources[positions])), 1)

    def test_random_control_matches_anchor_count_and_cell_mass(self):
        cells = np.asarray(["a", "a", "a", "b", "b", "b", "b"])
        template = np.asarray([True, True, False, True, False, True, False])
        weights = matched_random_anchor_weights(
            cells, template, np.random.default_rng(7)
        )
        self.assertEqual(int(np.sum(weights > 0)), int(template.sum()))
        for cell in np.unique(cells):
            mask = cells == cell
            self.assertAlmostEqual(float(weights[mask].sum()), float(mask.sum()))

    def test_owner_family_representative_is_median_checkpoint(self):
        frame = pd.DataFrame(
            {
                "owner": ["o", "o", "o", "p"],
                "family": ["qwen", "qwen", "qwen", "llama"],
                "model_id": ["low", "middle", "high", "only"],
                "train_accuracy": [0.2, 0.5, 0.9, 0.3],
                "model_half": ["discovery"] * 4,
            }
        )
        result = select_owner_family_representatives(frame)
        self.assertEqual(set(result.model_id), {"middle", "only"})

    def test_pairwise_metrics_detect_cross_family_close_flip(self):
        full = np.asarray([0.50, 0.505, 0.70])
        anchor = np.asarray([0.51, 0.49, 0.72])
        families = np.asarray(["qwen", "llama", "gemma"])
        owners = np.asarray(["a", "b", "c"])
        metrics = pairwise_ranking_metrics(
            full, anchor, families, owners, close_gap=0.01
        )
        self.assertEqual(metrics["close_cross_family_eligible_pairs"], 1)
        self.assertEqual(metrics["close_cross_family_flips"], 1)
        self.assertAlmostEqual(metrics["close_cross_family_flip_rate"], 1.0)

    def test_composition_bootstrap_prefers_native_blocks(self):
        sources = np.asarray(["a", "b", "c", "d", "e"] * 4)
        units, strategy = composition_bootstrap_units(sources, 100)
        self.assertEqual(strategy, "native_source_block")
        np.testing.assert_array_equal(units, sources)

    def test_composition_bootstrap_clusters_single_task_items(self):
        sources = np.asarray(["hellaswag"] * 1000)
        units, strategy = composition_bootstrap_units(sources, 100)
        self.assertEqual(strategy, "contiguous_item_cluster")
        self.assertEqual(len(np.unique(units)), 100)
        counts = pd.Series(units).value_counts()
        self.assertEqual(int(counts.min()), 10)
        self.assertEqual(int(counts.max()), 10)


if __name__ == "__main__":
    unittest.main()
