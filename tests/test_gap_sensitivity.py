import unittest

import numpy as np

from family_dif_benchmark_audit.robustness.gap_sensitivity_v3 import (
    gap_curve_counts,
    pair_context,
)


class RankingGapSensitivityTests(unittest.TestCase):
    def test_gap_curve_is_nested_and_detects_reversal(self):
        full = np.asarray([0.500, 0.504, 0.520, 0.560])
        alternative = np.asarray([0.510, 0.490, 0.530, 0.550])
        families = np.asarray(["qwen", "phi", "llama", "gemma"])
        owners = np.asarray(["a", "b", "c", "d"])
        rows = gap_curve_counts(
            full,
            alternative,
            pair_context(full, families, owners),
        )
        pairs = [row["eligible_pairs"] for row in rows]
        self.assertEqual(pairs, sorted(pairs))
        half_point = next(row for row in rows if np.isclose(row["gap_points"], 0.5))
        self.assertEqual(half_point["eligible_pairs"], 1)
        self.assertEqual(half_point["flips"], 1)

    def test_same_owner_pair_is_excluded(self):
        full = np.asarray([0.500, 0.504])
        alternative = np.asarray([0.510, 0.490])
        families = np.asarray(["qwen", "phi"])
        owners = np.asarray(["same", "same"])
        rows = gap_curve_counts(
            full,
            alternative,
            pair_context(full, families, owners),
        )
        self.assertTrue(all(row["eligible_pairs"] == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
