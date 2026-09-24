import unittest

import numpy as np
import pandas as pd

from family_dif_benchmark_audit.interpretation.blinded_content_v3 import (
    BINARY_AXES,
    ORDINAL_AXES,
    bh_adjust,
    cohen_kappa,
    conditional_family_statistic,
    paired_binary_results,
    validate_payload,
)


class BlindedContentAnnotationTests(unittest.TestCase):
    def test_cohen_kappa_is_one_for_non_degenerate_perfect_agreement(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        self.assertEqual(cohen_kappa(labels, labels), 1.0)

    def test_bh_adjust_matches_known_example(self):
        adjusted = bh_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])

    def test_validate_payload_rejects_invalid_binary_value(self):
        row = {"blinded_id": "item_0001"}
        row.update({axis: 0 for axis in BINARY_AXES})
        row.update({axis: 0 for axis in ORDINAL_AXES})
        row[BINARY_AXES[0]] = 2
        with self.assertRaises(ValueError):
            validate_payload({"annotations": [row]}, ["item_0001"])

    def test_paired_binary_result_detects_consistent_large_enrichment(self):
        rows = []
        for benchmark in ("a", "b", "c", "d", "e"):
            for pair_index in range(20):
                for group, value in (("stable_high_dif", 1), ("matched_control", 0)):
                    row = {
                        "annotator": "annotator_a",
                        "benchmark": benchmark,
                        "pair_id": "%s_%d" % (benchmark, pair_index),
                        "group": group,
                    }
                    row.update({axis: 0 for axis in BINARY_AXES})
                    row[BINARY_AXES[0]] = value
                    rows.append(row)
        result = paired_binary_results(pd.DataFrame(rows), "annotator_a")
        detected = result.set_index("axis").loc[BINARY_AXES[0]]
        self.assertEqual(detected["paired_difference"], 1.0)
        self.assertEqual(int(detected["benchmarks_same_direction"]), 5)
        self.assertLess(float(detected["bh_q"]), 0.05)

    def test_conditional_family_statistic_respects_benchmark_strata(self):
        families = np.asarray(["gemma", "gemma", "llama", "llama"] * 2)
        benchmarks = np.asarray(["a"] * 4 + ["b"] * 4)
        associated = np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
        balanced = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
        self.assertGreater(
            conditional_family_statistic(associated, families, benchmarks),
            conditional_family_statistic(balanced, families, benchmarks),
        )


if __name__ == "__main__":
    unittest.main()
