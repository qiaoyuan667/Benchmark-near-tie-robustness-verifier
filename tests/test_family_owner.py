import unittest

import pandas as pd

from family_dif_benchmark_audit.robustness.family_owner_v3 import (
    STRICT_LINEAGE_PATTERN,
    build_matrix_svg,
    criterion_rows,
    lexicographic_family_models,
)


class FamilyOwnerRobustnessTests(unittest.TestCase):
    def test_strict_lineage_filter_catches_named_derivatives(self):
        excluded = [
            "owner/Qwen2.5-7B-merge",
            "owner/Llama-3-LoRA",
            "owner/Phi-3-DPO",
            "owner/gemma-distilled",
        ]
        retained = ["Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-9b-it"]
        self.assertTrue(all(STRICT_LINEAGE_PATTERN.search(name) for name in excluded))
        self.assertTrue(all(not STRICT_LINEAGE_PATTERN.search(name) for name in retained))

    def test_lexicographic_checkpoint_is_one_per_owner_family(self):
        rows = []
        halves = {}
        families = ("gemma", "llama", "mistral", "phi", "qwen")
        for family in families:
            for index in range(15):
                owner = f"{family}_owner_{index:02d}"
                halves[owner] = "discovery" if index < 8 else "validation"
                rows.extend(
                    [
                        {
                            "owner": owner,
                            "family": family,
                            "model_id": f"{owner}/z_checkpoint",
                            "dataset_model_index": 2 * index,
                        },
                        {
                            "owner": owner,
                            "family": family,
                            "model_id": f"{owner}/a_checkpoint",
                            "dataset_model_index": 2 * index + 1,
                        },
                    ]
                )
        selected = lexicographic_family_models(pd.DataFrame(rows), halves)
        self.assertEqual(len(selected), 75)
        self.assertTrue(selected["model_id"].str.endswith("a_checkpoint").all())
        self.assertEqual(selected.groupby(["owner", "family"]).size().max(), 1)

    def test_gate_counts_only_the_four_baseline_positive_benchmarks(self):
        rows = []
        main = ("MMLU-Pro", "BBH", "MMLU", "HellaSwag")
        for variant in (
            "owner_cap_1",
            "owner_cap_3",
            "owner_cap_5_baseline",
            "lexicographic_checkpoint",
            "strict_lineage",
            "omit_gemma",
            "omit_llama",
            "omit_mistral",
            "omit_phi",
            "omit_qwen",
        ):
            for benchmark in main:
                rows.append(
                    {
                        "benchmark": benchmark,
                        "variant": variant,
                        "positive_excess": benchmark != "HellaSwag",
                        "significant_positive_excess": benchmark != "HellaSwag",
                    }
                )
            rows.append(
                {
                    "benchmark": "WinoGrande",
                    "variant": variant,
                    "positive_excess": False,
                    "significant_positive_excess": False,
                }
            )
        result = criterion_rows(pd.DataFrame(rows))
        self.assertTrue(result["passes_direction_gate"].all())
        self.assertTrue(result["passes_significance_gate"].all())

    def test_svg_contains_every_cell(self):
        variants = (
            "owner_cap_1",
            "owner_cap_3",
            "owner_cap_5_baseline",
            "lexicographic_checkpoint",
            "strict_lineage",
            "omit_gemma",
            "omit_llama",
            "omit_mistral",
            "omit_phi",
            "omit_qwen",
        )
        benchmarks = ("MMLU-Pro", "BBH", "MMLU", "HellaSwag", "WinoGrande")
        frame = pd.DataFrame(
            [
                {
                    "benchmark": benchmark,
                    "variant": variant,
                    "excess_close_flip_rate": 0.1,
                    "significant_positive_excess": True,
                }
                for benchmark in benchmarks
                for variant in variants
            ]
        )
        svg = build_matrix_svg(frame)
        self.assertEqual(svg.count("+10.0 pp*"), 50)
        self.assertIn("WinoGrande", svg)


if __name__ == "__main__":
    unittest.main()
