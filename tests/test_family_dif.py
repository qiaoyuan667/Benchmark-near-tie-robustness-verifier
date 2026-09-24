import unittest

import numpy as np
from scipy.special import expit

from family_dif_benchmark_audit.core.family_dif import (
    DEFAULT_FAMILIES,
    classify_family,
    fit_family_dif,
    fit_rasch_abilities,
    select_and_split_family_models,
)


class FamilyDIFTests(unittest.TestCase):
    def test_classifier_excludes_ambiguous_architecture_names(self):
        self.assertEqual(
            DEFAULT_FAMILIES, ("qwen", "llama", "gemma", "mistral", "phi")
        )
        self.assertEqual(classify_family("owner__Qwen2.5-7B")[0], "qwen")
        self.assertEqual(classify_family("owner__Llama-3.1-8B")[0], "llama")
        self.assertEqual(classify_family("owner__Mistral-7B")[0], "mistral")
        self.assertEqual(classify_family("owner__Mixtral-8x7B")[0], "mistral")
        family, reason = classify_family("owner__DeepSeek-Qwen-7B")
        self.assertIsNone(family)
        self.assertTrue(reason.startswith("ambiguous:"))

    def test_owner_groups_do_not_cross_halves_and_are_capped(self):
        model_ids = []
        for family in ("Qwen", "Llama"):
            for owner in range(8):
                for variant in range(6):
                    model_ids.append("o%d_%s__%s-%d" % (owner, family, family, variant))
        model_ids = np.asarray(model_ids)
        accuracy = np.linspace(0.2, 0.8, len(model_ids))
        selected = select_and_split_family_models(
            model_ids, accuracy, ["qwen", "llama"], 3, 10, 7
        )
        self.assertLessEqual(selected.groupby(["family", "owner"]).size().max(), 3)
        self.assertEqual(selected.groupby("owner")["model_half"].nunique().max(), 1)
        self.assertEqual(set(selected.model_half), {"discovery", "validation"})

    def test_family_dif_recovers_injected_relative_advantages(self):
        rng = np.random.default_rng(9)
        n_items = 500
        n_per_family = 30
        family_codes = np.repeat(np.arange(3), n_per_family)
        ability = rng.normal(0.0, 0.7, len(family_codes))
        item = rng.normal(0.0, 0.6, n_items)
        true_effect = rng.normal(0.0, 0.5, (n_items, 3))
        true_effect -= true_effect.mean(axis=1, keepdims=True)
        logits = item[:, None] + ability[None, :] + true_effect[:, family_codes]
        response = (rng.uniform(size=logits.shape) < expit(logits)).astype(np.int8)
        fitted_ability, _ = fit_rasch_abilities(response)
        _, fitted_effect = fit_family_dif(
            response, fitted_ability, family_codes, 3, penalty=3.0, cycles=5
        )
        correlations = [
            np.corrcoef(true_effect[:, family], fitted_effect[:, family])[0, 1]
            for family in range(3)
        ]
        self.assertGreater(float(np.mean(correlations)), 0.45)


if __name__ == "__main__":
    unittest.main()
