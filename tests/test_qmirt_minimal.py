import unittest

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from family_dif_benchmark_audit.discovery.qmirt_minimal import (
    bifactor_logits,
    category_item_predictions,
    choose_threshold,
    fit_bifactor_offsets,
    fit_general_ability,
    fit_multilevel_offsets,
    make_item_split,
    make_model_split,
    multilevel_logits,
)


class MinimalQMIRTTests(unittest.TestCase):
    def test_frozen_item_split_shape_and_determinism(self):
        categories = np.repeat(np.array(["a", "b", "c", "d"]), 25)
        train_a, test_a = make_item_split(categories, 7)
        train_b, test_b = make_item_split(categories, 7)
        self.assertEqual(len(train_a), 70)
        self.assertEqual(len(test_a), 30)
        self.assertTrue(np.array_equal(train_a, train_b))
        self.assertTrue(np.array_equal(test_a, test_b))
        self.assertFalse(np.intersect1d(train_a, test_a).size)

    def test_model_split_is_disjoint_and_deterministic(self):
        population_a, evaluation_a = make_model_split(100, 0.2, 11)
        population_b, evaluation_b = make_model_split(100, 0.2, 11)
        self.assertEqual(len(population_a), 80)
        self.assertEqual(len(evaluation_a), 20)
        self.assertTrue(np.array_equal(population_a, population_b))
        self.assertTrue(np.array_equal(evaluation_a, evaluation_b))
        self.assertFalse(np.intersect1d(population_a, evaluation_a).size)

    def test_category_predictions_use_fit_items_only(self):
        easiness = np.array([0.2, 0.4, 0.8, 0.6, 0.99])
        categories = np.array([0, 0, 1, 1, 0])
        fit = np.array([0, 1, 2, 3])
        pred = np.array([4])
        fit_pred_a, test_pred_a = category_item_predictions(
            easiness, categories, fit, pred, prior_items=0.0
        )
        changed = easiness.copy()
        changed[4] = 0.01
        fit_pred_b, test_pred_b = category_item_predictions(
            changed, categories, fit, pred, prior_items=0.0
        )
        self.assertTrue(np.allclose(fit_pred_a, fit_pred_b))
        self.assertTrue(np.allclose(test_pred_a, test_pred_b))
        self.assertAlmostEqual(test_pred_a[0], np.mean([0.2, 0.4]))

    def test_bifactor_recovers_category_signal_on_unseen_items(self):
        rng = np.random.default_rng(123)
        n_items = 800
        n_models = 80
        category = np.arange(n_items) % 4
        rng.shuffle(category)
        item_logit = rng.normal(0.0, 0.7, n_items)
        general = rng.normal(0.0, 0.8, n_models)
        theta = rng.normal(0.0, 1.0, (n_models, 4))
        theta -= theta.mean(axis=1, keepdims=True)
        logits = item_logit[:, None] + general[None, :] + theta[:, category].T
        response = (rng.uniform(size=logits.shape) < expit(logits)).astype(np.int8)
        train = np.arange(0, 600)
        test = np.arange(600, 800)

        alpha = fit_general_ability(item_logit[train], response[train])
        additive = bifactor_logits(item_logit[test], category[test], alpha)
        alpha_q, theta_q = fit_bifactor_offsets(
            item_logit[train],
            category[train],
            response[train],
            penalty=10.0,
            general_ability=alpha,
        )
        bifactor = bifactor_logits(item_logit[test], category[test], alpha_q, theta_q)
        truth = response[test]
        self.assertGreater(
            roc_auc_score(truth.ravel(), bifactor.ravel()),
            roc_auc_score(truth.ravel(), additive.ravel()) + 0.02,
        )

    def test_threshold_is_learned_on_logit_scale(self):
        truth = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        logits = np.array([-2.0, -1.0, 0.1, 0.2, 1.0, 2.0])
        threshold, balanced = choose_threshold(truth, logits)
        self.assertGreater(threshold, 0.1)
        self.assertLessEqual(threshold, 0.2)
        self.assertEqual(balanced, 1.0)

    def test_multilevel_q_recovers_two_overlapping_factors(self):
        rng = np.random.default_rng(456)
        n_items = 1000
        n_models = 60
        category = np.arange(n_items) % 4
        subdomain = np.arange(n_items) % 10
        rng.shuffle(category)
        rng.shuffle(subdomain)
        item_logit = rng.normal(0.0, 0.5, n_items)
        general = rng.normal(0.0, 0.7, n_models)
        category_theta = rng.normal(0.0, 0.6, (n_models, 4))
        source_theta = rng.normal(0.0, 0.8, (n_models, 10))
        logits = (
            item_logit[:, None]
            + general[None, :]
            + category_theta[:, category].T
            + source_theta[:, subdomain].T
        )
        response = (rng.uniform(size=logits.shape) < expit(logits)).astype(np.int8)
        train = np.arange(750)
        test = np.arange(750, 1000)
        alpha = fit_general_ability(item_logit[train], response[train])
        additive = bifactor_logits(item_logit[test], category[test], alpha)
        alpha_q, effects = fit_multilevel_offsets(
            item_logit[train],
            [category[train], subdomain[train]],
            [4, 10],
            response[train],
            [10.0, 10.0],
            general_ability=alpha,
            cycles=4,
        )
        hierarchical = multilevel_logits(
            item_logit[test],
            [category[test], subdomain[test]],
            alpha_q,
            effects,
        )
        self.assertGreater(
            roc_auc_score(response[test].ravel(), hierarchical.ravel()),
            roc_auc_score(response[test].ravel(), additive.ravel()) + 0.02,
        )


if __name__ == "__main__":
    unittest.main()
