"""Scientific invariants of post-hoc controls, matching and independent scoring."""

import itertools
import unittest

import numpy as np
import pandas as pd

from family_dif_benchmark_audit.diagnostics import capability_profile as profile
from family_dif_benchmark_audit.diagnostics import discrimination as disc
from family_dif_benchmark_audit.diagnostics import within_family as within


class DiagnosticTests(unittest.TestCase):
    def test_corrected_item_total_matches_direct_leave_one_out(self):
        y = np.random.default_rng(81).binomial(1, .6, (17, 30)).astype(float)
        y[0] = 0
        actual = disc.corrected_item_total_correlation(y)
        expected = []
        for row in y:
            rest = y.sum(axis=0) - row
            expected.append(np.corrcoef(row, rest)[0, 1] if row.std() and rest.std() else 0)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_discrimination_strata_preserve_counts_and_blueprint_mass(self):
        cells = np.array(["a"] * 7 + ["b"] * 9)
        selected = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
        strata = disc.discrimination_cells(cells, np.arange(16) % 4, 4)
        for seed in range(10):
            w = disc.discrimination_matched_weights(cells, strata, selected, np.random.default_rng(seed))
            for cell in np.unique(cells):
                mask = cells == cell
                self.assertAlmostEqual(w[mask].sum(), mask.sum())
            for stratum in np.unique(strata):
                mask = strata == stratum
                self.assertEqual((w[mask] > 0).sum(), selected[mask].sum())

    def test_reliability_matches_direct_weighted_items(self):
        y = np.random.default_rng(2).binomial(1, .4, (8, 20)).astype(float)
        w = np.array([2, 0, 2, 0, 0, 2, 0, 2], dtype=float)
        contributions = w[w > 0, None] * y[w > 0]
        total = contributions.sum(axis=0)
        expected = 4 / 3 * (1 - contributions.var(axis=1, ddof=1).sum() / total.var(ddof=1))
        self.assertAlmostEqual(disc.weighted_cronbach_alpha(total, y.var(axis=1, ddof=1), w), expected)
        self.assertAlmostEqual(disc.independent_item_score_sem(y.var(axis=1, ddof=1), w),
                               np.sqrt(contributions.var(axis=1, ddof=1).sum()) / w.sum())

    def test_pair_counts_match_independent_enumeration_with_ties_and_owners(self):
        full = np.array([.50, .505, .508, .508, .515, .7])
        score = np.array([.6, .59, .59, .57, .50, .3])
        families = np.array(["a", "b", "b", "a", "a", "c"])
        owners = np.array(["x", "y", "z", "x", "w", "v"])
        actual = within.pairwise_metrics(full, score, families, owners)
        expected = profile.reference_counts(full, score, families, owners)
        for key, value in expected.items():
            self.assertEqual(actual[key], value)

    def test_specificity_uses_paired_gap_median(self):
        observed = pd.DataFrame({"close_cross_family_flips": [5], "close_cross_family_eligible_pairs": [10],
                                 "close_same_family_flips": [2], "close_same_family_eligible_pairs": [10]})
        controls = pd.DataFrame({"replicate": [0, 1, 2], "close_cross_family_flips": [1, 6, 7],
                                 "close_cross_family_eligible_pairs": [10] * 3,
                                 "close_same_family_flips": [0, 5, 1],
                                 "close_same_family_eligible_pairs": [10] * 3})
        summary = within.summarize_benchmark("test", observed, controls)
        self.assertAlmostEqual(summary["specificity_excess_points"], 20)
        self.assertEqual(summary["specificity_p"], .5)
        # The paired median .1 differs from the difference of medians .5.
        self.assertNotAlmostEqual(summary["specificity_excess_points"],
                                  summary["cross_family_excess_points"] - summary["same_family_excess_points"])

    def test_assignment_known_optimum(self):
        costs = np.array([[.04, 10.0], [8.0, .09]])
        a, b = profile.solve_assignment(costs, np.zeros_like(costs))
        self.assertEqual(list(zip(a, b)), [(0, 0), (1, 1)])

    def test_caliper_maximum_cardinality(self):
        costs = np.array([[.01, .04], [.03, 9.0]])
        a, b = profile.solve_assignment(costs, np.zeros_like(costs), .25)
        self.assertEqual(set(zip(a, b)), {(0, 1), (1, 0)})

    def test_caliper_requires_both_components(self):
        a, _ = profile.solve_assignment(np.array([[.01]]), np.array([[.5]]), .25)
        self.assertEqual(len(a), 0)
        a, _ = profile.solve_assignment(np.ones((2, 3)), np.zeros((2, 3)), .25)
        self.assertEqual(len(a), 0)

    def test_assignment_matches_exhaustive_reference(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            p, b = rng.random((3, 3)), rng.random((3, 3))
            a, cols = profile.solve_assignment(p, b, .7)
            candidates = []
            for n in range(4):
                for rows in itertools.combinations(range(3), n):
                    for columns in itertools.permutations(range(3), n):
                        if all(p[i, j] <= .49 and b[i, j] <= .49 for i, j in zip(rows, columns)):
                            candidates.append((n, sum(p[i, j] + b[i, j] for i, j in zip(rows, columns))))
            best = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
            self.assertEqual(len(a), best[0])
            self.assertAlmostEqual(float((p + b)[a, cols].sum()), best[1])

    def test_all_scenarios_no_replacement_cell_mass_and_unmatched_fixed(self):
        cells = np.array(["a", "a", "a", "b", "b", "b", "b"])
        selected = np.array([True, True, False, True, False, True, False])
        weights = np.array([1.5, 1.5, 0, 2, 0, 2, 0])
        pairings = profile.construct_pairs(cells, selected, np.arange(14).reshape(7, 2) / 20,
                                          np.zeros(7), 3)
        self.assertEqual(set(pairings), set(profile.SCENARIOS))
        for pairs in pairings.values():
            self.assertEqual(len(np.unique(pairs)), 2 * len(pairs))
            unmatched = np.setdiff1d(np.flatnonzero(selected), pairs[:, 0])
            for switches in itertools.product([0, 1], repeat=len(pairs)):
                w = profile.make_weights(weights, pairs, np.array([switches]))
                profile.assert_weights(w, weights, cells)
                np.testing.assert_array_equal(w[0, unmatched], weights[unmatched])

    def test_profile_distance_equals_unclipped_response_space_rms(self):
        rng = np.random.default_rng(5)
        axes, _ = np.linalg.qr(rng.normal(size=(12, 4)))
        load = rng.normal(size=(2, 4))
        scale = np.array([.25, .4])
        embedding = load / scale[:, None] / np.sqrt(12)
        offsets = load / scale[:, None] @ axes.T
        self.assertAlmostEqual(np.linalg.norm(embedding[0] - embedding[1]),
                               np.sqrt(np.mean((offsets[0] - offsets[1]) ** 2)))

    def test_all_command_parsers_accept_explicit_roots(self):
        for module in [disc, within, profile]:
            args = module.parse_args(["--benchmarks", "winogrande", "--primary-root", "primary",
                                      "--output-dir", "diagnostic"])
            self.assertEqual(args.primary_root.name, "primary")
            self.assertEqual(args.output_dir.name, "diagnostic")


if __name__ == "__main__":
    unittest.main()
