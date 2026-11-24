"""Behavioral checks for reproducible, strictly budgeted parameter search."""

import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.search import optimize


class SearchTests(unittest.TestCase):
    bounds = {"x": (-2.0, 3.0), "y": (-4.0, 1.0), "fixed": (7.0, 7.0)}

    @staticmethod
    def objective(params):
        return -(params["x"] - 0.25) ** 2 - (params["y"] + 0.75) ** 2

    def test_deterministic_for_both_methods(self):
        for method in ("ga", "random"):
            with self.subTest(method=method):
                first = optimize(self.objective, self.bounds, method=method, budget=97, seed=41)
                second = optimize(self.objective, self.bounds, method=method, budget=97, seed=41)
                self.assertEqual(first, second)

    def test_exact_unique_budget_and_history_accounting(self):
        for method in ("ga", "random"):
            for budget in (1, 7, 32, 33, 97, 256):
                with self.subTest(method=method, budget=budget):
                    actual_calls = []

                    def counted(params):
                        actual_calls.append(tuple(params.items()))
                        return self.objective(params)

                    result = optimize(counted, self.bounds, method=method, budget=budget, seed=7)
                    self.assertEqual(len(actual_calls), budget)
                    self.assertEqual(len(set(actual_calls)), budget)
                    self.assertEqual(result["evaluations"], budget)
                    self.assertEqual(result["requests"], len(result["history"]))
                    self.assertEqual(result["requests"], budget + result["cache_hits"])
                    self.assertEqual(result["cache_hits"], sum(row["cache_hit"] for row in result["history"]))
                    self.assertEqual(result["best_score"], max(row["score"] for row in result["history"]))

    def test_all_requested_parameters_remain_within_bounds(self):
        for method in ("ga", "random"):
            with self.subTest(method=method):
                result = optimize(self.objective, self.bounds, method=method, budget=150, seed=21)
                for row in result["history"]:
                    self.assertEqual(list(row["params"]), list(self.bounds))
                    for name, (lower, upper) in self.bounds.items():
                        self.assertGreaterEqual(row["params"][name], lower)
                        self.assertLessEqual(row["params"][name], upper)

    def test_ga_caches_repeated_candidates(self):
        result = optimize(lambda params: params["x"], {"x": (0.0, 1.0)}, budget=128, seed=0)
        self.assertGreater(result["cache_hits"], 0)
        seen = {}
        for row in result["history"]:
            key = tuple(row["params"].items())
            self.assertEqual(row["cache_hit"], key in seen)
            if key in seen:
                self.assertEqual(row["score"], seen[key])
            seen[key] = row["score"]

    def test_nonfinite_scores_become_finite_and_serializable(self):
        for method in ("ga", "random"):
            for invalid in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(method=method, invalid=invalid):
                    result = optimize(lambda params: invalid, {"x": (-1.0, 1.0)}, method=method, budget=40)
                    self.assertEqual(result["evaluations"], 40)
                    self.assertTrue(math.isfinite(result["best_score"]))
                    self.assertLess(result["best_score"], -1e300)
                    json.dumps(result, allow_nan=False)

    def test_invalid_scores_cannot_beat_valid_scores(self):
        result = optimize(lambda params: float("nan") if params["x"] < 0 else 0.0,
                          {"x": (-1.0, 1.0)}, method="random", budget=50, seed=4)
        self.assertEqual(result["best_score"], 0.0)
        self.assertGreaterEqual(result["best_params"]["x"], 0.0)

    def test_objective_cannot_mutate_recorded_parameters(self):
        def mutating(params):
            score = params["x"]
            params["x"] = 1000
            return score

        result = optimize(mutating, {"x": (0.0, 1.0)}, method="random", budget=4)
        self.assertTrue(all(0 <= row["params"]["x"] <= 1 for row in result["history"]))

    def test_zero_budget_and_fully_fixed_bounds(self):
        zero = optimize(lambda params: self.fail("objective called for zero budget"), {"x": (0, 1)}, budget=0)
        self.assertEqual(zero["evaluations"], 0)
        self.assertEqual(zero["history"], [])
        self.assertIsNone(zero["best_params"])
        for method in ("ga", "random"):
            with self.subTest(method=method):
                fixed = optimize(lambda params: params["x"], {"x": (2, 2)}, method=method, budget=8)
                self.assertEqual(fixed["evaluations"], 1)
                self.assertEqual(fixed["best_params"], {"x": 2.0})

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"method": "other"}, {"budget": -1}, {"bounds": {}},
                       {"bounds": {"x": (2, 1)}}, {"bounds": {"x": (0, float("inf"))}}):
            arguments = {"bounds": {"x": (0, 1)}}
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                optimize(lambda params: 0, **arguments)


if __name__ == "__main__":
    unittest.main()
