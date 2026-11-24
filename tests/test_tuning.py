from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from demo import tuning


def prices(end):
    index = pd.date_range("2009-01-01", end, freq="B", name="date")
    close = 100 + np.arange(len(index)) * .02
    return pd.DataFrame({"open": close, "close": close, "tr_close": close, "dividend": 0.}, index=index)


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.train = prices("2018-12-31")
        self.validation = prices("2021-12-31")
        self.protocol = {"train": ["2010-01-01", "2018-12-31"], "validation": ["2019-01-01", "2021-12-31"],
            "training_folds": [["2010-01-01", "2012-12-31"], ["2013-01-01", "2015-12-31"], ["2016-01-01", "2018-12-31"]],
            "costs": {"commission_bps": 1, "slippage_bps": 5, "dividend_cash_lag_days": 35}, "initial_cash": 100000,
            "bounds": {"mean_reversion": [0, 2], "momentum": [0, 2], "band": [.02, .2]},
            "seeds": [47, 29, 11], "budget_per_run": 2}
        self.incumbent = {"mean_reversion": .6, "momentum": .3, "band": .2}

    def test_delta_objective_equal_budget_and_validation_after_all_searches(self):
        searches, training_half_calls, validation_calls, objectives, updates = [], [], [], [], []
        half_scores = {"2010-01-01": .3, "2013-01-01": .6, "2016-01-01": .9}

        def fake_backtest(frame, kind, params, start, end, costs, initial_cash, feat):
            self.assertTrue(frame.index.equals(feat.index))
            if start < "2019-01-01":
                self.assertLessEqual(frame.index.max(), pd.Timestamp("2018-12-31"))
                if kind == "half":
                    training_half_calls.append(start)
                    score = half_scores[start]
                else:
                    score = half_scores[start] + params["mean_reversion"]
            else:
                self.assertEqual(len(searches), 6, "Validation ran before all searches completed")
                self.assertLessEqual(frame.index.max(), pd.Timestamp("2021-12-31"))
                validation_calls.append((kind, params.copy()))
                score = .5 if kind == "half" else 1.
            closes = frame.loc[start:end, "close"]
            daily = pd.DataFrame({"equity": closes / closes.iloc[0] * initial_cash})
            return SimpleNamespace(metrics={"sharpe_rf0": score, "max_drawdown": .1, "annual_turnover": .2},
                                   daily=daily, trades=pd.DataFrame())

        def fake_search(objective, bounds, *, method, budget, seed):
            searches.append((method, seed, budget))
            params = {"mean_reversion": .4, "momentum": 1., "band": .12}
            objectives.extend(objective(params.copy()) for _ in range(budget))
            return {"best_params": params, "best_score": objectives[-1], "evaluations": budget,
                    "requests": budget + 1, "cache_hits": 1}

        with patch.object(tuning, "backtest", fake_backtest), patch.object(tuning, "optimize", fake_search):
            result = tuning.run_tuning(self.train, self.validation, self.protocol, self.incumbent,
                                       lambda *args: updates.append(args))
        np.testing.assert_allclose(objectives, .4)
        self.assertEqual(len(training_half_calls), 3)
        self.assertEqual(len(validation_calls), 10)
        self.assertEqual((result["winner"]["method"], result["winner"]["seed"]), ("ga", 11))
        self.assertAlmostEqual(result["winner"]["validation_delta"], .5)
        self.assertEqual(result["evaluations"], 12)
        self.assertEqual(result["requests"], 18)
        self.assertEqual(result["cache_hits"], 6)
        self.assertEqual(updates[-1][:2], (22, 22))
        self.assertEqual([x[0] for x in updates], list(range(23)))
        self.assertEqual(result["benchmark_params"]["same_band_half"], {"mean_reversion": 0., "momentum": 0., "band": .12})

    def test_callback_cancellation_propagates(self):
        class StopRequested(Exception):
            pass

        def cancel(*args):
            raise StopRequested()

        with patch.object(tuning, "backtest") as simulator:
            with self.assertRaises(StopRequested):
                tuning.run_tuning(self.train, self.validation, self.protocol, self.incumbent, cancel)
            simulator.assert_not_called()

    def test_rejects_future_data_and_overlapping_periods_before_computation(self):
        bad_protocol = deepcopy(self.protocol)
        bad_protocol["validation"][0] = "2018-01-01"
        cases = [(prices("2019-01-01"), self.validation, self.protocol),
                 (self.train, prices("2022-01-03"), self.protocol),
                 (self.train, self.validation, bad_protocol)]
        for train, validation, protocol in cases:
            with self.subTest(protocol=protocol), patch.object(tuning, "backtest") as simulator:
                with self.assertRaises(ValueError):
                    tuning.run_tuning(train, validation, protocol, self.incumbent, lambda *args: None)
                simulator.assert_not_called()

    def test_real_small_search_is_deterministic_and_does_not_mutate_inputs(self):
        # Synthetic historical data only: no original test outcomes are read.
        before_train, before_validation = self.train.copy(deep=True), self.validation.copy(deep=True)
        first = tuning.run_tuning(self.train, self.validation, self.protocol, self.incumbent, lambda *args: None)
        second = tuning.run_tuning(self.train, self.validation, self.protocol, self.incumbent, lambda *args: None)
        self.assertEqual(first, second)
        self.assertEqual(len(first["finalists"]), 6)
        self.assertEqual(first["evaluations"], 12)
        self.assertFalse(first["test_used"])
        pd.testing.assert_frame_equal(self.train, before_train)
        pd.testing.assert_frame_equal(self.validation, before_validation)

    def test_under_budget_search_is_rejected_before_validation(self):
        def incomplete(objective, bounds, **kwargs):
            return {"best_params": self.incumbent, "best_score": 0, "evaluations": 1, "requests": 1, "cache_hits": 0}

        with patch.object(tuning, "optimize", incomplete), patch.object(tuning, "backtest", return_value=SimpleNamespace(metrics={"sharpe_rf0": .5})) as simulator:
            with self.assertRaisesRegex(RuntimeError, "exact"):
                tuning.run_tuning(self.train, self.validation, self.protocol, self.incumbent, lambda *args: None)
            self.assertEqual(simulator.call_count, 3)


if __name__ == "__main__":
    unittest.main()
