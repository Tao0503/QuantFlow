import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from demo import regime_research as study
from quantlab.engine import Costs, step


def frame(size=280):
    rng = np.random.default_rng(19)
    price = 100 * np.exp(np.cumsum(rng.normal(.0008, .012, size)))
    return pd.DataFrame({"open": price, "close": price, "tr_close": price,
                         "dividend": 0.}, index=pd.date_range("2018-01-01", periods=size, freq="B", name="date"))


class RegimeStudyTests(unittest.TestCase):
    def test_feature_causality_warmup_and_exact_target_adapter(self):
        prices = frame()
        encoded, signals = study.regime_features(prices)
        self.assertTrue(signals.target_weight.iloc[:199].isna().all())
        altered = prices.copy()
        altered.iloc[240:, altered.columns.get_indexer(["open", "close", "tr_close"])] *= 3
        future_encoded, future_signals = study.regime_features(altered)
        pd.testing.assert_frame_equal(encoded.iloc[:240], future_encoded.iloc[:240])
        pd.testing.assert_frame_equal(signals.iloc[:240], future_signals.iloc[:240])
        for date in signals.index[signals.ready]:
            target = step("sigmoid", encoded.loc[date, "deviation"], 0, .5,
                          {"mean_reversion": 1., "momentum": 0., "band": 0.}).target_weight
            self.assertAlmostEqual(target, signals.loc[date, "target_weight"], places=12)

    def test_next_open_costs_inventory_and_ledger_reconciliation(self):
        prices = frame(350)
        result = study.backtest_regime(prices, "2018-01-01", "2020-01-01", Costs(10, 20, 35), 100000)
        self.assertGreater(len(result.trades), 0)
        self.assertTrue((result.trades.date > result.trades.signal_date).all())
        self.assertTrue((result.daily.cash >= 0).all())
        np.testing.assert_allclose(result.daily.equity, result.daily.cash + result.daily.shares * result.daily.close + result.daily.receivable)
        self.assertGreater(result.metrics["total_commission"], 0)
        self.assertGreater(result.metrics["slippage_cost"], 0)

    def test_paired_bootstrap_identical_series_is_exact_zero_and_deterministic(self):
        returns = np.random.default_rng(5).normal(.0002, .01, 120)
        same = study.paired_block_bootstrap(returns, returns, block_length=20, replications=200, seed=4)
        self.assertEqual(same["ci_lower"], 0.)
        self.assertEqual(same["ci_upper"], 0.)
        a = study.paired_block_bootstrap(returns + .001, returns, block_length=20, replications=200, seed=4)
        b = study.paired_block_bootstrap(returns + .001, returns, block_length=20, replications=200, seed=4)
        self.assertEqual(a, b)
        self.assertGreater(a["ci_lower"], 0)

    def test_bootstrap_rejects_unpaired_nonfinite_and_invalid_blocks(self):
        for a, b, block in (([1] * 30, [1] * 29, 20), ([1] * 29 + [np.nan], [1] * 30, 20),
                            ([1] * 10, [1] * 10, 20), ([1] * 30, [1] * 30, 1)):
            with self.assertRaises(ValueError):
                study.paired_block_bootstrap(a, b, block_length=block, replications=200)

    def test_freeze_never_computes_performance_and_refuses_changed_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "study"
            with patch.object(study, "_snapshot_hashes", return_value={"data": "original"}), \
                 patch.object(study, "backtest") as simulator, patch.object(study, "load_data") as loader:
                protocol = study.freeze(root, out)
                self.assertFalse(protocol["performance_computed_during_freeze"])
                simulator.assert_not_called()
                loader.assert_not_called()
                with self.assertRaises(FileExistsError):
                    study.freeze(root, out)
            with patch.object(study, "_snapshot_hashes", return_value={"data": "changed"}), patch.object(study, "backtest") as simulator:
                with self.assertRaisesRegex(ValueError, "changed"):
                    study.evaluate(root, out)
                simulator.assert_not_called()
            self.assertFalse((out / "evaluation_started.json").exists())

    def test_started_marker_prevents_repeated_holdout_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp), Path(temp) / "study"
            with patch.object(study, "_snapshot_hashes", return_value={}):
                study.freeze(root, out)
                (out / "evaluation_started.json").write_text(json.dumps({"already": True}))
                with patch.object(study, "load_data") as loader:
                    with self.assertRaisesRegex(FileExistsError, "already started"):
                        study.evaluate(root, out)
                    loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
