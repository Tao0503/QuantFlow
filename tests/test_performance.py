import json
import math
import unittest

import numpy as np
import pandas as pd

from demo.performance import extended_metrics, realized_trade_stats


def equity(returns, initial=100.):
    returns = np.asarray(returns, dtype=float)
    return pd.Series(initial * np.cumprod(1 + returns),
                     index=pd.date_range("2020-01-01", periods=len(returns), freq="B"))


class PerformanceTests(unittest.TestCase):
    def test_regression_alpha_beta_and_information_ratio_known_affine_returns(self):
        benchmark_returns = np.array([-.02, .01, -.01, .03, .005])
        strategy_returns = .0005 + 1.5 * benchmark_returns
        result = extended_metrics(equity(strategy_returns), equity(benchmark_returns), 100)
        active = strategy_returns - benchmark_returns
        self.assertAlmostEqual(result["beta"], 1.5, places=12)
        self.assertAlmostEqual(result["alpha_rf0"], .0005 * 252, places=12)
        self.assertAlmostEqual(result["tracking_error"], np.std(active, ddof=1) * math.sqrt(252), places=12)
        self.assertAlmostEqual(result["information_ratio"], np.mean(active) / np.std(active, ddof=1) * math.sqrt(252), places=12)
        self.assertEqual(result["benchmark_outperformance_days"], int((active > 1e-12).sum()))
        self.assertEqual(result["benchmark_outperformance_rate"], float((active > 1e-12).mean()))
        self.assertAlmostEqual(result["annual_excess_return"], np.prod(1 + strategy_returns) ** (252 / 5) - np.prod(1 + benchmark_returns) ** (252 / 5), places=10)

    def test_first_day_is_measured_against_initial_cash_and_downside_uses_all_days(self):
        returns = np.array([-.10, .10, 0., .02])
        reference = np.array([-.05, .05, 0., .01])
        result = extended_metrics(equity(returns), equity(reference), 100)
        downside = math.sqrt(np.mean(np.minimum(returns, 0.) ** 2) * 252)
        self.assertAlmostEqual(result["downside_volatility"], downside, places=12)
        self.assertAlmostEqual(result["sortino_rf0"], 252 * np.mean(returns) / downside, places=12)
        self.assertEqual(result["daily_win_rate"], .5)
        # The initial 10% loss must be in the drawdown denominator, even though
        # the first supplied close is already below the starting capital.
        cagr = (float(equity(returns).iloc[-1]) / 100) ** (252 / 4) - 1
        self.assertAlmostEqual(result["calmar"], cagr / .10, places=10)

    def test_identical_series_tracking_error_zero_ratios_undefined(self):
        values = equity([.01, -.02, .015, 0.])
        result = extended_metrics(values, values.copy(), 100)
        self.assertAlmostEqual(result["beta"], 1.)
        self.assertAlmostEqual(result["alpha_rf0"], 0.)
        self.assertEqual(result["tracking_error"], 0.)
        self.assertIsNone(result["information_ratio"])
        self.assertEqual(result["annual_excess_return"], 0.)
        self.assertEqual(result["benchmark_outperformance_days"], 0)
        json.dumps(result, allow_nan=False)

    def test_cash_and_no_negative_days_do_not_show_infinite_ratios(self):
        cash = equity([0.] * 4)
        result = extended_metrics(cash, cash.copy(), 100)
        for key in ("alpha_rf0", "beta", "information_ratio", "sortino_rf0", "calmar"):
            self.assertIsNone(result[key])
        self.assertEqual(result["downside_volatility"], 0.)
        self.assertEqual(result["daily_win_rate"], 0.)
        up = extended_metrics(equity([.01] * 4), cash, 100)
        self.assertIsNone(up["sortino_rf0"])
        self.assertIsNone(up["calmar"])
        self.assertEqual(up["daily_win_rate"], 1.)

    def test_one_session_has_no_sample_covariance_or_tracking_error(self):
        result = extended_metrics(equity([-.01]), equity([.005]), 100)
        for key in ("alpha_rf0", "beta", "information_ratio", "tracking_error"):
            self.assertIsNone(result[key])
        self.assertGreater(result["downside_volatility"], 0)
        self.assertLess(result["annual_excess_return"], 0)

    def test_daily_outperformance_uses_explicit_absolute_tolerance(self):
        result = extended_metrics(equity([0., 5e-13, 2e-12]), equity([0., 0., 0.]), 100)
        self.assertEqual(result["benchmark_outperformance_days"], 1)

    def test_rejects_unaligned_invalid_and_ambiguous_inputs_without_mutation(self):
        values = equity([.01, -.01, .02])
        original = values.copy(deep=True)
        for bad in (values.iloc[1:], values.iloc[::-1], values.set_axis([values.index[0]] * 3),
                    pd.Series([1, np.nan, 1], index=values.index), pd.Series([1, 0, 1], index=values.index),
                    pd.Series([True] * 3, index=values.index)):
            with self.assertRaises(ValueError):
                extended_metrics(values, bad, 100)
        for capital in (0, -1, True, np.nan, np.inf, "bad"):
            with self.assertRaises(ValueError):
                extended_metrics(values, values, capital)
        pd.testing.assert_series_equal(values, original)

    def test_fifo_partial_lots_and_both_commissions_count_each_sell_fill(self):
        trades = pd.DataFrame([
            {"quantity": 10, "price": 10., "commission": 1.},
            {"quantity": 5, "price": 12., "commission": 1.},
            {"quantity": -12, "price": 11., "commission": 1.2},
            {"quantity": -3, "price": 10., "commission": .3},
            {"quantity": 50, "price": 2., "commission": 10.},  # Open lot excluded.
        ])
        result = realized_trade_stats(trades)
        self.assertEqual(result["closed_fill_count"], 2)
        self.assertEqual(result["closed_fill_win_rate"], .5)
        self.assertAlmostEqual(result["realized_gains"], 5.4)
        self.assertAlmostEqual(result["realized_losses"], 6.9)
        self.assertAlmostEqual(result["realized_net_pnl"], -1.5)
        self.assertAlmostEqual(result["realized_profit_factor"], 5.4 / 6.9)
        json.dumps(result, allow_nan=False)

    def test_fifo_symbols_are_independent_and_no_closed_fills_is_undefined(self):
        none = realized_trade_stats(pd.DataFrame())
        self.assertEqual(none["closed_fill_count"], 0)
        self.assertIsNone(none["closed_fill_win_rate"])
        self.assertIsNone(none["realized_profit_factor"])
        trades = pd.DataFrame([
            {"symbol": "A", "quantity": 2, "price": 10., "commission": 0.},
            {"symbol": "B", "quantity": 1, "price": 100., "commission": 0.},
            {"symbol": "B", "quantity": -1, "price": 110., "commission": 0.},
        ])
        result = realized_trade_stats(trades)
        self.assertEqual(result["realized_net_pnl"], 10.)
        self.assertEqual(result["closed_fill_win_rate"], 1.)
        self.assertIsNone(result["realized_profit_factor"])

    def test_fifo_rejects_unfunded_sells_fractional_shares_and_date_reordering(self):
        for rows in ([{"quantity": -1, "price": 10., "commission": 0.}],
                     [{"quantity": 1.5, "price": 10., "commission": 0.}],
                     [{"quantity": 1, "price": 10., "commission": -1.}],
                     [{"date": "2020-01-03", "quantity": 1, "price": 10., "commission": 0.},
                      {"date": "2020-01-02", "quantity": -1, "price": 10., "commission": 0.}]):
            with self.assertRaises(ValueError):
                realized_trade_stats(pd.DataFrame(rows))


if __name__ == "__main__":
    unittest.main()
