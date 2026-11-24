import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from demo.portfolio import backtest_portfolio
from quantlab.engine import Costs, features


def frame(closes, opens=None, dividends=None, start="2020-01-01"):
    close = np.asarray(closes, float)
    opening = close if opens is None else np.asarray(opens, float)
    div = np.zeros(len(close)) if dividends is None else np.asarray(dividends, float)
    tr = close[0] * np.cumprod(np.r_[1., (close[1:] + div[1:]) / close[:-1]])
    return pd.DataFrame({"open": opening, "close": close, "dividend": div, "tr_close": tr,
                         "split_ratio": 1.}, index=pd.date_range(start, periods=len(close), freq="B", name="date"))


def ready_features(data):
    return pd.DataFrame({"deviation": 0., "momentum": 0.}, index=data.index)


class PortfolioEngineTests(unittest.TestCase):
    def run_half(self, frames, weights=None, cash=1000, costs=Costs(0, 0, 0)):
        first = max(f.index[0] for f in frames.values())
        last = min(f.index[-1] for f in frames.values())
        with patch("demo.portfolio.features", ready_features):
            return backtest_portfolio(frames, weights or {s: 1 / len(frames) for s in frames}, "half", {},
                                      str(first.date()), str(last.date()), costs, cash)

    def test_one_shared_cash_and_sleeve_targets(self):
        data = {"B": frame([50] * 4), "A": frame([100] * 4)}
        result = self.run_half(data, {"A": .6, "B": .4})
        first = result.daily.iloc[1]
        self.assertEqual(first.holdings["A"]["shares"], 3)
        self.assertEqual(first.holdings["B"]["shares"], 4)
        self.assertEqual(first.cash, 500)
        self.assertEqual(first.equity, 1000)
        self.assertEqual(first.exposure, .5)
        self.assertTrue((result.trades.date > result.trades.signal_date).all())
        for _, row in result.daily.iterrows():
            self.assertAlmostEqual(row.equity, row.cash + row.receivable + sum(h["market_value"] for h in row.holdings.values()))

    def test_entire_open_nav_is_frozen_before_costly_sells(self):
        data = {"A": frame([100, 100, 400, 400], opens=[100, 100, 400, 400]),
                "B": frame([100, 100, 100, 100])}
        result = self.run_half(data, cash=4000, costs=Costs(100, 0, 0))
        # Initial A/B holdings are ten each. After A rises, close NAV=6980,
        # both desired sleeves use 6980 even though selling A later costs cash.
        last_day = result.daily.index[-1]
        trades = result.trades[result.trades.date == last_day]
        self.assertEqual(trades.symbol.tolist(), ["A", "B"])
        self.assertEqual(trades.quantity.tolist(), [-6, 7])
        self.assertEqual(result.daily.iloc[-1].holdings["B"]["shares"], 17)

    def test_simultaneous_cash_pressure_is_proportional_and_order_independent(self):
        data = {"B": frame([10] * 3), "A": frame([10] * 3)}
        high = pd.DataFrame({"deviation": -60., "momentum": 0.}, index=data["A"].index)
        params = {"mean_reversion": 1., "momentum": 0., "band": 0.}
        with patch("demo.portfolio.features", lambda d: high):
            first = backtest_portfolio(data, {"B": .5, "A": .5}, "sigmoid", params,
                                       "2020-01-01", "2020-01-03", Costs(1000, 0, 0), 100)
            second = backtest_portfolio(dict(reversed(list(data.items()))), {"A": .5, "B": .5}, "sigmoid", params,
                                        "2020-01-01", "2020-01-03", Costs(1000, 0, 0), 100)
        pd.testing.assert_frame_equal(first.daily, second.daily)
        pd.testing.assert_frame_equal(first.trades, second.trades)
        holdings = first.daily.iloc[1].holdings
        self.assertEqual((holdings["A"]["shares"], holdings["B"]["shares"]), (5, 4))
        self.assertAlmostEqual(first.daily.iloc[1].cash, 1)
        self.assertTrue((first.daily.cash >= 0).all())

    def test_dividend_is_based_on_previous_inventory_and_proxy_lag(self):
        data = {"A": frame([100, 100, 99, 99, 99], dividends=[0, 0, 1, 0, 0]),
                "B": frame([50] * 5, dividends=[0, 2, 0, 0, 0])}
        result = self.run_half(data, {"A": .6, "B": .4}, costs=Costs(0, 0, 2))
        self.assertEqual(result.daily.iloc[1].dividend_entitlement, 0)  # B bought on ex-date.
        self.assertEqual(result.daily.iloc[2].dividend_entitlement, 3)
        self.assertEqual(result.daily.iloc[2].receivable, 3)
        self.assertEqual(result.daily.iloc[2].equity, 1000)
        self.assertEqual(result.daily.iloc[-1].receivable, 0)
        self.assertEqual(result.daily.iloc[3].dividend_paid_proxy, 3)
        self.assertEqual(result.daily.dividend_paid_proxy.sum(), 3)

    def test_any_sleeve_trigger_rebalances_the_complete_vector(self):
        data = {"A": frame([100, 100, 200, 200]), "B": frame([100] * 4)}
        result = self.run_half(data, cash=4000)
        final = result.trades[result.trades.date == data["A"].index[-1]]
        self.assertEqual(final.symbol.tolist(), ["A", "B"])
        self.assertEqual(final.quantity.tolist(), [-4, 2])
        # Only A crosses its scaled band in the unequal-allocation case below;
        # B must still receive the same portfolio-wide rebalancing event.
        data = {"A": frame([100, 100, 130, 130]), "B": frame([100] * 4)}
        result = self.run_half(data, {"A": .2, "B": .8}, cash=10000)
        final = result.trades[result.trades.date == data["A"].index[-1]]
        self.assertIn("B", final.symbol.tolist())
        self.assertEqual(final[final.symbol == "B"].quantity.iloc[0], 1)

    def test_calendar_gaps_reject_but_prelisting_feature_history_is_kept(self):
        a = frame(np.linspace(100, 120, 110))
        b = a.iloc[40:].copy()
        called_lengths = []

        def inspect(data):
            called_lengths.append(len(data))
            return features(data)

        with patch("demo.portfolio.features", inspect):
            result = backtest_portfolio({"A": a, "B": b}, {"A": .5, "B": .5}, "half", {},
                                        str(b.index[0].date()), str(b.index[-1].date()), Costs(0, 0, 0), 10000)
        self.assertEqual(called_lengths, [110, 70])
        self.assertEqual(result.trades.date.min(), b.index[64])
        with self.assertRaisesRegex(ValueError, "calendar"):
            self.run_half({"A": a, "B": b.drop(b.index[10])})

    def test_causality_and_costs_from_real_features(self):
        rng = np.random.default_rng(7)
        data = {s: frame(100 * np.exp(np.cumsum(rng.normal(.001, .02, 180)))) for s in ("A", "B")}
        params = {"mean_reversion": 1., "momentum": 1., "band": .03}
        first = backtest_portfolio(data, {"A": .7, "B": .3}, "sigmoid", params, "2020-01-01", "2021-01-01", Costs(10, 20, 35), 100000)
        changed = {s: d.copy() for s, d in data.items()}
        changed["A"].iloc[140:, changed["A"].columns.get_indexer(["open", "close", "tr_close"]) ] *= 2
        second = backtest_portfolio(changed, {"A": .7, "B": .3}, "sigmoid", params, "2020-01-01", "2021-01-01", Costs(10, 20, 35), 100000)
        pd.testing.assert_frame_equal(first.daily.iloc[:140], second.daily.iloc[:140])
        cutoff = data["A"].index[140]
        pd.testing.assert_frame_equal(first.trades[first.trades.date < cutoff].reset_index(drop=True),
                                     second.trades[second.trades.date < cutoff].reset_index(drop=True))
        for _, row in first.trades.iterrows():
            self.assertGreater(row.date, row.signal_date)
            self.assertGreaterEqual(row.price, row.open_price) if row.quantity > 0 else self.assertLessEqual(row.price, row.open_price)

    def test_invalid_allocations_prices_and_params(self):
        data = {"A": frame([100] * 70), "B": frame([100] * 70)}
        for weights in ({"A": .8, "B": .3}, {"A": -.1, "B": .5}, {"A": float("nan"), "B": .5}, {"A": .5}, {"A": 0, "B": 0}):
            with self.assertRaises(ValueError):
                backtest_portfolio(data, weights, "half", {}, "2020-01-01", "2021-01-01")
        with self.assertRaises(ValueError):
            backtest_portfolio(data, {"A": .5, "B": .5}, "sigmoid", {"band": .1}, "2020-01-01", "2021-01-01")
        with self.assertRaises(ValueError):
            backtest_portfolio(data, {"A": .5, "B": .5}, "half", {}, "2021-01-01", "2020-01-01")


if __name__ == "__main__":
    unittest.main()
