import unittest

import numpy as np
import pandas as pd

from quantlab.engine import Costs, backtest, features, step


def frame(closes, opens=None, dividends=None, start="2020-01-01"):
    c = np.asarray(closes, dtype=float)
    o = c.copy() if opens is None else np.asarray(opens, dtype=float)
    d = np.zeros(len(c)) if dividends is None else np.asarray(dividends, dtype=float)
    tr = c[0] * np.cumprod(np.r_[1.0, (c[1:] + d[1:]) / c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
        "close": c, "dividend": d, "volume": 10000, "split_ratio": 1., "tr_close": tr},
        index=pd.date_range(start, periods=len(c), freq="B", name="date"))


class EngineTests(unittest.TestCase):
    def test_sigmoid_uses_whole_signal_and_band(self):
        p = {"mean_reversion": 1., "momentum": 1., "band": .01}
        self.assertLess(step("sigmoid", 1, 0, .5, p).target_weight, .5)
        self.assertGreater(step("sigmoid", 0, 1, .5, p).target_weight, .5)
        self.assertIsNone(step("sigmoid", 0, 0, .5, p).target_weight)
        self.assertIsNone(step("sigmoid", float("nan"), 0, .5, p).target_weight)

    def test_next_session_open_and_initial_signal(self):
        d = frame([100, 200, 300], opens=[100, 150, 280])
        r = backtest(d, "buy_hold", {}, str(d.index[1].date()), str(d.index[-1].date()),
                     Costs(0, 0, 0), initial_cash=1500)
        self.assertEqual(r.trades.iloc[0].price, 150)
        self.assertEqual(r.trades.iloc[0].signal_date, d.index[0])
        self.assertEqual(r.trades.iloc[0].date, d.index[1])
        self.assertEqual(r.daily.iloc[-1].equity, 3000)

    def test_first_bar_has_no_same_close_fill(self):
        d = frame([100, 200, 300])
        r = backtest(d, "buy_hold", {}, "2020-01-01", "2020-01-03", Costs(0, 0, 0), 1000)
        self.assertEqual(r.daily.iloc[0].shares, 0)
        self.assertEqual(r.trades.iloc[0].date, d.index[1])
        self.assertTrue((r.trades.date > r.trades.signal_date).all())

    def test_dividend_entitlement_preserves_nav(self):
        d = frame([100, 100, 99, 99], dividends=[0, 0, 1, 0])
        r = backtest(d, "buy_hold", {}, "2020-01-01", "2020-01-06", Costs(0, 0, 35), 1000)
        self.assertEqual(r.daily.iloc[2].shares, 10)
        self.assertEqual(r.daily.iloc[2].receivable, 10)
        self.assertEqual(r.daily.iloc[2].equity, 1000)
        self.assertEqual(r.daily.iloc[2].cash, 0)

    def test_exdate_new_buyer_has_no_dividend(self):
        d = frame([100, 99, 99], dividends=[0, 1, 0])
        r = backtest(d, "buy_hold", {}, "2020-01-01", "2020-01-03", Costs(0, 0, 35), 1000)
        self.assertEqual(r.metrics["dividend_entitlement"], 0)
        self.assertEqual(r.daily.iloc[-1].receivable, 0)

    def test_dividend_cash_is_unspendable_until_proxy_date(self):
        d = frame([100] * 45, dividends=[0, 0, 1] + [0] * 42)
        r = backtest(d, "buy_hold", {}, "2020-01-01", "2020-04-01", Costs(0, 0, 35), 1000)
        ex = d.index[2]
        before = r.daily.loc[r.daily.index < ex + pd.Timedelta(days=35)]
        after = r.daily.loc[r.daily.index >= ex + pd.Timedelta(days=35)]
        self.assertEqual(before.cash.max(), 1000)  # Initial session before first order.
        self.assertEqual(before.iloc[-1].cash, 0)
        self.assertEqual(after.iloc[0].cash, 10)
        self.assertEqual(after.iloc[0].receivable, 0)

    def test_costs_cash_shares_and_equity_reconcile(self):
        d = frame([100, 100, 100])
        r = backtest(d, "buy_hold", {}, "2020-01-01", "2020-01-03", Costs(100, 100, 35), 1000)
        self.assertTrue((r.daily.cash >= 0).all())
        self.assertTrue((r.daily.shares >= 0).all())
        self.assertEqual(r.daily.iloc[-1].shares, 9)
        np.testing.assert_allclose(r.daily.equity,
            r.daily.cash + r.daily.shares * r.daily.close + r.daily.receivable)
        self.assertLess(r.metrics["final_equity"], 1000)
        t = r.trades.iloc[0]
        self.assertAlmostEqual(1000 - r.metrics["final_equity"], t.commission + t.slippage_cost)

    def test_sell_and_no_negative_inventory(self):
        d = frame([100] * 5)
        f = pd.DataFrame({"deviation": [0] * 5, "momentum": [1, 1, -1, -1, -1]}, index=d.index)
        r = backtest(d, "momentum", {}, "2020-01-01", "2020-01-07", Costs(0, 0, 0), 1000, f)
        self.assertEqual(r.trades.quantity.tolist(), [10, -10])
        self.assertEqual(r.daily.iloc[-1].shares, 0)
        self.assertEqual(r.daily.iloc[-1].cash, 1000)

    def test_future_changes_do_not_change_past_features_or_fills(self):
        d = frame(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(.001, .01, 160))))
        altered = d.copy()
        altered.iloc[120:, altered.columns.get_loc("tr_close")] *= 10
        altered.iloc[120:, altered.columns.get_loc("open")] *= 10
        altered.iloc[120:, altered.columns.get_loc("close")] *= 10
        altered.iloc[120:, altered.columns.get_loc("dividend")] = 25
        pd.testing.assert_frame_equal(features(d).iloc[:120], features(altered).iloc[:120])
        p = {"mean_reversion": 1., "momentum": 1., "band": .05}
        a = backtest(d, "sigmoid", p, "2020-01-01", "2021-01-01")
        b = backtest(altered, "sigmoid", p, "2020-01-01", "2021-01-01")
        pd.testing.assert_frame_equal(a.daily.iloc[:120], b.daily.iloc[:120])
        pd.testing.assert_frame_equal(a.trades[a.trades.date < d.index[120]].reset_index(drop=True),
                                     b.trades[b.trades.date < d.index[120]].reset_index(drop=True))

    def test_deterministic_replay_and_fresh_interval_cash(self):
        d = frame([100, 110, 90, 120, 100])
        a = backtest(d, "half", {}, "2020-01-02", "2020-01-07")
        b = backtest(d, "half", {}, "2020-01-02", "2020-01-07")
        self.assertEqual(a.metrics, b.metrics)
        pd.testing.assert_frame_equal(a.daily, b.daily)
        self.assertEqual(a.metrics["initial_cash"], 100000)
        self.assertTrue((a.trades.date > a.trades.signal_date).all())

    def test_invalid_costs(self):
        for args in [(-1, 0, 0), (0, -1, 0), (0, 10000, 0), (0, 0, -1), (float("nan"), 0, 0)]:
            with self.assertRaises(ValueError):
                Costs(*args)

    def test_slippage_always_adverse_including_opening_gap(self):
        d = frame([1, 2, 1, 3, .5], opens=[1, 1, 1.00050050025, 1, 1])
        r = backtest(d, "half", {}, "2020-01-01", "2020-01-07", Costs(0, 5, 0), 1000000)
        buys = r.trades[r.trades.quantity > 0]
        sells = r.trades[r.trades.quantity < 0]
        self.assertTrue((buys.price >= buys.open_price).all())
        self.assertTrue((sells.price <= sells.open_price).all())


if __name__ == "__main__":
    unittest.main()
