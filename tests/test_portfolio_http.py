"""Portfolio HTTP regressions with real cached data and isolated persistent state."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
import unittest

import pandas as pd

from demo.service import DemoService
from quantlab.data import load_data
from tests import test_demo_http as legacy_http


ROOT = Path(__file__).resolve().parents[1]


class PortfolioHTTPTests(unittest.TestCase):
    # Reuse the isolated server fixture, without inheriting the legacy test cases.
    setUp = legacy_http.DemoHTTPTests.setUp
    stop_server = legacy_http.DemoHTTPTests.stop_server
    request = legacy_http.DemoHTTPTests.request
    api = legacy_http.DemoHTTPTests.api
    wait_terminal = legacy_http.DemoHTTPTests.wait_terminal

    def submit(self, *, strategy="half", **overrides):
        body = {
            "allocations": {"SPY": .6, "GLD": .4}, "strategy": strategy,
            "start": "2020-01-02", "end": "2020-06-30", "initial_cash": 100000,
            "params": {"mean_reversion": 1., "momentum": 1., "band": .05},
            "costs": {"commission_bps": 1., "slippage_bps": 5.},
        }
        body.update(overrides)
        job = self.api("POST", "/api/backtests", body, expected=202)
        self.assertIn(job["status"], ("queued", "running", "completed"))
        result = self.wait_terminal("/api/backtests/" + job["id"])
        self.assertEqual(result["status"], "completed", result.get("error"))
        return body, result

    def test_invalid_allocations_are_rejected_before_creating_jobs(self):
        before = self.api("GET", "/api/backtests")["items"]
        invalid = [
            None, [], {}, {"SPY": 1.}, {"SPY": .5, "BTC": .5},
            {"SPY": .4, "GLD": .4}, {"SPY": -.1, "GLD": 1.1},
            {"SPY": 0., "GLD": 1.}, {"SPY": True, "GLD": 0.},
            {"SPY": float("nan"), "GLD": .5},
            {"SPY": float("inf"), "GLD": .5},
        ]
        for allocations in invalid:
            with self.subTest(allocations=allocations):
                self.api("POST", "/api/backtests", {
                    "allocations": allocations, "strategy": "half",
                    "start": "2020-01-02", "end": "2020-06-30",
                }, expected=400)
        for extra in ({"instance_id": "inst-spy"}, {"strategy": "momentum"}):
            with self.subTest(extra=extra):
                self.api("POST", "/api/backtests", {
                    "allocations": {"SPY": .5, "GLD": .5}, "strategy": "half", **extra,
                }, expected=400)
        self.assertEqual(self.api("GET", "/api/backtests")["items"], before)

    def test_real_shared_cash_ledger_and_prior_session_fills_for_both_strategies(self):
        frames = {symbol: load_data(ROOT / f"data/{symbol}.csv") for symbol in ("SPY", "GLD")}
        common = frames["SPY"].index.intersection(frames["GLD"].index).sort_values()
        for strategy in ("half", "sigmoid"):
            with self.subTest(strategy=strategy):
                body, result = self.submit(strategy=strategy)
                self.assertEqual(result["allocations"], body["allocations"])
                self.assertEqual(result["strategy"], strategy)
                self.assertIsInstance(result["symbol"], str)
                self.assertIn("SPY", result["symbol"])
                self.assertIn("GLD", result["symbol"])
                self.assertTrue(result["observed_history"])
                self.assertEqual(result["mode"], "historical_replay")
                self.assertEqual(set(result["baseline_metrics"]), {"half", "sigmoid"})
                dates = common[(common >= body["start"]) & (common <= body["end"])]
                chart = {pd.Timestamp(row["date"]): row for row in result["chart"]}
                self.assertEqual(set(chart), set(dates))  # The interval fits under the sampling cap.
                self.assertEqual(result["metrics"]["sessions"], len(dates))
                self.assertEqual(result["chart_total_sessions"], len(dates))
                self.assertEqual(result["trade_count"], len(result["trades"]))
                self.assertEqual(result["trades"], result["fills"])
                self.assertEqual({t["symbol"] for t in result["trades"]}, {"SPY", "GLD"})

                # Rebuild accounting independently from the public trades and raw prices.
                cash = float(body["initial_cash"])
                shares = {"SPY": 0, "GLD": 0}
                receivables = []
                entitlement_total = commission_total = slippage_total = 0.
                by_date = {}
                for trade in result["trades"]:
                    by_date.setdefault(pd.Timestamp(trade["date"]), []).append(trade)
                self.assertTrue(set(by_date).issubset(set(dates)))
                for date in dates:
                    due = sum(amount for payment, amount in receivables if payment <= date)
                    cash += due
                    receivables = [(payment, amount) for payment, amount in receivables if payment > date]
                    dividend = sum(shares[symbol] * frames[symbol].at[date, "dividend"] for symbol in shares)
                    entitlement_total += dividend
                    if dividend:
                        receivables.append((date + pd.Timedelta(days=35), dividend))
                    previous = common[common.get_loc(date) - 1]
                    for trade in by_date.get(date, []):
                        symbol = trade["symbol"]
                        quantity = trade["quantity"]
                        self.assertEqual(quantity, int(quantity))
                        self.assertNotEqual(quantity, 0)
                        self.assertEqual(pd.Timestamp(trade["signal_date"]), previous)
                        source_open = frames[symbol].at[date, "open"]
                        self.assertAlmostEqual(trade["open_price"], source_open, places=9)
                        adverse_fill = source_open * (1 + (.0005 if quantity > 0 else -.0005))
                        self.assertAlmostEqual(trade["price"], adverse_fill, places=9)
                        self.assertAlmostEqual(trade["notional"], quantity * trade["price"], places=7)
                        self.assertAlmostEqual(trade["commission"], abs(trade["notional"]) * .0001, places=9)
                        self.assertAlmostEqual(trade["slippage_cost"], abs(quantity) * source_open * .0005, places=9)
                        cash -= trade["notional"] + trade["commission"]
                        shares[symbol] += quantity
                        commission_total += trade["commission"]
                        slippage_total += trade["slippage_cost"]
                        self.assertGreaterEqual(cash, -1e-6)
                        self.assertGreaterEqual(shares[symbol], 0)
                    accrued = sum(amount for _, amount in receivables)
                    market_value = sum(shares[symbol] * frames[symbol].at[date, "close"] for symbol in shares)
                    equity = cash + accrued + market_value
                    self.assertAlmostEqual(chart[date]["equity"], equity, places=6)
                    if "cash" in chart[date]:
                        self.assertAlmostEqual(chart[date]["cash"], cash, places=6)
                    if "receivable" in chart[date]:
                        self.assertAlmostEqual(chart[date]["receivable"], accrued, places=6)
                    for key in ("equity", "half", "sigmoid"):
                        self.assertTrue(math.isfinite(chart[date][key]))
                    if strategy == "half":
                        self.assertAlmostEqual(chart[date]["equity"], chart[date]["half"], places=7)

                portfolio = result["portfolio"]
                self.assertEqual(set(portfolio["holdings"]), {"SPY", "GLD"})
                self.assertAlmostEqual(portfolio["cash"], cash, places=6)
                self.assertAlmostEqual(portfolio["receivable"], accrued, places=6)
                self.assertAlmostEqual(portfolio["equity"], equity, places=6)
                for symbol, holding in portfolio["holdings"].items():
                    self.assertEqual(holding["shares"], shares[symbol])
                    self.assertAlmostEqual(holding["close"], frames[symbol].at[dates[-1], "close"], places=9)
                    self.assertAlmostEqual(holding["market_value"], shares[symbol] * holding["close"], places=6)
                    self.assertAlmostEqual(holding["weight"], holding["market_value"] / equity, places=9)
                total_value = portfolio["cash"] + portfolio["receivable"] + sum(
                    h["market_value"] for h in portfolio["holdings"].values())
                self.assertAlmostEqual(total_value, portfolio["equity"], places=6)
                self.assertLessEqual(sum(h["weight"] for h in portfolio["holdings"].values()), 1 + 1e-9)
                self.assertAlmostEqual(result["metrics"]["final_equity"], equity, places=6)
                self.assertAlmostEqual(result["metrics"]["total_commission"], commission_total, places=7)
                self.assertAlmostEqual(result["metrics"]["slippage_cost"], slippage_total, places=7)
                self.assertAlmostEqual(result["metrics"]["dividend_entitlement"], entitlement_total, places=7)
                for name, baseline in result["baseline_metrics"].items():
                    self.assertAlmostEqual(chart[dates[-1]][name], baseline["final_equity"], places=6)
                self.assertEqual(result["metrics"], result["baseline_metrics"][strategy])

    def test_completed_portfolio_persists_in_sqlite_lists_and_bootstrap(self):
        body, result = self.submit(allocations={"GLD": .5, "SPY": .5})
        listed = self.api("GET", "/api/backtests")["items"]
        listed_row = next(row for row in listed if row["id"] == result["id"])
        self.assertEqual(listed_row["allocations"], body["allocations"])
        boot = self.api("GET", "/api/bootstrap")
        boot_row = next(row for row in boot["backtests"] if row["id"] == result["id"])
        self.assertEqual(boot_row["metrics"], result["metrics"])
        self.assertEqual(boot_row["allocations"], body["allocations"])
        reopened = DemoService(self.db_path, ROOT)
        try:
            persisted = reopened.get_backtest(result["id"])
            for key in ("status", "allocations", "symbol", "strategy", "metrics", "baseline_metrics", "portfolio", "trades"):
                self.assertEqual(persisted[key], result[key], key)
            again = next(row for row in reopened.bootstrap()["backtests"] if row["id"] == result["id"])
            self.assertEqual(again["allocations"], body["allocations"])
        finally:
            reopened.close()

    def test_portfolio_csv_export_keeps_the_asset_identity(self):
        _, result = self.submit()
        status, payload, headers = self.request("GET", "/api/backtests/" + result["id"] + "/export")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/csv"))
        self.assertIn("attachment", headers["Content-Disposition"])
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
        self.assertIn("symbol", reader.fieldnames)
        exported = list(reader)
        self.assertEqual(len(exported), len(result["trades"]))
        self.assertEqual({row["symbol"] for row in exported}, {"SPY", "GLD"})
        for row, trade in zip(exported, result["trades"]):
            self.assertEqual(row["symbol"], trade["symbol"])
            self.assertEqual(row["date"], trade["date"])
            self.assertAlmostEqual(float(row["quantity"]), trade["quantity"])
            self.assertAlmostEqual(float(row["notional"]), trade["notional"], places=7)

    def test_legacy_single_asset_backtest_still_works_and_persists(self):
        task = self.api("POST", "/api/backtests", {
            "symbol": "SPY", "strategy": "half", "start": "2020-01-02", "end": "2020-06-30",
            "initial_cash": 25000,
        }, expected=202)
        result = self.wait_terminal("/api/backtests/" + task["id"])
        self.assertEqual(result["status"], "completed", result.get("error"))
        self.assertEqual(result["symbol"], "SPY")
        self.assertFalse(result.get("allocations"))
        self.assertEqual(set(result["baseline_metrics"]), {"half", "buy_hold_reinvest"})
        self.assertAlmostEqual(result["portfolio"]["equity"], result["metrics"]["final_equity"], places=7)
        self.assertGreaterEqual(result["portfolio"]["cash"], 0)
        self.assertGreaterEqual(result["portfolio"]["shares"], 0)
        self.assertEqual(result["metrics"], result["baseline_metrics"]["half"])
        boot_row = next(row for row in self.api("GET", "/api/bootstrap")["backtests"] if row["id"] == result["id"])
        self.assertEqual(boot_row["symbol"], "SPY")
        self.assertEqual(boot_row["metrics"], result["metrics"])
        status, payload, _ = self.request("GET", "/api/backtests/" + result["id"] + "/export")
        self.assertEqual(status, 200)
        self.assertTrue(payload.decode("utf-8-sig").startswith("date,signal_date,quantity,"))


if __name__ == "__main__":
    unittest.main()
