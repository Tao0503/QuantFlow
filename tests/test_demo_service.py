"""Integration tests of the demo with initialized prices and research engine."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from demo.service import DemoService
import demo.service as service_module


ROOT = Path(__file__).resolve().parents[1]


class DemoServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "demo.sqlite"
        self.service = DemoService(self.path, ROOT)

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def wait_for(self, getter, item_id, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = getter(item_id)
            if row["status"] not in ("queued", "running", "pending"):
                return row
            time.sleep(.02)
        self.fail("Background task did not finish")

    def test_seed_persistence_and_explicit_version_binding(self):
        first = self.service.bootstrap()
        self.assertEqual(len(first["genomes"]), 6)
        self.assertEqual(len(first["instances"]), 2)
        old = first["current_genome"]["id"]
        other = next(g for g in first["genomes"] if g["id"] != old)
        self.service.promote_genome(other["id"])
        self.assertEqual(self.service._get("instances", "inst-spy")["genome_id"], old)
        self.service.promote_genome(other["id"], "inst-spy")
        self.service.close()
        self.service = DemoService(self.path, ROOT)
        again = self.service.bootstrap()
        self.assertEqual(again["current_genome"]["id"], other["id"])
        self.assertEqual(sum(g["role"] == "champion" for g in again["genomes"]), 1)
        self.assertEqual(self.service._get("instances", "inst-spy")["genome_id"], other["id"])

    def test_instance_creation_updates_and_validation(self):
        row = self.service.create_instance({"name": "Gold balance", "symbol": "GLD", "template": "half", "initial_cash": 25000})
        changed = self.service.update_instance(row["id"], {"status": "running", "capital": 30000})
        self.assertEqual(changed["initial_cash"], 30000)
        self.assertEqual(changed["status"], "running")
        for body in ({"initial_cash": float("nan")}, {"symbol": "INVALID"}, {"template": "bad"}, {"monthly_budget": 5}):
            with self.assertRaises(ValueError):
                self.service.create_instance(body)
        with self.assertRaises(KeyError):
            self.service.update_instance("missing", {})

    def test_real_optimization_training_boundary_and_progress(self):
        real_score = service_module.training_score
        observed = []

        def checked_score(frame, *args):
            observed.append(str(frame.index.max().date()))
            return real_score(frame, *args)

        with patch.object(service_module, "training_score", checked_score):
            row = self.service.start_optimization({"method": "random", "population": 16, "generations": 1, "seed": 7, "spawn_mode": "manual", "params": {"mean_reversion": .6, "momentum": .3, "band": .1}})
            final = self.wait_for(self.service.get_task, row["id"])
        self.assertEqual(final["status"], "completed", final.get("error"))
        self.assertEqual(final["progress"], {"completed": 16, "total": 16})
        self.assertEqual(len(final["history"]), 16)
        self.assertTrue(all(d <= "2018-12-31" for d in observed))
        self.assertFalse(final["test_used"])
        genome = self.service._get("genomes", final["genome_id"])
        self.assertEqual(genome["role"], "challenger")
        for key, value in genome["params"].items():
            low, high = final["effective_bounds"][key]
            self.assertTrue(low <= value <= high)

    def test_cancel_does_not_publish_partial_genome(self):
        real_score = service_module.training_score
        entered, release = threading.Event(), threading.Event()

        def waiting_score(*args):
            entered.set()
            release.wait(5)
            return real_score(*args)

        with patch.object(service_module, "training_score", waiting_score):
            row = self.service.start_optimization({"population": 16, "generations": 1, "spawn_mode": "random"})
            self.assertTrue(entered.wait(5))
            cancelled = self.service.cancel_task(row["id"])
            release.set()
            self.assertEqual(cancelled["status"], "cancelled")
            self.service.executor.submit(lambda: None).result(timeout=5)
            time.sleep(.05)
        self.assertEqual(self.service.get_task(row["id"])["status"], "cancelled")
        self.assertEqual(len(self.service.bootstrap()["genomes"]), 6)

    def test_actual_backtest_benchmarks_and_historical_replay(self):
        row = self.service.start_backtest({"symbol": "SPY", "start": "2022-01-01", "end": "2025-12-31"})
        result = self.wait_for(self.service.get_backtest, row["id"])
        self.assertEqual(result["status"], "completed", result.get("error"))
        frozen = json.loads((ROOT / "results" / "run_001" / "results.json").read_text())
        expected = next(
            metrics for metrics in frozen["metrics"]
            if (metrics["symbol"], metrics["period"], metrics["strategy"])
            == ("SPY", "test", "selected_sigmoid")
        )
        self.assertAlmostEqual(result["metrics"]["cagr"], expected["cagr"], places=9)
        self.assertEqual(set(result["baseline_metrics"]), {"half", "buy_hold_reinvest"})
        self.assertLessEqual(len(result["chart"]), 400)
        self.assertTrue(result["observed_history"])
        for trade in result["trades"]:
            self.assertLess(trade["signal_date"], trade["date"])
        replay = self.service.get_replay("inst-spy")
        self.assertFalse(replay["replay"]["can_send_orders"])
        self.assertEqual(replay["metrics"], result["metrics"])
        self.assertGreater(replay["portfolio"]["cash"], 0)
        self.assertEqual(len(replay["daily"]), replay["metrics"]["sessions"])
        self.assertEqual(len(replay["bars"]), replay["metrics"]["sessions"])
        self.assertFalse(replay["chart_sampled"])

    def test_invalid_backtests_and_recovery(self):
        for body in ({"start": "2025-01-01", "end": "2022-01-01"}, {"symbol": "ABC"},
                     {"costs": {"slippage_bps": -1}}, {"params": {"mean_reversion": 3, "momentum": 1, "band": .1}}):
            with self.assertRaises(ValueError):
                self.service.start_backtest(body)
        with self.assertRaises(ValueError):
            self.service.start_optimization({"seed": float("inf")})
        self.service._put("tasks", {"id": "unfinished", "status": "running", "history": []})
        self.service.close()
        self.service = DemoService(self.path, ROOT)
        self.assertEqual(self.service.get_task("unfinished")["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
