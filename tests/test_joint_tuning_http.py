"""Real HTTP checks for joint search isolation, selection and explicit promotion."""
from __future__ import annotations

import math
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import pandas as pd

from demo.service import DemoService
from demo.server import Handler
import demo.service as service_module
from tests import test_demo_http as legacy_http


ROOT = Path(__file__).resolve().parents[1]


class JointTuningHTTPTests(unittest.TestCase):
    # Reuse the temporary SQLite and real loopback server without inherited tests.
    setUp = legacy_http.DemoHTTPTests.setUp
    stop_server = legacy_http.DemoHTTPTests.stop_server
    request = legacy_http.DemoHTTPTests.request
    api = legacy_http.DemoHTTPTests.api
    wait_terminal = legacy_http.DemoHTTPTests.wait_terminal

    def test_invalid_budgets_do_not_create_tuning_runs(self):
        before = self.api("GET", "/api/bootstrap")
        for budget in (63, 1025, -1, 64.5, True, None, "invalid", float("nan"), float("inf")):
            with self.subTest(budget=budget):
                self.api("POST", "/api/optimization/tuning", {"budget_per_run": budget}, expected=400)
        after = self.api("GET", "/api/bootstrap")
        self.assertEqual(after["tuning_runs"], before["tuning_runs"])
        self.assertEqual(after["tasks"], before["tasks"])
        self.assertEqual(after["current_genome"], before["current_genome"])

    def test_real_six_run_tuning_is_isolated_ranked_persisted_and_not_applied(self):
        # Polling is intentionally frequent; suppress only this test's access log.
        access_log = patch.object(Handler, "log_message", return_value=None)
        access_log.start()
        self.addCleanup(access_log.stop)
        before = self.api("GET", "/api/bootstrap")
        incumbent = before["current_genome"]
        before_instances = {row["id"]: row for row in before["instances"]}
        original = service_module.run_tuning
        observed_boundaries = []

        def checked_tuning(training_frame, validation_frame, protocol, incumbent_params, on_progress):
            self.assertLessEqual(training_frame.index.max(), pd.Timestamp("2018-12-31"))
            self.assertLessEqual(validation_frame.index.max(), pd.Timestamp("2021-12-31"))
            self.assertFalse((training_frame.index >= pd.Timestamp("2019-01-01")).any())
            self.assertFalse((validation_frame.index >= pd.Timestamp("2022-01-01")).any())
            self.assertEqual(incumbent_params, incumbent["params"])
            observed_boundaries.append((training_frame.index.max(), validation_frame.index.max()))
            return original(training_frame, validation_frame, protocol, incumbent_params, on_progress)

        with patch.object(service_module, "run_tuning", side_effect=checked_tuning):
            job = self.api("POST", "/api/optimization/tuning", {"budget_per_run": 64}, expected=202)
            self.assertIn(job["status"], ("queued", "running", "completed"))
            result = self.wait_terminal("/api/optimization/tasks/" + job["id"], timeout=45)
        self.assertEqual(result["status"], "completed", result.get("error"))
        self.assertEqual(len(observed_boundaries), 1)
        self.assertEqual(result["type"], "joint_tuning")
        self.assertEqual(result["budget_per_run"], 64)
        self.assertEqual(result["evaluations"], 384)
        self.assertFalse(result["test_used"])
        self.assertEqual(result["progress"]["completed"], 394)
        self.assertEqual(result["progress"]["total"], 394)
        finalists = result["finalists"]
        self.assertEqual(len(finalists), 6)
        expected_pairs = {(method, seed) for method in ("ga", "random") for seed in (11, 29, 47)}
        self.assertEqual({(row["method"], row["seed"]) for row in finalists}, expected_pairs)
        for candidate in finalists:
            self.assertEqual(set(candidate["params"]), {"mean_reversion", "momentum", "band"})
            self.assertTrue(math.isfinite(candidate["training_score"]))
            self.assertGreaterEqual(candidate["validation"]["start"], "2019-01-01")
            self.assertLessEqual(candidate["validation"]["end"], "2021-12-31")
            self.assertAlmostEqual(candidate["validation_delta"], candidate["validation"]["sharpe_rf0"]
                                   - result["benchmarks"]["half"]["sharpe_rf0"], places=12)
        self.assertEqual(len(result["histories"]), 6)
        self.assertEqual(sum(row["evaluations"] for row in result["histories"]), 384)
        for history in result["histories"]:
            self.assertEqual(history["budget"], 64)
            self.assertEqual(history["evaluations"], 64)
            self.assertEqual(history["requests"], 64 + history["cache_hits"])
        ranked = sorted(finalists, key=lambda row: (
            -row["validation"]["sharpe_rf0"], row["validation"]["max_drawdown"],
            row["validation"]["annual_turnover"], row["method"], row["seed"],
        ))
        winner = result["winner"]
        for key in ("params", "method", "seed", "training_score", "validation"):
            self.assertEqual(winner[key], ranked[0][key], key)
        self.assertIn("validation_delta", winner)
        self.assertEqual(set(result["benchmarks"]), {"half", "incumbent", "fixed_sigmoid", "same_band_half"})

        after = self.api("GET", "/api/bootstrap")
        self.assertEqual(after["current_genome"], incumbent)
        self.assertEqual({row["id"]: row for row in after["instances"]}, before_instances)
        self.assertNotIn(job["id"], {row["id"] for row in after["tasks"]})
        self.assertIn(job["id"], {row["id"] for row in after["tuning_runs"]})
        classic_ids = {row["id"] for row in self.api("GET", "/api/optimization/tasks")["items"]}
        self.assertNotIn(job["id"], classic_ids)
        genome = next(row for row in after["genomes"] if row["id"] == result["genome_id"])
        self.assertEqual(genome["role"], "challenger")
        self.assertEqual(genome["params"], winner["params"])
        self.assertEqual(len(after["genomes"]), len(before["genomes"]) + 1)
        reopened = DemoService(self.db_path, ROOT)
        try:
            persisted = reopened.get_task(job["id"])
            for key in ("status", "type", "winner", "finalists", "benchmarks", "evaluations", "test_used", "genome_id"):
                self.assertEqual(persisted[key], result[key], key)
            persisted_boot = reopened.bootstrap()
            self.assertEqual(persisted_boot["current_genome"], incumbent)
            self.assertIn(job["id"], {row["id"] for row in persisted_boot["tuning_runs"]})
            self.assertNotIn(job["id"], {row["id"] for row in persisted_boot["tasks"]})
        finally:
            reopened.close()

    def test_default_budget_and_cancellation_preserve_champion_and_candidates(self):
        before = self.api("GET", "/api/bootstrap")
        entered, release = threading.Event(), threading.Event()
        original = service_module.run_tuning

        def paused_tuning(*args):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("Test cancellation barrier timed out")
            return original(*args)

        with patch.object(service_module, "run_tuning", side_effect=paused_tuning):
            job = self.api("POST", "/api/optimization/tuning", {}, expected=202)
            try:
                self.assertTrue(entered.wait(timeout=5))
                self.assertEqual(job["budget_per_run"], 256)
                cancelled = self.api("POST", "/api/optimization/tasks/" + job["id"] + "/cancel", {})
                self.assertEqual(cancelled["status"], "cancelled")
            finally:
                release.set()
            result = self.wait_terminal("/api/optimization/tasks/" + job["id"])
            deadline = time.monotonic() + 10
            while job["id"] in self.service.cancel_events and time.monotonic() < deadline:
                time.sleep(.01)
        self.assertEqual(result["status"], "cancelled")
        self.assertNotIn(job["id"], self.service.cancel_events)
        self.assertNotIn("genome_id", result)
        after = self.api("GET", "/api/bootstrap")
        self.assertEqual(after["current_genome"], before["current_genome"])
        self.assertEqual(after["instances"], before["instances"])
        self.assertEqual(after["genomes"], before["genomes"])


if __name__ == "__main__":
    unittest.main()
