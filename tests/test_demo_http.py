"""Real local HTTP acceptance checks with isolated SQLite state and cached prices."""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from demo.server import MAX_BODY, make_server
from demo.service import DemoService
from quantlab.engine import Costs, backtest, features
from quantlab.data import load_data
import demo.service as service_module


ROOT = Path(__file__).resolve().parents[1]


class DemoHTTPTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.static = self.temp / "dist"
        self.static.mkdir()
        (self.static / "index.html").write_text("<!doctype html><title>Acceptance app</title>")
        (self.static / "assets").mkdir()
        (self.static / "assets/app.js").write_text("console.log('acceptance');")
        self.db_path = self.temp / "workbench.sqlite3"
        self.service = DemoService(self.db_path, ROOT)
        self.server = make_server(port=0, service=self.service, static_dir=self.static)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.service.close(wait=True)

    def request(self, method, path, body=None, *, raw=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=15)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        actual_headers = {"Content-Type": "application/json"} if data is not None else {}
        actual_headers.update(headers or {})
        try:
            connection.request(method, path, body=data, headers=actual_headers)
            response = connection.getresponse()
            payload = response.read()
            response_headers = dict(response.getheaders())
            if response_headers.get("Content-Type", "").startswith("application/json"):
                payload = json.loads(payload)
            return response.status, payload, response_headers
        finally:
            connection.close()

    def api(self, method, path, body=None, expected=200):
        status, payload, headers = self.request(method, path, body)
        self.assertEqual(status, expected, (path, payload))
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        return payload

    def wait_terminal(self, path, *, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self.api("GET", path)
            if task.get("status") in {"completed", "cancelled", "failed", "interrupted"}:
                return task
            time.sleep(.02)
        self.fail(f"Task did not finish: {path}")

    def test_health_bootstrap_and_resource_lists(self):
        health = self.api("GET", "/api/health")
        self.assertEqual(health["status"], "ok")
        boot = self.api("GET", "/api/bootstrap")
        for key in ("instances", "genomes", "tasks", "backtests", "overview", "protocol"):
            self.assertIn(key, boot)
        for path, key in (("instances", "instances"), ("genomes", "genomes"),
                          ("optimization/tasks", "tasks"), ("backtests", "backtests")):
            items = self.api("GET", "/api/" + path)["items"]
            self.assertIsInstance(items, list)
            self.assertEqual([row["id"] for row in items], [row["id"] for row in boot[key]])

    def test_json_inputs_missing_ids_and_unknown_api_errors(self):
        for raw in (b"{", b"[]", b"null", b'{"name":NaN}',
                    b'{"name":Infinity}', b'{"unused":1e999}'):
            with self.subTest(body=raw):
                status, payload, _ = self.request("POST", "/api/instances", raw=raw)
                self.assertEqual(status, 400, payload)
                self.assertIn("error", payload)
        status, payload, _ = self.request("POST", "/api/instances", raw=b"{}",
                                           headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415, payload)
        for method, path, body in (
            ("GET", "/api/missing-resource", None),
            ("GET", "/api/optimization/tasks/missing-id", None),
            ("GET", "/api/backtests/missing-id", None),
            ("PATCH", "/api/instances/missing-id", {"status": "paused"}),
            ("POST", "/api/genomes/missing-id/promote", {}),
        ):
            with self.subTest(path=path):
                self.api(method, path, body, expected=404)

    def test_request_body_limit_is_checked_before_read(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.putrequest("POST", "/api/instances")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(MAX_BODY + 1))
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 413)
            self.assertIn("error", json.loads(response.read()))
        finally:
            connection.close()

    def test_local_host_and_same_origin_writes(self):
        for headers in ({"Host": "outside.example"}, {"Origin": "https://outside.example"},
                        {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                status, payload, _ = self.request("POST", "/api/instances", {}, headers=headers)
                self.assertEqual(status, 403, payload)
        origin = f"http://127.0.0.1:{self.server.server_port}"
        status, payload, _ = self.request("POST", "/api/instances",
            {"name": "Same origin", "symbol": "SPY", "template": "half", "initial_cash": 20000},
            headers={"Origin": origin})
        self.assertEqual(status, 201, payload)

    def test_static_assets_spa_fallback_and_path_containment(self):
        status, payload, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Acceptance app", payload)
        self.assertTrue(headers["Content-Type"].startswith("text/html"))
        self.assertEqual(self.request("GET", "/research")[0], 200)
        self.assertEqual(self.request("GET", "/assets/app.js")[0], 200)
        self.assertEqual(self.request("GET", "/assets/missing.js")[0], 404)
        secret = self.temp / "secret.txt"
        secret.write_text("OUTSIDE_STATIC_SENTINEL")
        (self.static / "assets/leak.txt").symlink_to(secret)
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/leak.txt"):
            with self.subTest(path=path):
                status, payload, _ = self.request("GET", path)
                self.assertIn(status, (403, 404), payload)
        (self.static / "index.html").unlink()
        (self.static / "index.html").symlink_to(secret)
        # The SPA fallback must apply the same containment rule as normal files.
        self.assertIn(self.request("GET", "/unknown-route")[0], (403, 404))

    def test_invalid_domain_parameters_do_not_create_records(self):
        before = self.api("GET", "/api/bootstrap")
        cases = [
            ("/api/instances", {"symbol": "BTC"}),
            ("/api/instances", {"initial_cash": -1}),
            ("/api/instances", {"initial_cash": True}),
            ("/api/instances", {"monthly_budget": 100}),
            ("/api/instances", {"status": "live"}),
            ("/api/optimization/tasks", {"symbol": "GLD"}),
            ("/api/optimization/tasks", {"population": 0}),
            ("/api/optimization/tasks", {"generations": 33}),
            ("/api/optimization/tasks", {"method": "unknown"}),
            ("/api/optimization/tasks", {"spawn_mode": "manual", "params": {"band": .1}}),
            ("/api/backtests", {"start": "2025-01-01", "end": "2024-01-01"}),
            ("/api/backtests", {"end": "2026-01-01"}),
            ("/api/backtests", {"costs": {"slippage_bps": -1}}),
            ("/api/backtests", {"params": {"mean_reversion": 3., "momentum": 1., "band": .05}}),
        ]
        for path, body in cases:
            with self.subTest(path=path, body=body):
                self.api("POST", path, body, expected=400)
        after = self.api("GET", "/api/bootstrap")
        for key in ("instances", "tasks", "backtests", "genomes"):
            self.assertEqual(len(before[key]), len(after[key]), key)

    def test_real_optimize_train_boundary_promote_and_restart_persistence(self):
        boot = self.api("GET", "/api/bootstrap")
        original_champion = boot["current_genome"]["id"]
        instance = self.api("POST", "/api/instances", {
            "name": "HTTP acceptance", "symbol": "GLD", "template": "sigmoid", "initial_cash": 25000,
        }, expected=201)
        self.assertEqual(instance["genome_id"], original_champion)
        seen_training_frames = []
        seen_validation_frames = []
        original_score = service_module.training_score
        original_backtest = service_module.backtest

        def checked_score(frame, feature_frame, *args, **kwargs):
            self.assertLessEqual(frame.index.max(), __import__("pandas").Timestamp("2018-12-31"))
            self.assertTrue(frame.index.equals(feature_frame.index))
            seen_training_frames.append(frame.index.max())
            return original_score(frame, feature_frame, *args, **kwargs)

        def checked_validation(frame, *args, **kwargs):
            self.assertLessEqual(frame.index.max(), __import__("pandas").Timestamp("2021-12-31"))
            seen_validation_frames.append(frame.index.max())
            return original_backtest(frame, *args, **kwargs)

        with patch.object(service_module, "training_score", side_effect=checked_score), \
             patch.object(service_module, "backtest", side_effect=checked_validation):
            task = self.api("POST", "/api/optimization/tasks", {
                "method": "ga", "population": 16, "generations": 1, "seed": 13, "spawn_mode": "random",
            }, expected=202)
            task = self.wait_terminal("/api/optimization/tasks/" + task["id"])
        self.assertEqual(task["status"], "completed", task.get("error"))
        self.assertEqual(len(seen_training_frames), 16)
        self.assertEqual(len(seen_validation_frames), 1)
        self.assertEqual(task["evaluations"], 16)
        self.assertEqual(task["progress"], {"completed": 16, "total": 16})
        self.assertFalse(task["test_used"])
        genome_id = task["genome_id"]
        self.api("POST", f"/api/genomes/{genome_id}/promote", {})
        promoted = self.api("GET", "/api/bootstrap")
        self.assertEqual(promoted["current_genome"]["id"], genome_id)
        old_instance = next(row for row in promoted["instances"] if row["id"] == instance["id"])
        self.assertEqual(old_instance["genome_id"], original_champion)
        self.api("POST", f"/api/genomes/{genome_id}/promote", {"instance_id": instance["id"]})
        self.api("PATCH", "/api/instances/" + instance["id"], {"status": "running"})
        reopened = DemoService(self.db_path, ROOT)
        try:
            persisted = reopened.bootstrap()
            self.assertEqual(persisted["current_genome"]["id"], genome_id)
            row = next(row for row in persisted["instances"] if row["id"] == instance["id"])
            self.assertEqual(row["genome_id"], genome_id)
            self.assertEqual(row["status"], "running")
        finally:
            reopened.close()
        replay = self.api("GET", "/api/replay/" + instance["id"])
        self.assertEqual(replay["mode"], "historical_replay")
        self.assertFalse(replay["replay"]["can_send_orders"])
        self.assertEqual(replay["instance"]["genome_id"], genome_id)

    def test_real_backtest_benchmarks_and_csv_export(self):
        params = {"mean_reversion": 1., "momentum": 1., "band": .05}
        request = {"symbol": "SPY", "params": params, "start": "2019-01-01", "end": "2021-12-31",
                   "initial_cash": 50000, "costs": {"commission_bps": 1., "slippage_bps": 5.}}
        task = self.api("POST", "/api/backtests", request, expected=202)
        result = self.wait_terminal("/api/backtests/" + task["id"])
        self.assertEqual(result["status"], "completed", result.get("error"))
        self.assertTrue(result["observed_history"])
        self.assertEqual(result["mode"], "historical_replay")
        frame = load_data(ROOT / "data/SPY.csv")
        feature_frame = features(frame)
        for kind in ("sigmoid", "half", "buy_hold_reinvest"):
            expected = backtest(frame, kind, params, request["start"], request["end"], Costs(), 50000,
                                feature_frame=feature_frame)
            actual = result["metrics"] if kind == "sigmoid" else result["baseline_metrics"][kind]
            for key in ("final_equity", "cagr", "sharpe_rf0", "max_drawdown", "total_commission"):
                self.assertAlmostEqual(actual[key], expected.metrics[key], places=9, msg=(kind, key))
        self.assertLessEqual(len(result["chart"]), 400)
        self.assertEqual(result["chart_total_sessions"], result["metrics"]["sessions"])
        self.assertEqual(result["trades"], result["fills"])
        self.assertAlmostEqual(result["portfolio"]["equity"], result["metrics"]["final_equity"])
        for trade in result["trades"]:
            self.assertLess(trade["signal_date"], trade["date"])
        status, payload, headers = self.request("GET", "/api/backtests/" + task["id"] + "/export")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/csv"))
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertTrue(payload.decode("utf-8-sig").startswith("date,signal_date,quantity,"))

    def test_cancelled_optimization_does_not_publish_a_partial_genome(self):
        before = len(self.api("GET", "/api/genomes")["items"])
        entered, release = threading.Event(), threading.Event()
        original = service_module.training_score

        def slow_first_objective(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("Test cancellation barrier timed out")
            return original(*args, **kwargs)

        with patch.object(service_module, "training_score", side_effect=slow_first_objective):
            task = self.api("POST", "/api/optimization/tasks", {
                "method": "random", "population": 16, "generations": 1, "spawn_mode": "random",
            }, expected=202)
            try:
                self.assertTrue(entered.wait(timeout=5))
                cancelled = self.api("POST", "/api/optimization/tasks/" + task["id"] + "/cancel", {})
                self.assertEqual(cancelled["status"], "cancelled")
            finally:
                release.set()
            result = self.wait_terminal("/api/optimization/tasks/" + task["id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertNotIn("genome_id", result)
        # Wait for the worker's cooperative cancellation path to finish, too.
        deadline = time.monotonic() + 5
        while task["id"] in self.service.cancel_events and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertNotIn(task["id"], self.service.cancel_events)
        self.assertEqual(len(self.api("GET", "/api/genomes")["items"]), before)


if __name__ == "__main__":
    unittest.main()
