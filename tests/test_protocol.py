"""Runner boundary and freeze checks using synthetic data; never read outcomes."""

from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from quantlab import run


REAL_ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "data").mkdir()
        (self.root / "quantlab").mkdir()
        (self.root / "quantlab/engine.py").write_text("# synthetic method snapshot\n")
        self.config = json.loads((REAL_ROOT / "config.json").read_text())
        # Reverse arrival order so the seed tie-break cannot pass accidentally.
        self.config["seeds"] = [47, 29, 11]
        (self.root / "config.json").write_text(json.dumps(self.config))
        self.manifest = {"datasets": []}
        (self.root / "data/manifest.json").write_text(json.dumps(self.manifest))
        self.frame = pd.DataFrame({"tr_close": range(100, 108)}, index=pd.to_datetime([
            "2009-12-31", "2010-01-04", "2018-12-31", "2019-01-02",
            "2021-12-31", "2022-01-03", "2025-12-31", "2026-01-02",
        ]))
        self.search_calls = []
        self.training_calls = []
        self.validation_calls = []
        self.lookup = {}

    def fit_fixture(self, name="fit", validation_metrics=None, budget_delta=0):
        out = self.root / name

        def fake_features(frame):
            return pd.DataFrame({"deviation": 0., "momentum": 0.}, index=frame.index)

        def fake_training(frame, feature_frame, params, folds, costs, initial_cash):
            self.assertLessEqual(frame.index.max(), pd.Timestamp("2018-12-31"))
            self.assertTrue(feature_frame.index.equals(frame.index))
            self.assertTrue(all(pd.Timestamp(end) <= pd.Timestamp("2018-12-31") for _, end in folds))
            self.assertEqual(initial_cash, self.config["initial_cash"])
            self.training_calls.append(frame.index.copy())
            return 1.

        def fake_optimize(objective, bounds, *, method, budget, seed):
            self.assertEqual(bounds, self.config["bounds"])
            self.search_calls.append((method, seed, budget))
            marker = (0.2 if method == "ga" else 1.2) + seed / 1000
            params = {"mean_reversion": marker, "momentum": 1., "band": .05}
            self.lookup[marker] = (method, seed)
            objective(params)
            return {"best_params": params, "best_score": float(seed),
                    "evaluations": budget + budget_delta, "cache_hits": 0, "history": []}

        def fake_backtest(frame, kind, params, start, end, costs, initial_cash, feature_frame):
            self.assertEqual(kind, "sigmoid")
            self.assertEqual([start, end], self.config["validation"])
            self.assertLessEqual(frame.index.max(), pd.Timestamp("2021-12-31"))
            self.assertTrue(feature_frame.index.equals(frame.index))
            self.assertEqual(initial_cash, self.config["initial_cash"])
            key = self.lookup[params["mean_reversion"]]
            self.validation_calls.append(key)
            metrics = {"sharpe_rf0": 1., "max_drawdown": .1, "annual_turnover": .2}
            if validation_metrics:
                metrics.update(validation_metrics.get(key, {}))
            return SimpleNamespace(metrics=metrics)

        with ExitStack() as stack:
            stack.enter_context(patch.object(run, "ROOT", self.root))
            stack.enter_context(patch.object(run, "download_all", return_value=self.manifest))
            loader = stack.enter_context(patch.object(run, "load_data", return_value=self.frame))
            stack.enter_context(patch.object(run, "features", side_effect=fake_features))
            stack.enter_context(patch.object(run, "training_score", side_effect=fake_training))
            stack.enter_context(patch.object(run, "optimize", side_effect=fake_optimize))
            stack.enter_context(patch.object(run, "backtest", side_effect=fake_backtest))
            stack.enter_context(patch("sys.stdout", new=io.StringIO()))
            run.fit(out)
            loader.assert_called_once_with(self.root / "data/SPY.csv")
        return out

    def test_fit_isolates_periods_and_selects_only_six_budgeted_champions(self):
        out = self.fit_fixture()
        expected = {(method, seed, self.config["search_budget_per_method_seed"])
                    for method in ("ga", "random") for seed in (11, 29, 47)}
        self.assertEqual(set(self.search_calls), expected)
        self.assertEqual(len(self.search_calls), 6)
        self.assertEqual(len(self.training_calls), 6)
        self.assertEqual(len(self.validation_calls), 6)
        self.assertEqual(len(list(out.glob("search_*.json"))), 6)
        selection = json.loads((out / "selection.json").read_text())
        self.assertEqual(len(selection["candidates"]), 6)
        self.assertTrue(selection["test_has_not_been_evaluated_by_this_pipeline"])
        self.assertFalse((out / "results.json").exists())
        # Equal validation metrics must ignore the deliberately different training scores.
        self.assertEqual((selection["winner"]["method"], selection["winner"]["seed"]), ("ga", 11))

    def test_declared_validation_priority_and_deterministic_ties(self):
        cases = [
            ("sharpe", {("random", 47): {"sharpe_rf0": 2.}}, ("random", 47)),
            ("drawdown", {("ga", 29): {"max_drawdown": .05}}, ("ga", 29)),
            ("turnover", {("random", 11): {"annual_turnover": .01}}, ("random", 11)),
            ("method_before_seed", {("ga", 47): {"sharpe_rf0": 2.},
                                     ("random", 11): {"sharpe_rf0": 2.}}, ("ga", 47)),
            ("seed", {}, ("ga", 11)),
        ]
        for name, overrides, expected in cases:
            with self.subTest(priority=name):
                out = self.fit_fixture(name, overrides)
                winner = json.loads((out / "selection.json").read_text())["winner"]
                self.assertEqual((winner["method"], winner["seed"]), expected)

    def test_fit_rejects_an_under_budget_optimizer_result(self):
        with self.assertRaisesRegex(AssertionError, "Unequal candidate"):
            self.fit_fixture(budget_delta=-1)
        self.assertEqual(len(self.search_calls), 1)
        self.assertEqual(self.validation_calls, [])

    def test_evaluate_rejects_changed_method_config_protocol_and_manifest(self):
        changes = [
            ("method", "quantlab/engine.py", "Method code or config"),
            ("config", "config.json", "Method code or config"),
            ("manifest", "data/manifest.json", "Data manifest changed"),
            ("protocol", None, "Frozen protocol changed"),
        ]
        for name, relative, message in changes:
            with self.subTest(changed=name):
                out = self.fit_fixture("freeze_" + name)
                path = self.root / relative if relative else out / "protocol_frozen.json"
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                try:
                    with patch.object(run, "ROOT", self.root), \
                         patch.object(run, "download_all", return_value=self.manifest), \
                         patch.object(run, "load_data") as loader, \
                         patch.object(run, "backtest") as simulator:
                        with self.assertRaisesRegex(ValueError, message):
                            run.evaluate(out)
                        loader.assert_not_called()
                        simulator.assert_not_called()
                    self.assertFalse((out / "results.json").exists())
                finally:
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
