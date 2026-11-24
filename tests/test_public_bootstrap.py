"""Public clone setup must be deterministic, offline, and non-destructive."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from demo.bootstrap_public import ROOT, SYMBOLS, bootstrap, digest, generate_synthetic_data, synthetic_chart
from quantlab.data import download_all, load_data, parse_chart


class PublicBootstrapTests(unittest.TestCase):
    def test_deterministic_data_and_existing_ingestion_contract(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = Path(first), Path(second)
            with patch("quantlab.data.urlopen", side_effect=AssertionError("Synthetic setup must not access network")):
                one, two = generate_synthetic_data(a), generate_synthetic_data(b)
                self.assertTrue(download_all(a)["synthetic"])
            self.assertTrue(one["synthetic"])
            self.assertEqual([row["symbol"] for row in one["datasets"]], list(SYMBOLS))
            for left, right in zip(one["datasets"], two["datasets"]):
                self.assertEqual(left["raw_sha256"], right["raw_sha256"])
                self.assertEqual(left["csv_sha256"], right["csv_sha256"])
                self.assertEqual(left["raw_sha256"], digest(a / left["raw_path"]))
                frame = load_data(a / left["csv_path"])
                self.assertEqual(str(frame.index[0].date()), "2004-01-01")
                self.assertEqual(str(frame.index[-1].date()), "2025-12-31")
                self.assertTrue((frame.index.weekday < 5).all())
                self.assertGreater(len(frame), 5500)
                if left["symbol"] in ("SPY", "IEF"):
                    self.assertGreater((frame.dividend > 0).sum(), 0)
            self.assertEqual(load_data(a / "data/SPY.csv").index.tolist(), load_data(a / "data/IEF.csv").index.tolist())

    def test_generated_chart_is_explicitly_labelled_and_parseable(self):
        for symbol in SYMBOLS:
            payload = synthetic_chart(symbol)
            self.assertTrue(payload["synthetic"])
            self.assertIn("不是", payload["notice"])
            frame, warnings = parse_chart(payload, symbol)
            self.assertFalse(warnings)
            self.assertFalse(frame.empty)
            json.dumps(payload, allow_nan=False)

    def test_refuses_real_or_partial_data_results_and_state(self):
        for relative in ("data", "results", "demo/state"):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                blocker = root / relative
                blocker.mkdir(parents=True)
                evidence = blocker / "existing.txt"
                evidence.write_text("preserve original evidence")
                with self.assertRaises(FileExistsError):
                    bootstrap(root)
                self.assertEqual(evidence.read_text(), "preserve original evidence")
                self.assertFalse(list(root.glob(".bootstrap-public-*")))

    def test_failure_before_publication_leaves_clean_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for package in ("quantlab", "demo"):
                (root / package).mkdir()
                for path in (ROOT / package).glob("*.py"):
                    shutil.copy2(path, root / package / path.name)
            for name in ("config.json", "verify_results.py"):
                shutil.copy2(ROOT / name, root / name)
            before = digest(root / "config.json")
            with patch("demo.bootstrap_public._pipeline", side_effect=RuntimeError("calculation interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    bootstrap(root)
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "results").exists())
            self.assertFalse(list(root.glob(".bootstrap-public-*")))
            self.assertEqual(digest(root / "config.json"), before)


if __name__ == "__main__":
    unittest.main()
