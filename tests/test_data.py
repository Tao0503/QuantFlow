"""Deterministic data contract tests; no network access required."""
import copy
import unittest

import numpy as np
import pandas as pd

from quantlab.data import parse_chart, validate_data


def sample_payload():
    dates = pd.date_range("2020-01-02", periods=3, freq="B", tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
    timestamps = [int(date.timestamp()) for date in dates]
    return {"chart": {"result": [{
        "meta": {"symbol": "SPY", "exchangeTimezoneName": "America/New_York"},
        "timestamp": timestamps,
        "indicators": {"quote": [{"open": [100, 98, 99], "high": [101, 100, 102],
                                     "low": [99, 97, 98], "close": [100, 99, 101], "volume": [10, 20, 30]}]},
        "events": {"dividends": {str(timestamps[1]): {"date": timestamps[1], "amount": 1.0}}},
    }], "error": None}}


class DataTests(unittest.TestCase):
    def test_session_dates_and_ex_dividend_total_return(self):
        data, warnings = parse_chart(sample_payload(), "SPY")
        self.assertEqual(str(data.index[0].date()), "2020-01-02")
        self.assertEqual(data.loc["2020-01-03", "dividend"], 1.0)
        self.assertAlmostEqual(data.loc["2020-01-03", "tr_close"], 100.0)
        self.assertEqual(warnings, [])

    def test_future_price_does_not_rewrite_past_features(self):
        payload = sample_payload()
        first, _ = parse_chart(payload, "SPY")
        changed = copy.deepcopy(payload)
        quote = changed["chart"]["result"][0]["indicators"]["quote"][0]
        quote["close"][-1] = 102
        quote["high"][-1] = 103
        second, _ = parse_chart(changed, "SPY")
        np.testing.assert_array_equal(first["tr_close"].iloc[:-1], second["tr_close"].iloc[:-1])

    def test_unsupported_split_is_rejected(self):
        payload = sample_payload()
        result = payload["chart"]["result"][0]
        result["events"]["splits"] = {"s": {"date": result["timestamp"][1], "numerator": 2, "denominator": 1}}
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_chart(payload, "SPY")

    def test_missing_dividend_is_rejected_for_spy(self):
        payload = sample_payload()
        payload["chart"]["result"][0]["events"] = {}
        with self.assertRaisesRegex(ValueError, "dividend"):
            parse_chart(payload, "SPY")

    def test_duplicate_unsorted_and_after_cutoff_dates(self):
        data, _ = parse_chart(sample_payload(), "SPY")
        for invalid in (pd.concat([data, data.iloc[:1]]), data.iloc[::-1], data.set_axis(data.index + pd.DateOffset(years=8))):
            with self.assertRaises(ValueError):
                validate_data(invalid)

    def test_invalid_price_and_nonfinite_are_rejected(self):
        data, _ = parse_chart(sample_payload(), "SPY")
        for name, value in (("open", -1), ("high", 1), ("close", np.nan), ("split_ratio", 2)):
            bad = data.copy()
            bad.iloc[0, bad.columns.get_loc(name)] = value
            with self.assertRaises(ValueError):
                validate_data(bad)


if __name__ == "__main__":
    unittest.main()
