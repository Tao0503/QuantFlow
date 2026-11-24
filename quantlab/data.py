"""Download, cache and validate daily ETF history used by the research experiment.

Prices are Yahoo's unadjusted daily OHLC (subject to vendor revisions). Dividends
are mapped to their ex-date, not their later payment date. ``tr_close`` is a
forward-built total-return index for features, never an executable price.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SYMBOLS = ("SPY", "GLD")
START = "2004-01-01"
END_EXCLUSIVE = "2026-01-01"
COLUMNS = ("open", "high", "low", "close", "volume", "dividend", "split_ratio", "tr_close")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_data(frame: pd.DataFrame, *, cutoff: str = END_EXCLUSIVE) -> None:
    """Reject ambiguous dates, unsupported splits and invalid market prices."""
    if frame.empty:
        raise ValueError("Empty daily price data")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex")
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Session dates must be sorted, unique and nonmissing")
    if frame.index.tz is not None or not (frame.index == frame.index.normalize()).all():
        raise ValueError("Expected timezone-naive midnight session dates")
    if frame.index.min() < pd.Timestamp(START) or frame.index.max() >= pd.Timestamp(cutoff):
        raise ValueError("Daily data lies outside the declared research coverage")
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing data columns: {sorted(missing)}")
    values = frame[list(COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite numeric market data")
    if (frame[["open", "high", "low", "close", "tr_close"]] <= 0).any().any():
        raise ValueError("OHLC and total-return levels must be positive")
    if (frame[["volume", "dividend"]] < 0).any().any():
        raise ValueError("Volume and dividends must be nonnegative")
    if not np.allclose(frame["split_ratio"].to_numpy(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Splits are unsupported: explicit share and price accounting is required")
    tolerance = 1e-7
    if (frame["high"] + tolerance < frame[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("High is below open, low or close")
    if (frame["low"] - tolerance > frame[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("Low is above open, high or close")


def parse_chart(payload: dict, symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """Parse a saved chart response without fetching or using adjusted-close data."""
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo chart failed for {symbol}: {chart.get('error')}")
    result = chart["result"][0]
    if result.get("meta", {}).get("symbol", symbol).upper() != symbol.upper():
        raise ValueError("Provider returned a different symbol")
    zone = ZoneInfo(result.get("meta", {}).get("exchangeTimezoneName", "America/New_York"))

    def session_date(ts: int) -> pd.Timestamp:
        return pd.Timestamp(datetime.fromtimestamp(int(ts), timezone.utc).astimezone(zone).date())

    timestamps = result.get("timestamp", [])
    if not timestamps:
        raise ValueError(f"No timestamps for {symbol}")
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame({name: quote.get(name, []) for name in ("open", "high", "low", "close", "volume")},
                         index=pd.DatetimeIndex([session_date(t) for t in timestamps], name="date"))
    if len(frame) != len(timestamps):
        raise ValueError("Mismatched chart arrays")
    warnings: list[str] = []
    null_rows = frame.isna().any(axis=1)
    if null_rows.any():
        removed = [str(d.date()) for d in frame.index[null_rows]]
        warnings.append(f"Removed {len(removed)} vendor rows with missing OHLCV: {removed}")
        frame = frame.loc[~null_rows].copy()
    frame = frame.astype(float)
    frame["dividend"] = 0.0
    frame["split_ratio"] = 1.0
    for event in result.get("events", {}).get("dividends", {}).values():
        date = session_date(event["date"])
        if date not in frame.index:
            raise ValueError(f"Dividend event {date.date()} lacks a valid session price")
        frame.loc[date, "dividend"] += float(event["amount"])
    for event in result.get("events", {}).get("splits", {}).values():
        ratio = float(event["numerator"]) / float(event["denominator"])
        if not np.isclose(ratio, 1.0, rtol=0, atol=1e-12):
            raise ValueError(f"Unsupported {symbol} split event: {event}")
    # Causal forward accumulation: changes to a future observation cannot change
    # earlier feature levels, unlike a vendor's backward-adjusted history.
    returns = (frame["close"] + frame["dividend"]) / frame["close"].shift(1)
    returns.iloc[0] = 1.0
    frame["tr_close"] = float(frame["close"].iloc[0]) * returns.cumprod()
    validate_data(frame)
    if symbol.upper() == "SPY" and not (frame["dividend"] > 0).any():
        raise ValueError("SPY history must include at least one verified dividend event")
    return frame[list(COLUMNS)], warnings


def load_data(path: str | Path) -> pd.DataFrame:
    """Read a cached CSV and validate the same public contract used at ingestion."""
    frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
    validate_data(frame)
    return frame


def _download(symbol: str) -> tuple[bytes, str]:
    period1 = int(datetime(2004, 1, 1, tzinfo=timezone.utc).timestamp())
    period2 = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    query = urlencode({"period1": period1, "period2": period2, "interval": "1d", "events": "div,splits"})
    failures = []
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{symbol}?{query}"
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 QuantResearchEducational/1.0"})
            with urlopen(req, timeout=40) as response:
                raw = response.read()
            json.loads(raw)
            return raw, url
        except Exception as exc:
            failures.append(f"{host}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Unable to download {symbol}: {'; '.join(failures)}")


def download_all(root: str | Path, *, refresh: bool = False) -> dict:
    """Create immutable-by-default research caches and a SHA256 provenance manifest.

    ``root`` is the repository directory; data lives in ``root/data``. Existing
    caches are reused only if their hashes and parsed values pass validation.
    Refreshing is explicit because vendors can revise historical observations.
    """
    root = Path(root).resolve()
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists() and not refresh:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["datasets"]:
            for kind in ("raw", "csv"):
                path = root / item[f"{kind}_path"]
                if _sha256(path) != item[f"{kind}_sha256"]:
                    raise ValueError(f"Cached {kind} hash mismatch: {path}")
            load_data(root / item["csv_path"])
        return manifest
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "provider": "Yahoo Finance chart endpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": START,
        "requested_end_exclusive": END_EXCLUSIVE,
        "currency": "USD",
        "session_timezone": "America/New_York",
        "corporate_action_policy": (
            "Raw daily OHLC for execution; dividend field on ex-date; tr_close is a causal "
            "forward total-return close for features. Dividends are not payment-date cash. "
            "Nontrivial splits are rejected. No adjusted-close prices are used for fills."
        ),
        "warnings": [
            "One vendor only; historical values and event records may be revised.",
            "ETF selection was made retrospectively; this is not a point-in-time fund universe.",
            "Daily OHLC does not identify spreads, liquidity, auction fills or intraday order paths.",
            "Public data availability and redistribution rights must be checked before publication.",
        ],
        "datasets": [],
    }
    for symbol in SYMBOLS:
        raw, url = _download(symbol)
        fetched = datetime.now(timezone.utc).isoformat()
        raw_path = raw_dir / f"{symbol}_chart.json"
        raw_path.write_bytes(raw)
        frame, warnings = parse_chart(json.loads(raw), symbol)
        csv_path = data_dir / f"{symbol}.csv"
        frame.to_csv(csv_path, date_format="%Y-%m-%d", float_format="%.12g")
        # Confirm the shipped cache, not only the pre-serialization dataframe.
        saved = load_data(csv_path)
        manifest["datasets"].append({
            "symbol": symbol,
            "url": url,
            "fetched_utc": fetched,
            "first_session": str(saved.index.min().date()),
            "last_session": str(saved.index.max().date()),
            "rows": len(saved),
            "dividend_events": int((saved["dividend"] > 0).sum()),
            "dividend_amount_sum": float(saved["dividend"].sum()),
            "split_events": 0,
            "raw_path": str(raw_path.relative_to(root)),
            "csv_path": str(csv_path.relative_to(root)),
            "raw_sha256": _sha256(raw_path),
            "csv_sha256": _sha256(csv_path),
            "warnings": warnings,
        })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--refresh", action="store_true", help="Explicitly replace cached vendor snapshots")
    args = parser.parse_args()
    result = download_all(args.root, refresh=args.refresh)
    print(json.dumps({"provider": result["provider"], "datasets": [
        {k: row[k] for k in ("symbol", "rows", "first_session", "last_session", "dividend_events")}
        for row in result["datasets"]
    ]}, indent=2))


if __name__ == "__main__":
    main()
