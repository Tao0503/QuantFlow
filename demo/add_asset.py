"""Import an immutable optional USD ETF snapshot for portfolio experiments.

Usage: python -m demo.add_asset IEF --name "美国7–10年期国债 ETF"

This importer reuses the frozen Yahoo downloader and parser without changing
them or data/manifest.json. New snapshots and their separate manifest live in
data/portfolio/. Existing snapshots are hash-checked and reused, never refreshed
implicitly. Prices must cover the full frozen 2022–2025 US-session calendar.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

import pandas as pd

from quantlab.data import END_EXCLUSIVE, START, _download, load_data, parse_chart


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAMES = {"IEF": "美国7–10年期国债 ETF"}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ticker(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Ticker must be a string")
    symbol = value.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,11}", symbol):
        raise ValueError("Ticker must contain 1–12 letters, digits, dots or hyphens, starting with a letter")
    if symbol in ("SPY", "GLD"):
        raise ValueError("SPY and GLD already belong to the separate frozen data manifest")
    return symbol


def _prepare(raw: bytes, symbol: str) -> tuple[pd.DataFrame, dict, list[str], list[dict]]:
    payload = json.loads(raw)
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Provider rejected {symbol}: {chart.get('error')}")
    result = chart["result"][0]
    meta = result.get("meta", {})
    if meta.get("symbol", "").upper() != symbol:
        raise ValueError("Provider ticker identity does not match the requested symbol")
    if meta.get("currency") != "USD":
        raise ValueError("Only USD-denominated instruments are supported; no currency conversion is modeled")
    if meta.get("instrumentType") != "ETF":
        raise ValueError("This optional asset catalog accepts ETFs only")
    if meta.get("exchangeTimezoneName") != "America/New_York":
        raise ValueError("Only the America/New_York US trading-session timezone is supported")
    timestamps = result.get("timestamp", [])
    if not timestamps:
        raise ValueError("Provider response has no daily prices")
    zone = ZoneInfo("America/New_York")

    def session_date(value):
        return datetime.fromtimestamp(int(value), timezone.utc).astimezone(zone).date().isoformat()

    earliest, latest = min(map(session_date, timestamps)), max(map(session_date, timestamps))
    # Yahoo can include a prior-local-date dividend at a UTC period boundary.
    # Keep the downloaded bytes verbatim; adapt only the parser input, recording
    # every excluded event. Events lacking an interior session still fail in
    # the unchanged parser instead of being silently ignored.
    prepared = copy.deepcopy(payload)
    events = prepared["chart"]["result"][0].get("events", {})
    excluded = []
    for category, entries in events.items():
        for key, event in list(entries.items()):
            event_date = session_date(event["date"])
            if event_date < earliest or event_date > latest:
                # Nontrivial splits must fail even when on a boundary.
                if category == "splits" and float(event["numerator"]) != float(event["denominator"]):
                    raise ValueError(f"Unsupported split event: {event}")
                excluded.append({"type": category, "date": event_date,
                                 "reason": "Outside the downloaded daily-session coverage", "event": event})
                del entries[key]
    frame, warnings = parse_chart(prepared, symbol)
    if excluded:
        warnings.append(f"Excluded {len(excluded)} out-of-coverage corporate-action events from parser input; exact vendor response retained")
    return frame, meta, warnings, excluded


def _coverage(root: Path, frame: pd.DataFrame) -> dict:
    reference = load_data(root / "data/SPY.csv")
    frozen = reference.loc["2022-01-01":"2025-12-31"].index
    candidate = frame.loc["2022-01-01":"2025-12-31"].index
    if frozen.empty or not frozen.equals(candidate):
        missing = frozen.difference(candidate).strftime("%Y-%m-%d").tolist()
        extra = candidate.difference(frozen).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Optional asset must cover every 2022–2025 reference session; missing={missing[:10]}, extra={extra[:10]}")
    overlap_start = max(reference.index[0], frame.index[0])
    overlap_end = min(reference.index[-1], frame.index[-1])
    if not reference.loc[overlap_start:overlap_end].index.equals(frame.loc[overlap_start:overlap_end].index):
        raise ValueError("Optional asset has inconsistent interior dates in shared historical coverage")
    return {"reference_symbol": "SPY", "reference_csv_sha256": _hash(root / "data/SPY.csv"),
            "required_start": "2022-01-01", "required_end": "2025-12-31",
            "first_session": str(frozen[0].date()), "last_session": str(frozen[-1].date()),
            "sessions": len(frozen), "all_sessions_match": True,
            "full_overlap_calendar_matches": True}


def _verified_cached(root: Path, manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("datasets"), list):
        raise ValueError("Invalid optional-asset manifest")
    seen = set()
    folder = (root / "data/portfolio").resolve()
    for item in manifest["datasets"]:
        symbol = _ticker(item["symbol"])
        if symbol in seen:
            raise ValueError("Duplicate symbols in optional-asset manifest")
        seen.add(symbol)
        paths = {}
        for kind in ("raw", "csv"):
            path = (root / item[f"{kind}_path"]).resolve()
            if folder not in path.parents or not path.is_file():
                raise ValueError(f"Invalid optional snapshot path: {path}")
            if _hash(path) != item[f"{kind}_sha256"]:
                raise ValueError(f"Immutable optional snapshot hash mismatch: {path}")
            paths[kind] = path
        parsed, _, _, _ = _prepare(paths["raw"].read_bytes(), symbol)
        saved = load_data(paths["csv"])
        if len(parsed) != len(saved) or not parsed.index.equals(saved.index):
            raise ValueError("Raw response and cached optional price sessions disagree")
        _coverage(root, saved)


def add_asset(symbol: str, *, root: Path = ROOT, name: str | None = None) -> dict:
    """Validate and append a real ETF snapshot, or verify an existing snapshot."""
    symbol = _ticker(symbol)
    root = Path(root).resolve()
    if name is not None and (not isinstance(name, str) or not 1 <= len(name.strip()) <= 100):
        raise ValueError("Display name must contain 1–100 characters")
    folder = root / "data/portfolio"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".import.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _add_locked(root, folder, symbol, name)


def _add_locked(root: Path, folder: Path, symbol: str, name: str | None) -> dict:
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _verified_cached(root, manifest)
        existing = next((row for row in manifest["datasets"] if row["symbol"] == symbol), None)
        if existing:
            return manifest
    else:
        manifest = {"schema_version": 1, "provider": "Yahoo Finance chart endpoint",
            "created_utc": datetime.now(timezone.utc).isoformat(), "datasets": [],
            "scope": "Optional interactive portfolio data; independent of the frozen run_001 data manifest",
            "requested_start": START, "requested_end_exclusive": END_EXCLUSIVE,
            "currency": "USD", "session_timezone": "America/New_York",
            "corporate_action_policy": "Raw OHLC for execution, ex-date dividends, causal forward total-return close for features; nontrivial splits rejected. Dividend cash uses the simulator's documented payment-date proxy.",
            "warnings": ["Single vendor with potentially revised historical observations.",
                         "Asset choice is retrospective; adding assets does not create unseen validation data.",
                         "Not part of the original frozen SPY/GLD experiment or a currency-hedged portfolio."]}
    raw, url = _download(symbol)
    fetched = datetime.now(timezone.utc).isoformat()
    frame, meta, warnings, excluded = _prepare(raw, symbol)
    _coverage(root, frame)
    raw_name, csv_name = f"raw/{symbol}_chart.json", f"{symbol}.csv"
    raw_path, csv_path = folder / raw_name, folder / csv_name
    if raw_path.exists() or csv_path.exists():
        raise FileExistsError("Unregistered optional snapshot files already exist; refusing to replace them")
    (folder / "raw").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".import-", dir=folder) as temporary:
        staging = Path(temporary)
        staged_raw, staged_csv = staging / "chart.json", staging / "prices.csv"
        staged_raw.write_bytes(raw)
        frame.to_csv(staged_csv, date_format="%Y-%m-%d", float_format="%.12g")
        saved = load_data(staged_csv)
        coverage = _coverage(root, saved)
        item = {"symbol": symbol, "name": (name or DEFAULT_NAMES.get(symbol) or meta.get("longName") or symbol).strip(),
            "provider_name": meta.get("longName") or meta.get("shortName"),
            "currency": meta["currency"], "exchange": meta.get("exchangeName"),
            "exchange_name": meta.get("fullExchangeName"), "session_timezone": meta["exchangeTimezoneName"],
            "instrument_type": meta["instrumentType"], "url": url, "fetched_utc": fetched,
            "csv_path": str(csv_path.relative_to(root)), "raw_path": str(raw_path.relative_to(root)),
            "csv_sha256": _hash(staged_csv), "raw_sha256": _hash(staged_raw),
            "first_session": str(saved.index[0].date()), "last_session": str(saved.index[-1].date()),
            "rows": len(saved), "dividend_events": int((saved.dividend > 0).sum()),
            "dividend_amount_sum": float(saved.dividend.sum()), "split_events": 0,
            "coverage_validation": coverage, "excluded_out_of_coverage_events": excluded, "warnings": warnings}
        updated = {**manifest, "updated_utc": fetched, "datasets": sorted([*manifest["datasets"], item], key=lambda d: d["symbol"])}
        staged_manifest = staging / "manifest.json"
        staged_manifest.write_text(json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Exclusive hard links atomically expose complete immutable snapshots;
        # publish the manifest only after both files have been fully validated.
        linked = []
        try:
            for source, destination in ((staged_raw, raw_path), (staged_csv, csv_path)):
                os.link(source, destination)
                linked.append(destination)
            os.replace(staged_manifest, manifest_path)
        except Exception:
            for destination in linked:
                destination.unlink()
            raise
    _verified_cached(root, updated)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="USD US-session ETF ticker, for example IEF")
    parser.add_argument("--name", help="Optional display name for a new snapshot")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = add_asset(args.symbol, root=args.root, name=args.name)
    selected = next(row for row in manifest["datasets"] if row["symbol"] == _ticker(args.symbol))
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
