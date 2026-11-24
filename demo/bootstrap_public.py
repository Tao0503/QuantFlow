"""Bootstrap a clean public clone without redistributing vendor market data.

Default: ``python -m demo.bootstrap_public --source synthetic``. This generates
deterministic artificial SPY/GLD/IEF fixtures, then actually runs the research
pipeline. They are software demonstration data, never historical observations.
Use ``--source yahoo`` explicitly to download a fresh vendor snapshot instead.
All work is staged; existing data, results, or demo state are never replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from quantlab.data import END_EXCLUSIVE, START, load_data, parse_chart, validate_data


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_PROVIDER = "QuantFlow deterministic synthetic fixtures v1"
SYMBOLS = ("SPY", "GLD", "IEF")
SEEDS = {"SPY": 2026112401, "GLD": 2026112402, "IEF": 2026112403}
NOTICE = (
    "合成数据软件演示：SPY、GLD、IEF 只是功能测试标签，价格、交易日和分红均由确定性程序生成，"
    "不是对应 ETF 的真实历史行情。本页数值只说明代码确实完成了计算，不能作为投资研究结论、"
    "真实市场回测或原始研究的复现结果。工作日历包含真实交易所可能休市的日期。"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def synthetic_chart(symbol: str) -> dict:
    """Return deterministic, Yahoo-shaped *artificial* daily OHLCV fixtures."""
    if symbol not in SYMBOLS:
        raise ValueError(f"Unsupported synthetic symbol: {symbol}")
    dates = pd.bdate_range(START, pd.Timestamp(END_EXCLUSIVE) - pd.Timedelta(days=1))
    size = len(dates)
    rng = np.random.default_rng(SEEDS[symbol])
    common = np.random.default_rng(2026112400).normal(size=size)
    initial, drift, volatility, correlation = {
        "SPY": (100., .00024, .009, .75),
        "GLD": (45., .00017, .010, .15),
        "IEF": (90., .00005, .0035, -.20),
    }[symbol]
    index = np.arange(size)
    noise = correlation * common + np.sqrt(1 - correlation ** 2) * rng.normal(size=size)
    varying_volatility = volatility * (1 + .35 * np.sin(index / 173.) ** 2)
    overnight = drift * .35 + noise * varying_volatility * .45
    intraday = drift * .65 + rng.normal(size=size) * varying_volatility * .70
    # Deterministic changing regimes exercise exits and volatility diagnostics.
    intraday += .0008 * np.sin(index / 97. + SYMBOLS.index(symbol))
    spread = np.abs(rng.normal(.003, .0015, size))
    timestamps = [int(day.tz_localize("UTC").timestamp()) + 17 * 3600 for day in dates]
    quote = {key: [] for key in ("open", "high", "low", "close", "volume")}
    dividends = {}
    previous = initial
    for position, day in enumerate(dates):
        distribution = 0.
        if symbol == "SPY" and day.month in (3, 6, 9, 12) and day.weekday() == 4 and 15 <= day.day <= 21:
            distribution = round(previous * .004, 6)
        elif symbol == "IEF" and day.day <= 3 and (position == 0 or dates[position - 1].month != day.month):
            distribution = round(previous * .0016, 6)
        opening = previous * np.exp(overnight[position]) - distribution
        closing = opening * np.exp(intraday[position])
        quote["open"].append(float(opening))
        quote["close"].append(float(closing))
        quote["high"].append(float(max(opening, closing) * np.exp(spread[position])))
        quote["low"].append(float(min(opening, closing) * np.exp(-spread[position])))
        quote["volume"].append(int(rng.integers(1_000_000, 25_000_001)))
        if distribution:
            timestamp = timestamps[position]
            dividends[str(timestamp)] = {"date": timestamp, "amount": distribution}
        previous = closing
    return {
        "synthetic": True, "provider": SYNTHETIC_PROVIDER, "generator_seed": SEEDS[symbol],
        "notice": NOTICE,
        "chart": {"error": None, "result": [{
            "meta": {"symbol": symbol, "exchangeTimezoneName": "America/New_York",
                     "currency": "USD", "instrumentType": "ETF", "synthetic": True,
                     "longName": f"SYNTHETIC {symbol} software fixture", "exchangeName": "SYNTHETIC"},
            "timestamp": timestamps, "indicators": {"quote": [quote]},
            "events": {"dividends": dividends, "splits": {}},
        }]},
    }


def generate_synthetic_data(root: Path) -> dict:
    """Generate fresh compatible caches; refuse every preexisting data path."""
    root = Path(root).resolve()
    folder = root / "data"
    if folder.exists() or folder.is_symlink():
        raise FileExistsError(f"Refusing existing data path: {folder}")
    (folder / "raw").mkdir(parents=True)
    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1, "provider": SYNTHETIC_PROVIDER, "synthetic": True,
        "source": "synthetic", "data_mode": "synthetic", "generator_version": 1,
        "created_utc": created, "requested_start": START, "requested_end_exclusive": END_EXCLUSIVE,
        "currency": "USD", "session_timezone": "America/New_York",
        "corporate_action_policy": "Artificial raw OHLC; fabricated ex-date distributions; causal forward total-return close; no splits.",
        "warnings": [NOTICE, "Generated for software exercises only; no vendor market data or actual exchange calendar is represented."],
        "datasets": [],
    }
    for symbol in SYMBOLS:
        raw_path, csv_path = folder / "raw" / f"{symbol}_chart.json", folder / f"{symbol}.csv"
        payload = synthetic_chart(symbol)
        write_json(raw_path, payload)
        frame, warnings = parse_chart(payload, symbol)
        validate_data(frame)
        with csv_path.open("x", encoding="utf-8", newline="") as output:
            frame.to_csv(output, date_format="%Y-%m-%d", float_format="%.12g")
        saved = load_data(csv_path)
        manifest["datasets"].append({
            "symbol": symbol, "name": f"{symbol} 合成演示数据", "synthetic": True,
            "provider": SYNTHETIC_PROVIDER, "generator_seed": SEEDS[symbol],
            "url": f"synthetic://quantflow/v1/{symbol}", "fetched_utc": created,
            "first_session": str(saved.index[0].date()), "last_session": str(saved.index[-1].date()),
            "rows": len(saved), "dividend_events": int((saved.dividend > 0).sum()),
            "dividend_amount_sum": float(saved.dividend.sum()), "split_events": 0,
            "raw_path": str(raw_path.relative_to(root)), "csv_path": str(csv_path.relative_to(root)),
            "raw_sha256": digest(raw_path), "csv_sha256": digest(csv_path),
            "warnings": [NOTICE, *warnings],
        })
    write_json(folder / "manifest.json", manifest)
    return manifest


def _assert_clean(root: Path) -> None:
    # Empty paths also count: they might be a partial setup or a symlink target.
    for relative in ("data", "results", "demo/state"):
        path = root / relative
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing existing {relative}: {path}. Use a separate clean clone; nothing was overwritten.")
    for relative in ("config.json", "quantlab/run.py", "quantlab/report.py", "demo/regime_research.py", "verify_results.py"):
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Required source file missing from clone: {relative}")


def _copy_sources(root: Path, staging: Path) -> None:
    for package in ("quantlab", "demo"):
        (staging / package).mkdir()
        for path in (root / package).glob("*.py"):
            shutil.copy2(path, staging / package / path.name)
    for name in ("config.json", "verify_results.py"):
        shutil.copy2(root / name, staging / name)


def _run(staging: Path, arguments: list[str], log) -> None:
    print("Running:", " ".join([sys.executable, *arguments]), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(staging)
    environment["MPLCONFIGDIR"] = str(staging / ".mplconfig")
    try:
        subprocess.run([sys.executable, *arguments], cwd=staging, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as exc:
        log.flush()
        tail = (staging / "results/bootstrap_public.log").read_text(encoding="utf-8")[-6000:]
        raise RuntimeError(f"Staged command failed: {' '.join(arguments)}\n{tail}") from exc
    log.flush()


def _pipeline(staging: Path, source: str) -> None:
    (staging / "results").mkdir()
    with (staging / "results/bootstrap_public.log").open("x", encoding="utf-8") as log:
        if source == "synthetic":
            generate_synthetic_data(staging)
        else:
            _run(staging, ["-m", "quantlab.data"], log)
            _run(staging, ["-m", "demo.add_asset", "IEF"], log)
        _run(staging, ["-m", "quantlab.run", "all", "--out", "results/run_001"], log)
        _run(staging, ["verify_results.py", "--run", "results/run_001"], log)
        _run(staging, ["-m", "quantlab.report", "--out", "results/run_001"], log)
        _run(staging, ["-m", "demo.regime_research", "freeze", "--out", "results/regime_study"], log)
        _run(staging, ["-m", "demo.regime_research", "evaluate", "--out", "results/regime_study"], log)


def _label_synthetic_reports(staging: Path) -> None:
    # Labels belong to this newly generated demo only, not original research.
    for relative in ("run_001/REPORT.md", "regime_study/REPORT.md"):
        path = staging / "results" / relative
        path.write_text(f"> **合成数据演示，不是真实市场研究。** {NOTICE}\n\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    dashboard = staging / "results/run_001/dashboard.html"
    dashboard.write_text(dashboard.read_text(encoding="utf-8").replace(
        "<main>", '<main><p style="padding:16px;border:2px solid #a35c20;background:#fff1da;color:#693b12">'
        '<strong>合成数据演示，不是真实市场研究。</strong> ' + html.escape(NOTICE) + '</p>', 1), encoding="utf-8")
    report_manifest_path = staging / "results/run_001/report_manifest.json"
    report_manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
    report_manifest["synthetic"] = True
    report_manifest["demo_label_added_by"] = "demo.bootstrap_public"
    for relative in report_manifest["output_sha256"]:
        report_manifest["output_sha256"][relative] = digest(staging / "results/run_001" / relative)
    report_manifest_path.write_text(json.dumps(report_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    regime_manifest_path = staging / "results/regime_study/artifact_manifest.json"
    regime_manifest = json.loads(regime_manifest_path.read_text(encoding="utf-8"))
    regime_manifest["synthetic"] = True
    regime_manifest["files_sha256"]["REPORT.md"] = digest(staging / "results/regime_study/REPORT.md")
    regime_manifest_path.write_text(json.dumps(regime_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bootstrap(root: Path = ROOT, *, source: str = "synthetic") -> dict:
    """Calculate in isolation, then publish only into absent data/results paths."""
    if source not in ("synthetic", "yahoo"):
        raise ValueError("source must be synthetic or yahoo")
    root = Path(root).resolve()
    _assert_clean(root)
    if source == "synthetic":
        print("SYNTHETIC DEMO ONLY — " + NOTICE, flush=True)
    else:
        print("Explicit Yahoo download requested; network access and current vendor availability are required.", flush=True)
    with tempfile.TemporaryDirectory(prefix=".bootstrap-public-", dir=root) as temporary:
        staging = Path(temporary)
        _copy_sources(root, staging)
        _pipeline(staging, source)
        if source == "synthetic":
            _label_synthetic_reports(staging)
        record = {"schema_version": 1, "source": source, "synthetic": source == "synthetic",
                  "provider": SYNTHETIC_PROVIDER if source == "synthetic" else "Yahoo Finance chart endpoint",
                  "created_utc": datetime.now(timezone.utc).isoformat(), "symbols": list(SYMBOLS),
                  "notice": NOTICE if source == "synthetic" else "Fresh vendor snapshot, not a reproduction of the original saved research snapshot.",
                  "original_research_reproduced": False,
                  "completed_stages": ["data", "fit", "evaluate", "independent_audit", "report", "regime_freeze", "regime_evaluate"],
                  "data_manifest_sha256": digest(staging / "data/manifest.json"),
                  "artifacts": ["results/run_001/selection.json", "results/run_001/results.json",
                                "results/run_001/dashboard.html", "results/audit.json", "results/regime_study/results.json"]}
        write_json(staging / "results/bootstrap_public.json", record)
        # Recheck immediately before publishing. A failed calculation has so far
        # left every public input/artifact path absent and can simply be retried.
        _assert_clean(root)
        os.rename(staging / "data", root / "data")
        try:
            os.rename(staging / "results", root / "results")
        except BaseException:
            # Restore our first directory if the second publish fails. No real
            # preexisting inputs can be present because both guards passed.
            os.rename(root / "data", staging / "data")
            raise
    print(f"Prepared {source} demonstration in {root}; no original research outputs were copied.", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Clean clone root containing config.json, quantlab/, demo/ and verify_results.py")
    parser.add_argument("--source", choices=("synthetic", "yahoo"), default="synthetic",
                        help="synthetic is deterministic and offline; yahoo explicitly downloads new vendor snapshots")
    args = parser.parse_args()
    try:
        record = bootstrap(args.root, source=args.source)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Bootstrap failed safely: {exc}\nExisting data/results were not overwritten.\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
