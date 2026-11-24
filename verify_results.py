"""Read-only verification of a frozen run; only the audit JSON is written.

Usage: .venv/bin/python verify_results.py --run results/run_001
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from quantlab.engine import metrics_from_daily


ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify(run):
    checks = []
    counts = {"code_files": 0, "datasets": 0, "search_files": 0,
              "unique_evaluations": 0, "search_requests": 0, "cache_hits": 0,
              "test_ledger_pairs": 0, "test_daily_rows": 0, "test_fills": 0,
              "selected_metric_sets": 0}

    def check(name, operation):
        try:
            details = operation()
            checks.append({"name": name, "passed": True, "details": details or {}})
            return details
        except Exception as exc:
            checks.append({"name": name, "passed": False,
                           "error": f"{type(exc).__name__}: {exc}"})
            return None

    try:
        frozen = read_json(run / "protocol_frozen.json")
        selection = read_json(run / "selection.json")
        results = read_json(run / "results.json")
        manifest = read_json(ROOT / "data/manifest.json")
        config = frozen["config"]
    except Exception as exc:
        checks.append({"name": "required_run_files", "passed": False,
                       "error": f"{type(exc).__name__}: {exc}"})
        return {"passed": False, "checks": checks, "counts": counts}

    def frozen_hashes():
        current_code = {str(path.relative_to(ROOT)): digest(path)
                        for path in sorted((ROOT / "quantlab").glob("*.py"))
                        if path.stem != "report"}
        expected_code = {name: value for name, value in frozen["code_sha256"].items()
                         if Path(name).stem != "report"}
        require(current_code == expected_code, "Frozen method code hashes or file set differ")
        require(digest(ROOT / "config.json") == frozen["config_sha256"], "Config hash differs")
        require(read_json(ROOT / "config.json") == config, "Frozen and current configs differ")
        require(digest(ROOT / "data/manifest.json") == frozen["data_manifest_sha256"],
                "Data manifest hash differs")
        require(manifest["datasets"] == frozen["datasets"], "Frozen dataset records differ")
        require(digest(run / "protocol_frozen.json") == selection["protocol_sha256"],
                "Selection protocol hash differs")
        require(digest(run / "selection.json") == results["selection_sha256"],
                "Result selection hash differs")
        counts["code_files"] = len(current_code)
        return {"code_files": sorted(current_code), "report_py_excluded": True}

    check("frozen_code_config_manifest_and_hash_chain", frozen_hashes)

    for dataset in manifest["datasets"]:
        def dataset_hashes(record=dataset):
            for kind in ("raw", "csv"):
                require(digest(ROOT / record[f"{kind}_path"]) == record[f"{kind}_sha256"],
                        f"{kind} data hash differs for {record['symbol']}")
            counts["datasets"] += 1
            return {"symbol": record["symbol"], "raw_and_csv_verified": True}
        check(f"data_hashes_{dataset['symbol']}", dataset_hashes)

    candidates = selection["candidates"]
    expected_searches = {(method, int(seed)) for method in ("ga", "random") for seed in config["seeds"]}

    def candidate_selection():
        require(len(expected_searches) == 6 and len(candidates) == 6, "Expected six training candidates")
        require({(row["method"], row["seed"]) for row in candidates} == expected_searches,
                "Candidate method/seed set differs from protocol")
        for candidate in candidates:
            for metric in ("sharpe_rf0", "max_drawdown", "annual_turnover"):
                require(math.isfinite(candidate["validation"][metric]), "Nonfinite validation metric")
        ranked = sorted(candidates, key=lambda row: (-row["validation"]["sharpe_rf0"],
            row["validation"]["max_drawdown"], row["validation"]["annual_turnover"],
            row["method"], row["seed"]))
        winner = ranked[0]
        require(selection["winner"] == winner, "Winner differs from deterministic validation ranking")
        require(results["selected_params"] == winner["params"], "Selected parameters differ")
        require(results["selected_method"] == winner["method"] and results["selected_seed"] == winner["seed"],
                "Selected method/seed differ")
        require(selection["selection_rule"] == config["selection"], "Selection rule differs")
        return {"winner_method": winner["method"], "winner_seed": winner["seed"],
                "ranked_candidates": [{"method": row["method"], "seed": row["seed"]} for row in ranked]}

    check("deterministic_validation_selection", candidate_selection)

    def search_file_set():
        expected = {f"search_{method}_{seed}.json" for method, seed in expected_searches}
        require({path.name for path in run.glob("search_*.json")} == expected, "Search file set differs")
        require(config["search_budget_per_method_seed"] == 256, "Expected frozen unique budget of 256")
        return {"expected_files": sorted(expected), "unique_budget_per_search": 256}

    check("search_file_set_and_budget", search_file_set)
    for method, seed in sorted(expected_searches):
        def search_history(search_method=method, search_seed=seed):
            search = read_json(run / f"search_{search_method}_{search_seed}.json")
            history = search["history"]
            seen = {}
            hits = 0
            names = list(config["bounds"])
            for index, row in enumerate(history):
                require(set(row["params"]) == set(names), f"Parameter names differ at request {index}")
                key = tuple(row["params"][name] for name in names)
                require(all(math.isfinite(value) for value in key), "Nonfinite parameter")
                require(all(config["bounds"][name][0] <= row["params"][name] <= config["bounds"][name][1]
                            for name in names), "Parameter out of bounds")
                require(math.isfinite(row["score"]), "Nonfinite recorded score")
                require(isinstance(row["cache_hit"], bool), "Cache marker must be boolean")
                require(row["cache_hit"] == (key in seen), f"Incorrect cache marker at request {index}")
                if key in seen:
                    require(row["score"] == seen[key], f"Cached score changed at request {index}")
                    hits += 1
                else:
                    seen[key] = row["score"]
            require(len(seen) == search["evaluations"] == 256, "Unique objective-call budget differs")
            require(len(history) == search["requests"], "History/request count differs")
            require(hits == search["cache_hits"], "Cache-hit count differs")
            require(len(history) == len(seen) + hits, "Request accounting does not reconcile")
            require(search["best_score"] == max(row["score"] for row in history), "Best score differs")
            best_key = tuple(search["best_params"][name] for name in names)
            require(best_key in seen and seen[best_key] == search["best_score"], "Best parameters not found in history")
            candidate = next(row for row in candidates if (row["method"], row["seed"]) == (search_method, search_seed))
            require(candidate["params"] == search["best_params"], "Candidate/search parameters differ")
            require(candidate["training_score"] == search["best_score"], "Candidate/search scores differ")
            require(candidate["evaluations"] == 256 and candidate["cache_hits"] == hits,
                    "Candidate/search accounting differs")
            counts["search_files"] += 1
            counts["unique_evaluations"] += len(seen)
            counts["search_requests"] += len(history)
            counts["cache_hits"] += hits
            return {"evaluations": len(seen), "requests": len(history), "cache_hits": hits}
        check(f"search_history_{method}_{seed}", search_history)

    test_metrics = [row for row in results["metrics"] if row["period"] == "test"]
    symbols = (config["primary_symbol"], config["transfer_symbol"])
    strategies = ("selected_sigmoid", "fixed_sigmoid", "buy_hold", "buy_hold_reinvest", "half", "momentum", "cash")
    expected_stems = {f"{symbol}_test_{strategy}" for symbol in symbols for strategy in strategies}

    def ledger_file_set():
        require(len(test_metrics) == len(expected_stems), "Wrong number of test metric sets")
        stems = {f"{row['symbol']}_test_{row['strategy']}" for row in test_metrics}
        require(stems == expected_stems, "Test metric strategy set differs")
        for suffix in ("_daily.csv", "_trades.csv"):
            actual = {path.name.removesuffix(suffix) for path in (run / "ledgers").glob(f"*_test_*{suffix}")}
            require(actual == expected_stems, f"Test ledger set differs for {suffix}")
        require(config["costs"]["commission_bps"] == 1.0, "Expected one-basis-point base commission")
        return {"expected_pairs": len(expected_stems)}

    check("test_ledger_file_set", ledger_file_set)
    for stem in sorted(expected_stems):
        def ledger_invariants(ledger_stem=stem):
            daily = pd.read_csv(run / "ledgers" / f"{ledger_stem}_daily.csv", parse_dates=["date"]).set_index("date")
            trades = pd.read_csv(run / "ledgers" / f"{ledger_stem}_trades.csv", parse_dates=["date", "signal_date"])
            require(len(daily) > 0, "Empty daily ledger")
            require(daily.index.is_unique and daily.index.is_monotonic_increasing, "Daily dates are not ordered and unique")
            require(bool(np.isfinite(daily.select_dtypes(include="number").to_numpy()).all()), "Nonfinite daily values")
            require(bool((daily.cash >= 0).all() and (daily.shares >= 0).all()), "Negative cash or shares")
            require(bool((daily.receivable >= 0).all()), "Negative receivables")
            require(bool((daily.shares == np.floor(daily.shares)).all()), "Fractional shares in whole-share ledger")
            residual = (daily.equity - daily.cash - daily.shares * daily.close - daily.receivable).abs()
            maximum_error = float(residual.max())
            require(maximum_error <= TOLERANCE, f"Equity reconciliation error {maximum_error}")
            if len(trades):
                require(bool(np.isfinite(trades.select_dtypes(include="number").drop(columns=["signal"], errors="ignore").to_numpy()).all()),
                        "Nonfinite trade accounting values")
                require(bool((trades.date > trades.signal_date).all()), "Fill is not after signal date")
                require(bool(trades.date.isin(daily.index).all()), "Fill outside ledger dates")
                require(bool((trades.quantity != 0).all()), "Zero-quantity fill")
                require(bool((trades.loc[trades.quantity > 0, "price"] >= trades.loc[trades.quantity > 0, "open_price"]).all()),
                        "BUY filled below reference open")
                require(bool((trades.loc[trades.quantity < 0, "price"] <= trades.loc[trades.quantity < 0, "open_price"]).all()),
                        "SELL filled above reference open")
                fee_error = (trades.commission - trades.notional.abs() * .0001).abs()
                require(float(fee_error.max()) <= TOLERANCE, "Commission does not match abs(notional)*.0001")
                require(float((trades.notional - trades.quantity * trades.price).abs().max()) <= TOLERANCE,
                        "Notional differs from quantity times fill price")
            counts["test_ledger_pairs"] += 1
            counts["test_daily_rows"] += len(daily)
            counts["test_fills"] += len(trades)
            return {"daily_rows": len(daily), "fills": len(trades), "max_equity_error": maximum_error}
        check(f"ledger_invariants_{stem}", ledger_invariants)

    for symbol in symbols:
        def recompute_selected(selected_symbol=symbol):
            stem = f"{selected_symbol}_test_selected_sigmoid"
            daily = pd.read_csv(run / "ledgers" / f"{stem}_daily.csv", parse_dates=["date"]).set_index("date")
            trades = pd.read_csv(run / "ledgers" / f"{stem}_trades.csv", parse_dates=["date", "signal_date"])
            recomputed = metrics_from_daily(daily, trades, config["initial_cash"])
            matches = [row for row in test_metrics if row["symbol"] == selected_symbol and row["strategy"] == "selected_sigmoid"]
            require(len(matches) == 1, "Selected test metric set missing or duplicated")
            recorded = matches[0]
            differences = {}
            for metric, value in recomputed.items():
                if isinstance(value, (int, float)):
                    require(math.isfinite(float(value)) and math.isfinite(float(recorded[metric])), "Nonfinite metric")
                    error = abs(float(value) - float(recorded[metric]))
                    differences[metric] = error
                    require(error <= TOLERANCE, f"Recomputed {metric} differs by {error}")
                else:
                    require(value == recorded[metric], f"Recomputed {metric} differs")
            counts["selected_metric_sets"] += 1
            return {"metrics_checked": len(recomputed), "max_absolute_error": max(differences.values())}
        check(f"selected_test_metrics_{symbol}", recompute_selected)

    return {"passed": all(item["passed"] for item in checks), "checks": checks, "counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/run_001")
    parser.add_argument("--audit", type=Path, default=ROOT / "results/audit.json")
    args = parser.parse_args()
    run = args.run.resolve()
    try:
        audit = verify(run)
    except Exception as exc:
        audit = {"passed": False, "checks": [{"name": "audit_execution", "passed": False,
                 "error": f"{type(exc).__name__}: {exc}"}], "counts": {}}
    audit.update({"schema_version": 1, "audited_utc": datetime.now(timezone.utc).isoformat(),
                  "run_directory": str(run), "tolerance": TOLERANCE,
                  "verifier_sha256": digest(Path(__file__))})
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    failures = [item for item in audit["checks"] if not item["passed"]]
    print(f"{'PASS' if audit['passed'] else 'FAIL'}: {len(audit['checks']) - len(failures)}/{len(audit['checks'])} checks; audit={args.audit.resolve()}")
    for failure in failures:
        print(f"  {failure['name']}: {failure['error']}", file=sys.stderr)
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
