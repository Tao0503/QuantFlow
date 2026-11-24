"""Compare two unchanged runs and save exact repeatability and test evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=Path("results/run_001"))
    parser.add_argument("--second", type=Path, default=Path("results/run_002"))
    args = parser.parse_args()
    a, b = args.first, args.second
    checks = {}
    search_paths = sorted(a.glob("search_*.json"))
    for path in search_paths:
        checks[f"exact_search_history_{path.stem}"] = read(path) == read(b / path.name)
    select_a, select_b = read(a / "selection.json"), read(b / "selection.json")
    for key in ("method", "seed", "params", "training_score", "validation"):
        checks[f"exact_selection_{key}"] = select_a["winner"][key] == select_b["winner"][key]
    result_a, result_b = read(a / "results.json"), read(b / "results.json")
    for key in ("metrics", "cost_stress", "parameter_sensitivity", "seed_diagnostics", "selected_params"):
        checks[f"exact_results_{key}"] = result_a[key] == result_b[key]
    ledgers = sorted((a / "ledgers").glob("*.csv"))
    checks["all_ledger_files_byte_identical"] = all(sha(p) == sha(b / "ledgers" / p.name) for p in ledgers)
    checks["ledger_file_set_identical"] = {p.name for p in ledgers} == {p.name for p in (b / "ledgers").glob("*.csv")}
    process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                             capture_output=True, text=True)
    log = a.parent / "tests.log"
    log.write_text(process.stdout + process.stderr, encoding="utf-8")
    checks["tests_pass"] = process.returncode == 0
    value = {"checked_utc": datetime.now(timezone.utc).isoformat(), "passed": all(checks.values()),
             "checks": checks, "search_runs_compared": len(search_paths),
             "ledger_files_compared": len(ledgers), "test_log": str(log),
             "note": "Same-data rerun verifies computational reproducibility; it creates no new independent holdout evidence."}
    path = a.parent / "verification.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
