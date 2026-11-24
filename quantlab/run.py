"""Frozen fit/evaluate stages with all candidate histories and ledgers saved."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .data import download_all, load_data
from .engine import Costs, backtest, features, training_score
from .search import optimize

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "quantlab").glob("*.py"))
            if p.stem != "report"}


def fit(out: Path):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text())
    manifest = download_all(ROOT)
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "config": config,
              "config_sha256": digest(ROOT / "config.json"), "code_sha256": code_hashes(),
              "data_manifest_sha256": digest(ROOT / "data/manifest.json"),
              "datasets": manifest["datasets"],
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    save_json(out / "protocol_frozen.json", frozen)
    costs = Costs(**config["costs"])
    # Physically restrict the optimizer's input to training data plus earlier warmup.
    full = load_data(ROOT / f"data/{config['primary_symbol']}.csv")
    train = full.loc[:config["train"][1]].copy()
    feat_train = features(train)
    val = full.loc[:config["validation"][1]].copy()
    feat_val = features(val)
    candidates = []
    for method in ("ga", "random"):
        for seed in config["seeds"]:
            started = time.perf_counter()
            result = optimize(lambda p: training_score(train, feat_train, p,
                config["training_folds"], costs, config["initial_cash"]), config["bounds"], method=method,
                budget=config["search_budget_per_method_seed"], seed=seed)
            if result["evaluations"] != config["search_budget_per_method_seed"]:
                raise AssertionError("Unequal candidate evaluation budget")
            save_json(out / f"search_{method}_{seed}.json", result)
            v = backtest(val, "sigmoid", result["best_params"], *config["validation"], costs,
                         config["initial_cash"], feat_val)
            row = {"method": method, "seed": seed, "params": result["best_params"],
                   "training_score": result["best_score"], "validation": v.metrics,
                   "evaluations": result["evaluations"], "cache_hits": result["cache_hits"],
                   "elapsed_seconds": time.perf_counter() - started}
            candidates.append(row)
            save_json(out / "candidates_progress.json", candidates)
            print(f"{method} seed={seed} evaluations={row['evaluations']} "
                  f"train={row['training_score']:.3f} val_sharpe={v.metrics['sharpe_rf0']:.3f} "
                  f"seconds={row['elapsed_seconds']:.1f}", flush=True)
    ranked = sorted(candidates, key=lambda r: (-r["validation"]["sharpe_rf0"],
        r["validation"]["max_drawdown"], r["validation"]["annual_turnover"], r["method"], r["seed"]))
    selection = {"selected_utc": datetime.now(timezone.utc).isoformat(),
                 "selection_rule": config["selection"], "winner": ranked[0], "candidates": candidates,
                 "test_has_not_been_evaluated_by_this_pipeline": True,
                 "protocol_sha256": digest(out / "protocol_frozen.json")}
    save_json(out / "selection.json", selection)
    print("Selection frozen. Test outcomes have not been evaluated.", flush=True)


def evaluate(out: Path):
    if (out / "results.json").exists():
        raise FileExistsError("Results already exist; preserve original run and use a new output directory")
    protocol = json.loads((out / "protocol_frozen.json").read_text())
    selection = json.loads((out / "selection.json").read_text())
    if digest(out / "protocol_frozen.json") != selection["protocol_sha256"]:
        raise ValueError("Frozen protocol changed after selection")
    if code_hashes() != protocol["code_sha256"] or digest(ROOT / "config.json") != protocol["config_sha256"]:
        raise ValueError("Method code or config changed since fit; a new run is required")
    download_all(ROOT)  # Enforce cached raw and CSV hashes.
    if digest(ROOT / "data/manifest.json") != protocol["data_manifest_sha256"]:
        raise ValueError("Data manifest changed since fit")
    config = protocol["config"]
    params = selection["winner"]["params"]
    costs = Costs(**config["costs"])
    strategies = {"selected_sigmoid": ("sigmoid", params),
        "fixed_sigmoid": ("sigmoid", {"mean_reversion": 1., "momentum": 1., "band": .05}),
        "buy_hold": ("buy_hold", {}), "buy_hold_reinvest": ("buy_hold_reinvest", {}), "half": ("half", {}),
        "momentum": ("momentum", {}), "cash": ("cash", {})}
    results = {"selection_sha256": digest(out / "selection.json"), "selected_params": params,
        "selected_method": selection["winner"]["method"], "selected_seed": selection["winner"]["seed"],
        "evaluated_utc": datetime.now(timezone.utc).isoformat(), "metrics": [],
        "cost_stress": [], "parameter_sensitivity": [], "seed_diagnostics": []}
    ledgers = out / "ledgers"
    ledgers.mkdir()
    for symbol in (config["primary_symbol"], config["transfer_symbol"]):
        frame = load_data(ROOT / f"data/{symbol}.csv")
        f = features(frame)
        periods = {"train": config["train"], "validation": config["validation"], "test": config["test"]}
        for period, (start, end) in periods.items():
            for name, (kind, p) in strategies.items():
                r = backtest(frame, kind, p, start, end, costs, config["initial_cash"], f)
                results["metrics"].append({"symbol": symbol, "period": period, "strategy": name, **r.metrics})
                if period == "test" or name == "selected_sigmoid":
                    stem = f"{symbol}_{period}_{name}"
                    r.daily.to_csv(ledgers / f"{stem}_daily.csv", float_format="%.12g")
                    r.trades.to_csv(ledgers / f"{stem}_trades.csv", index=False, float_format="%.12g")
        # Predeclared diagnostics below are not used to update the frozen choice.
        for multiplier in (0., 1., 2., 5.):
            stressed = Costs(costs.commission_bps * multiplier, costs.slippage_bps * multiplier,
                             costs.dividend_cash_lag_days)
            for name in ("selected_sigmoid", "buy_hold", "half", "momentum"):
                kind, p = strategies[name]
                r = backtest(frame, kind, p, *config["test"], stressed, config["initial_cash"], f)
                results["cost_stress"].append({"symbol": symbol, "strategy": name,
                    "multiplier": multiplier, **r.metrics})
        for field in config["bounds"]:
            for scale in (.8, 1.2):
                altered = params.copy()
                altered[field] = float(np.clip(params[field] * scale, *config["bounds"][field]))
                r = backtest(frame, "sigmoid", altered, *config["test"], costs, config["initial_cash"], f)
                results["parameter_sensitivity"].append({"symbol": symbol, "field": field,
                    "scale": scale, "params": altered, **r.metrics})
        for candidate in selection["candidates"]:
            r = backtest(frame, "sigmoid", candidate["params"], *config["test"], costs,
                         config["initial_cash"], f)
            results["seed_diagnostics"].append({"symbol": symbol, "method": candidate["method"],
                "seed": candidate["seed"], **r.metrics})
    save_json(out / "results.json", results)
    pd.DataFrame(results["metrics"]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(results["cost_stress"]).to_csv(out / "cost_stress.csv", index=False)
    for r in results["metrics"]:
        if r["period"] == "test":
            print(f"{r['symbol']} {r['strategy']:<18} CAGR={r['cagr']:.2%} "
                  f"Sharpe={r['sharpe_rf0']:.3f} MDD={r['max_drawdown']:.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "evaluate", "all"))
    parser.add_argument("--out", type=Path, default=ROOT / "results/run_001")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage in ("fit", "all"):
        fit(out)
    if args.stage in ("evaluate", "all"):
        evaluate(out)


if __name__ == "__main__":
    main()
