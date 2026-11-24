"""Pure, budget-matched training search and disjoint validation selection.

``run_tuning`` accepts only caller-supplied pre-test price prefixes. It performs
no network, filesystem, database or UI operations and propagates callback
exceptions, allowing the orchestration layer to cancel or persist progress.
The caller owns the retrospective-study disclosure and frozen protocol.

Progress counts unique candidate evaluations plus ten validation simulations:
six finalists, half, incumbent, fixed Sigmoid and a zero-signal control using the
winner's band. Training half baselines are precomputed once and do not consume
the candidate search budget. Benchmark callbacks use method='benchmark', seed=-1.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from quantlab.engine import Costs, backtest, features
from quantlab.search import optimize
from .performance import ledger_metrics


PARAMETER_NAMES = {"mean_reversion", "momentum", "band"}
FIXED_PARAMS = {"mean_reversion": 1., "momentum": 1., "band": .05}


def _dates(value, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain start and end dates")
    try:
        start, end = (pd.Timestamp(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {name} dates") from None
    if any(pd.isna(item) or item.tz is not None or item != item.normalize() for item in (start, end)) or start > end:
        raise ValueError(f"{name} must contain ordered timezone-naive session dates")
    return start, end


def _params(value, label: str) -> dict:
    if not isinstance(value, Mapping) or set(value) != PARAMETER_NAMES:
        raise ValueError(f"{label} must contain the three Sigmoid parameters")
    chosen = {}
    for key, item in value.items():
        if isinstance(item, bool):
            raise ValueError(f"Invalid {label} parameter: {key}")
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {label} parameter: {key}") from None
        if not math.isfinite(parsed):
            raise ValueError(f"Nonfinite {label} parameter: {key}")
        chosen[key] = parsed
    if not 0 <= chosen["band"] <= 1:
        raise ValueError(f"{label} band must be between zero and one")
    return chosen


def _frame(frame, name: str, cutoff: pd.Timestamp, periods: list[tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty dataframe")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None or index.has_duplicates or index.hasnans or not index.is_monotonic_increasing:
        raise ValueError(f"{name} must have a sorted unique timezone-naive session index")
    if not (index == index.normalize()).all():
        raise ValueError(f"{name} must use daily session dates")
    if index.max() > cutoff:
        raise ValueError(f"{name} includes data after its permitted cutoff {cutoff.date()}")
    required = ["open", "close", "dividend", "tr_close"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{name} is missing price or dividend columns")
    try:
        values = frame[required].to_numpy(float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} contains nonnumeric prices") from None
    if not np.isfinite(values).all() or (values[:, [0, 1, 3]] <= 0).any() or (values[:, 2] < 0).any():
        raise ValueError(f"{name} contains invalid prices or dividends")
    if "split_ratio" in frame and not np.allclose(frame.split_ratio.to_numpy(float), 1., rtol=0, atol=1e-12):
        raise ValueError(f"{name} contains unsupported split events")
    for start, end in periods:
        if not ((index >= start) & (index <= end)).any():
            raise ValueError(f"{name} has no prices for {start.date()} to {end.date()}")


def _score_metric(metrics: dict, name: str) -> float:
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"Simulator returned a nonfinite metric: {name}")
    return value


def _compact(history: list[dict], maximum: int = 80) -> list[dict]:
    if len(history) <= maximum:
        return history
    indices = np.unique(np.linspace(0, len(history) - 1, maximum, dtype=int))
    return [history[int(index)] for index in indices]


def run_tuning(training_frame, validation_frame, protocol: dict, incumbent_params: dict,
               on_progress: callable) -> dict:
    """Train six equally budgeted candidates, then select using validation only.

    ``validation_delta`` is the validation net Sharpe minus the validation half
    baseline's net Sharpe. It is a model-selection diagnostic, not an independent
    final-test result or risk-adjusted alpha. Every finalist is a single search
    run's exact training winner. No validation result can influence a search.
    """
    if not isinstance(protocol, dict) or not callable(on_progress):
        raise ValueError("A protocol dictionary and progress callback are required")
    train_start, train_end = _dates(protocol.get("train"), "train")
    validation_start, validation_end = _dates(protocol.get("validation"), "validation")
    if validation_start <= train_end:
        raise ValueError("Validation must start strictly after the training interval")
    folds = [_dates(pair, "training fold") for pair in protocol.get("training_folds", [])]
    if not folds or any(a < train_start or b > train_end for a, b in folds):
        raise ValueError("Training folds must lie entirely inside the training interval")
    ordered = sorted(folds)
    if any(ordered[index][0] <= ordered[index - 1][1] for index in range(1, len(ordered))):
        raise ValueError("Training folds must be disjoint")
    _frame(training_frame, "training_frame", train_end, folds)
    _frame(validation_frame, "validation_frame", validation_end, [(validation_start, validation_end)])
    incumbent = _params(incumbent_params, "incumbent")
    bounds = protocol.get("bounds")
    if not isinstance(bounds, Mapping) or set(bounds) != PARAMETER_NAMES:
        raise ValueError("Bounds must define exactly the three Sigmoid parameters")
    normalized_bounds = {}
    for name, pair in bounds.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(isinstance(v, bool) for v in pair):
            raise ValueError(f"Invalid bounds for {name}")
        try:
            low, high = map(float, pair)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid bounds for {name}") from None
        if not math.isfinite(low) or not math.isfinite(high) or not math.isfinite(high - low) or low > high:
            raise ValueError(f"Invalid bounds for {name}")
        if name == "band" and not (0 <= low <= high <= 1):
            raise ValueError("Band bounds must lie inside zero to one")
        normalized_bounds[name] = (low, high)
    if not any(high > low for low, high in normalized_bounds.values()):
        raise ValueError("Search needs a nondegenerate parameter space for equal budgets")
    seeds = protocol.get("seeds")
    if not isinstance(seeds, (list, tuple)) or len(seeds) != 3 or any(
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0 for seed in seeds):
        raise ValueError("Exactly three nonnegative integer seeds are required")
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("The three configured seeds must be distinct")
    budget = protocol.get("budget_per_run")
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("budget_per_run must be a positive integer")
    budget = int(budget)
    try:
        initial_cash = float(protocol["initial_cash"])
        costs = Costs(**protocol["costs"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid initial cash or costs configuration") from None
    if isinstance(protocol["initial_cash"], bool) or not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be finite and positive")

    total = 2 * len(seeds) * budget + 10
    completed = evaluations = requests = cache_hits = 0
    finalists, histories = [], []
    train_features = features(training_frame)
    # Compute on its own provided prefix, never by slicing a feature vector that
    # was fitted or computed with inaccessible post-validation observations.
    validation_features = features(validation_frame)
    on_progress(0, total, "ga", seeds[0], 0., "train")
    baseline_rows = []
    for start, end in folds:
        baseline = backtest(training_frame, "half", {}, str(start.date()), str(end.date()), costs,
                            initial_cash, train_features).metrics
        baseline_rows.append({"start": str(start.date()), "end": str(end.date()), "metrics": baseline})
    half_sharpes = np.array([_score_metric(row["metrics"], "sharpe_rf0") for row in baseline_rows])

    for method in ("ga", "random"):
        for seed in seeds:
            best_score = None
            run_history = []

            def objective(params):
                nonlocal completed, best_score
                sharpes = []
                for start, end in folds:
                    metrics = backtest(training_frame, "sigmoid", params, str(start.date()), str(end.date()),
                                       costs, initial_cash, train_features).metrics
                    sharpes.append(_score_metric(metrics, "sharpe_rf0"))
                deltas = np.asarray(sharpes) - half_sharpes
                score = float(np.mean(deltas) - .25 * np.std(deltas, ddof=0))
                if not math.isfinite(score):
                    raise ValueError("Nonfinite candidate training score")
                best_score = score if best_score is None else max(best_score, score)
                completed += 1
                run_history.append({"evaluation": len(run_history) + 1, "score": score, "best_score": best_score})
                on_progress(completed, total, method, seed, best_score, "train")
                return score

            search = optimize(objective, normalized_bounds, method=method, budget=budget, seed=seed)
            if search["evaluations"] != budget or len(run_history) != budget:
                raise RuntimeError("Optimizer did not complete its exact unique-evaluation budget")
            params = _params(search["best_params"], "search winner")
            for key, value in params.items():
                if not normalized_bounds[key][0] <= value <= normalized_bounds[key][1]:
                    raise RuntimeError("Optimizer returned an out-of-bounds training winner")
            score = float(search["best_score"])
            if not math.isfinite(score):
                raise RuntimeError("Optimizer returned a nonfinite training winner")
            finalists.append({"params": params, "method": method, "seed": seed, "training_score": score})
            evaluations += int(search["evaluations"])
            requests += int(search["requests"])
            cache_hits += int(search["cache_hits"])
            histories.append({"method": method, "seed": seed, "budget": budget,
                              "evaluations": int(search["evaluations"]), "requests": int(search["requests"]),
                              "cache_hits": int(search["cache_hits"]), "history": _compact(run_history)})

    # All six searches have returned before the first validation simulation.
    best_training_score = max(row["training_score"] for row in finalists)

    def validation_run(kind, params, method="benchmark", seed=-1, score=None):
        nonlocal completed
        result = backtest(validation_frame, kind, params, str(validation_start.date()), str(validation_end.date()),
                          costs, initial_cash, validation_features)
        for metric in ("sharpe_rf0", "max_drawdown", "annual_turnover"):
            _score_metric(result.metrics, metric)
        completed += 1
        on_progress(completed, total, method, seed, best_training_score if score is None else score, "validation")
        return result

    half = validation_run("half", {})
    for finalist in finalists:
        simulation = validation_run("sigmoid", finalist["params"], finalist["method"], finalist["seed"], finalist["training_score"])
        metrics = ledger_metrics(simulation, half, initial_cash)
        finalist["validation"] = metrics
        finalist["validation_delta"] = float(metrics["sharpe_rf0"] - half.metrics["sharpe_rf0"])
    ranked = sorted(finalists, key=lambda row: (-row["validation"]["sharpe_rf0"],
                    row["validation"]["max_drawdown"], row["validation"]["annual_turnover"], row["method"], row["seed"]))
    winner = ranked[0]
    zero_signal = {"mean_reversion": 0., "momentum": 0., "band": winner["params"]["band"]}
    benchmarks = {name: ledger_metrics(simulation, half, initial_cash) for name, simulation in {
        "half": half,
        "incumbent": validation_run("sigmoid", incumbent),
        "fixed_sigmoid": validation_run("sigmoid", FIXED_PARAMS.copy()),
        "same_band_half": validation_run("sigmoid", zero_signal),
    }.items()}
    if completed != total or evaluations != 2 * len(seeds) * budget:
        raise AssertionError("Tuning work counts do not match the frozen budget")
    return {"winner": winner, "finalists": ranked, "benchmarks": benchmarks,
            "benchmark_params": {"incumbent": incumbent, "fixed_sigmoid": FIXED_PARAMS.copy(), "same_band_half": zero_signal},
            "training_fold_baselines": baseline_rows,
            "evaluations": evaluations, "requests": requests, "cache_hits": cache_hits,
            "histories": histories, "budget_per_run": budget, "completed": completed, "total": total,
            "objective": "mean(training fold net Sharpe minus paired half net Sharpe) - 0.25 * population std of those deltas",
            "selection_rule": "highest validation Sharpe, lower max drawdown, lower turnover, method name, seed",
            "test_used": False,
            "note": "Validation participates in selection. The same-band half control isolates a constant 50% target with the winner's band; it is not evidence from unseen test data."}
