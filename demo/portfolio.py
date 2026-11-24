"""Multi-asset, whole-share backtesting with one shared cash ledger.

This module is separate from the frozen single-asset research implementation.
Allocation weights are maximum sleeves, not independently funded accounts:
``target[symbol] = allocation[symbol] * strategy_target[symbol]``. The sum of
allocations cannot exceed one; the unused allocation remains cash. A half
strategy therefore targets half of each sleeve, not half of each asset's price.

Features are calculated on each full history before common-calendar alignment.
Within overlapping coverage, calendars must agree exactly: missing interior
sessions are errors, not silently forward-filled or removed. Leading history
before the latest listing date can still warm up the older asset's features.
Both strategies wait until all assets have finite feature values. A close that
crosses any sleeve's scaled band schedules the complete target vector for the
next shared session's open. Every quantity uses the same pre-trade open NAV.

Sells execute first. If cash cannot fund all requested purchases including fees,
purchase quantities are scaled by the same available-cash/requested-cost ratio
and rounded down. Remaining cash buys at most one extra share per symbol, in
descending fractional-share remainder order, ties by symbol; unaffordable shares
are skipped and all residual cash is retained. No symbol receives first access
to the whole cash balance merely because it appears first in the input mapping.

Dividends belong to yesterday's shares before open trades. They enter receivable
NAV on ex-date and become cash after Costs.dividend_cash_lag_days calendar days,
matching the original simulator's documented payment-date proxy. Terminal
receivables stay in NAV. There is no leverage, cash interest, tax or inflow.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from quantlab.engine import BacktestResult, Costs, features, metrics_from_daily, step


TRADE_COLUMNS = ["date", "signal_date", "symbol", "quantity", "price", "open_price",
                 "notional", "commission", "slippage_cost", "target_weight", "signal"]


def _validated_inputs(frames, weights, kind, params, start, end, initial_cash):
    if not isinstance(frames, Mapping) or not frames:
        raise ValueError("frames must be a nonempty symbol-to-dataframe mapping")
    if not isinstance(weights, Mapping) or set(frames) != set(weights):
        raise ValueError("frames and allocation weights must contain the same symbols")
    if not all(isinstance(s, str) and s.strip() for s in frames):
        raise ValueError("Symbols must be nonempty strings")
    allocations = {}
    for symbol, value in weights.items():
        if isinstance(value, bool):
            raise ValueError("Allocation weights must be finite numbers")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("Allocation weights must be finite numbers") from None
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Allocation weights must be between zero and one")
        allocations[symbol] = value
    total = math.fsum(allocations.values())
    if total <= 0 or total > 1 + 1e-12:
        raise ValueError("Positive allocation weights must sum to at most one")
    # Remove only machine-rounding overshoot, never economic overallocation.
    if total > 1:
        allocations = {symbol: weight / total for symbol, weight in allocations.items()}
    try:
        cash = float(initial_cash)
    except (TypeError, ValueError):
        raise ValueError("Initial cash must be finite and positive") from None
    if isinstance(initial_cash, bool) or not math.isfinite(cash) or cash <= 0:
        raise ValueError("Initial cash must be finite and positive")
    if kind not in ("sigmoid", "half"):
        raise ValueError("Portfolio strategy must be sigmoid or half")
    chosen = dict(params or {})
    if kind == "sigmoid":
        for key in ("mean_reversion", "momentum", "band"):
            try:
                value = float(chosen[key])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Missing or invalid sigmoid parameter: {key}") from None
            if isinstance(chosen[key], bool) or not math.isfinite(value):
                raise ValueError(f"Sigmoid parameter must be finite: {key}")
            chosen[key] = value
        if not 0 <= chosen["band"] <= 1:
            raise ValueError("Sigmoid band must be between zero and one")
    try:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    except (TypeError, ValueError):
        raise ValueError("Invalid evaluation dates") from None
    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts.tz is not None or end_ts.tz is not None or start_ts > end_ts:
        raise ValueError("Evaluation dates must be ordered timezone-naive dates")
    symbols = sorted(frames)
    for symbol in symbols:
        frame = frames[symbol]
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"Empty or invalid frame: {symbol}")
        idx = frame.index
        if not isinstance(idx, pd.DatetimeIndex) or idx.tz is not None or idx.hasnans or idx.has_duplicates or not idx.is_monotonic_increasing:
            raise ValueError(f"Invalid sorted unique session index: {symbol}")
        if not (idx == idx.normalize()).all():
            raise ValueError(f"Daily session timestamps must be midnight: {symbol}")
        required = ["open", "close", "dividend", "tr_close"]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"Missing price or dividend columns: {symbol}")
        try:
            numeric = frame[required].to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"Nonnumeric price or dividend data: {symbol}") from None
        if not np.isfinite(numeric).all() or (numeric[:, [0, 1, 3]] <= 0).any() or (numeric[:, 2] < 0).any():
            raise ValueError(f"Invalid price or dividend values: {symbol}")
        if "split_ratio" in frame and not np.allclose(frame.split_ratio.to_numpy(float), 1, rtol=0, atol=1e-12):
            raise ValueError(f"Splits require explicit share accounting: {symbol}")
    overlap_start = max(frames[s].index[0] for s in symbols)
    overlap_end = min(frames[s].index[-1] for s in symbols)
    if overlap_start > overlap_end:
        raise ValueError("Asset histories have no overlapping coverage")
    shared = frames[symbols[0]].loc[overlap_start:overlap_end].index
    for symbol in symbols[1:]:
        if not shared.equals(frames[symbol].loc[overlap_start:overlap_end].index):
            raise ValueError(f"Inconsistent interior trading calendar: {symbol}")
    positions = np.flatnonzero((shared >= start_ts) & (shared <= end_ts))
    if not len(positions):
        raise ValueError("No shared prices in requested evaluation interval")
    return symbols, allocations, chosen, cash, shared, positions


def _cash_limited_buys(requested: dict[str, int], unit_costs: dict[str, float], cash: float) -> dict[str, int]:
    """Proportional funding followed by a bounded largest-remainder pass."""
    total_cost = math.fsum(quantity * unit_costs[symbol] for symbol, quantity in requested.items())
    if total_cost <= cash:
        return dict(requested)
    if total_cost <= 0 or cash <= 0:
        return {symbol: 0 for symbol in requested}
    ratio = cash / total_cost
    ideals = {symbol: quantity * ratio for symbol, quantity in requested.items()}
    bought = {symbol: min(requested[symbol], math.floor(ideal)) for symbol, ideal in ideals.items()}
    remainder_cash = cash - math.fsum(bought[s] * unit_costs[s] for s in bought)
    ranked = sorted(requested, key=lambda s: (-(ideals[s] - bought[s]), s))
    for symbol in ranked:
        if bought[symbol] < requested[symbol] and unit_costs[symbol] <= remainder_cash:
            bought[symbol] += 1
            remainder_cash -= unit_costs[symbol]
    return bought


def backtest_portfolio(frames: Mapping[str, pd.DataFrame], weights: Mapping[str, float],
                       kind: str, params: dict | None, start: str, end: str,
                       costs: Costs = Costs(), initial_cash: float = 100000) -> BacktestResult:
    """Return a genuine shared-cash portfolio ledger, trades and aggregate metrics.

    ``daily.holdings`` maps symbols to shares, close, market_value and weight.
    All metric-required scalar columns match the single-asset result. For legacy
    consumers ``daily.shares`` is the sum of units across assets (not a tradable
    instrument), and ``daily.close`` is equity/initial_cash (a normalized NAV),
    so actual prices must always be read from the nested holdings.
    """
    symbols, allocations, chosen, cash, dates, positions = _validated_inputs(
        frames, weights, kind, params, start, end, initial_cash)
    initial_cash = cash
    if not isinstance(costs, Costs):
        raise ValueError("costs must be an engine.Costs instance")
    if costs.commission_bps > 10000 or not math.isfinite(float(costs.dividend_cash_lag_days)):
        raise ValueError("Unsupported commission or dividend cash lag")
    # Full per-asset feature history is retained before trimming listing overlap.
    aligned_features = {s: features(frames[s]).loc[dates] for s in symbols}
    opens = {s: frames[s].loc[dates, "open"].to_numpy(float) for s in symbols}
    closes = {s: frames[s].loc[dates, "close"].to_numpy(float) for s in symbols}
    dividends = {s: frames[s].loc[dates, "dividend"].to_numpy(float) for s in symbols}
    deviations = {s: aligned_features[s].deviation.to_numpy(float) for s in symbols}
    momenta = {s: aligned_features[s].momentum.to_numpy(float) for s in symbols}
    shares = {s: 0 for s in symbols}
    receivables: list[tuple[pd.Timestamp, float]] = []
    fee_rate, slip = costs.commission_bps / 10000, costs.slippage_bps / 10000
    band = .05 if kind == "half" else chosen["band"]
    pending = None
    signal_date = None
    daily_rows, trade_rows = [], []

    def decide(index: int, equity: float):
        if not all(math.isfinite(deviations[s][index]) and math.isfinite(momenta[s][index]) for s in symbols):
            return None
        vector, signals = {}, {}
        for symbol in symbols:
            allocation = allocations[symbol]
            current = shares[symbol] * closes[symbol][index] / equity
            if kind == "half":
                # raw half target exactly .5; step's existing band must not
                # suppress targets before the portfolio-level trigger check.
                intent = step("half", deviations[symbol][index], momenta[symbol][index], 0.0, chosen)
            else:
                raw_params = {**chosen, "band": 0.0}
                sleeve_weight = current / allocation if allocation > 0 else 0.0
                intent = step("sigmoid", deviations[symbol][index], momenta[symbol][index], sleeve_weight, raw_params)
            target = allocation * float(intent.target_weight)
            if not math.isfinite(target) or not 0 <= target <= allocation:
                raise ValueError(f"Invalid target allocation from strategy: {symbol}")
            vector[symbol], signals[symbol] = target, intent.signal
        trigger = any(allocations[s] > 0 and
                      abs(vector[s] - shares[s] * closes[s][index] / equity) >= band * allocations[s]
                      for s in symbols)
        return (vector, signals) if trigger else None

    first = int(positions[0])
    if first > 0:
        pending = decide(first - 1, cash)
        signal_date = dates[first - 1]

    for index in positions:
        session = dates[index]
        paid = math.fsum(amount for due, amount in receivables if due <= session)
        cash += paid
        receivables = [(due, amount) for due, amount in receivables if due > session]
        entitlement = math.fsum(shares[s] * dividends[s][index] for s in symbols)
        if entitlement:
            if costs.dividend_cash_lag_days == 0:
                cash += entitlement
                paid += entitlement
            else:
                receivables.append((session + pd.Timedelta(days=costs.dividend_cash_lag_days), entitlement))
        accrued = math.fsum(amount for _, amount in receivables)
        commission = slippage_cost = 0.0
        if pending is not None:
            targets, signals = pending
            # This reference is intentionally frozen before any sell or buy.
            open_equity = cash + accrued + math.fsum(shares[s] * opens[s][index] for s in symbols)
            if not math.isfinite(open_equity) or open_equity <= 0:
                raise ValueError("Nonpositive or nonfinite pre-trade portfolio NAV")
            desired = {s: max(0, math.floor(targets[s] * open_equity / opens[s][index])) for s in symbols}

            def execute(symbol: str, quantity: int):
                nonlocal cash, commission, slippage_cost
                if quantity == 0:
                    return
                direction = 1 if quantity > 0 else -1
                opening = opens[symbol][index]
                fill = opening * (1 + direction * slip)
                notional = quantity * fill
                fee = abs(notional) * fee_rate
                adverse = abs(quantity) * opening * slip
                cash -= notional + fee
                shares[symbol] += quantity
                if cash < -max(1e-7, initial_cash * 1e-12) or shares[symbol] < 0:
                    raise AssertionError("Shared-cash or long-only inventory constraint violated")
                cash = max(0.0, cash)
                commission += fee
                slippage_cost += adverse
                trade_rows.append({"date": session, "signal_date": signal_date, "symbol": symbol,
                    "quantity": quantity, "price": fill, "open_price": opening, "notional": notional,
                    "commission": fee, "slippage_cost": adverse, "target_weight": targets[symbol],
                    "signal": signals[symbol]})

            for symbol in symbols:
                if desired[symbol] < shares[symbol]:
                    execute(symbol, desired[symbol] - shares[symbol])
            requested = {s: desired[s] - shares[s] for s in symbols if desired[s] > shares[s]}
            unit_costs = {s: opens[s][index] * (1 + slip) * (1 + fee_rate) for s in requested}
            buys = _cash_limited_buys(requested, unit_costs, cash)
            for symbol in sorted(buys):
                execute(symbol, buys[symbol])

        market_values = {s: shares[s] * closes[s][index] for s in symbols}
        equity = cash + accrued + math.fsum(market_values.values())
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("Nonpositive or nonfinite closing portfolio NAV")
        holdings = {s: {"shares": int(shares[s]), "close": float(closes[s][index]),
                        "market_value": float(market_values[s]), "weight": float(market_values[s] / equity)} for s in symbols}
        daily_rows.append({"date": session, "cash": cash, "shares": sum(shares.values()),
            "receivable": accrued, "close": equity / initial_cash, "equity": equity,
            "exposure": math.fsum(market_values.values()) / equity, "commission": commission,
            "slippage_cost": slippage_cost, "dividend_entitlement": entitlement,
            "dividend_paid_proxy": paid, "holdings": holdings})
        pending = decide(int(index), equity)
        signal_date = session

    daily = pd.DataFrame(daily_rows).set_index("date")
    trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
    metrics = metrics_from_daily(daily, trades, initial_cash)
    return BacktestResult(metrics, daily, trades)
