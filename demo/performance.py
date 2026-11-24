"""Pure full-ledger performance statistics using explicit QuantFlow conventions.

The categories are inspired by common research dashboards, including JoinQuant,
but these formulas are QuantFlow's own declared conventions. They are not an
assertion that values equal JoinQuant output or meet its submission requirements.

Inputs are aligned daily closing account equity, including cash and receivables,
with no external capital flows. Both series start from ``initial_cash``; the
first observation's return is therefore close[0]/initial_cash-1. The remaining
returns are simple close-to-close returns. No date is silently dropped, filled
or resampled. Annualization uses 252 sessions and a zero risk-free return.

Regression alpha is the annualized intercept of strategy daily returns on
benchmark daily returns, not the difference between their compound returns.
Sortino downside risk uses sqrt(mean(min(r,0)^2)*252), with all sessions in the
denominator rather than only negative sessions. Daily wins mean r>0; benchmark
outperformance counts r-b>1e-12. Calmar is strategy CAGR divided by its maximum
drawdown, measured including initial capital as the first running peak.

Undefined ratios return None (including sample statistics with fewer than two
observations or an effectively zero denominator). Standard-deviation values at
or below 1e-12 per daily return unit are treated as numerical zero. Zero observed
downside volatility itself remains a valid zero, although Sortino is undefined.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np
import pandas as pd


ANNUAL_SESSIONS = 252
DAILY_ZERO_TOLERANCE = 1e-12
OUTPERFORMANCE_TOLERANCE = 1e-12


def _series(value: pd.Series, name: str) -> np.ndarray:
    if not isinstance(value, pd.Series) or value.empty:
        raise ValueError(f"{name} must be a nonempty daily equity Series")
    index = value.index
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{name} must have sorted unique nonmissing session dates")
    if not (index == index.normalize()).all():
        raise ValueError(f"{name} must contain one midnight timestamp per daily session")
    if pd.api.types.is_bool_dtype(value.dtype):
        raise ValueError(f"{name} must contain numeric account equity, not booleans")
    try:
        values = value.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError):
        raise ValueError(f"{name} contains nonnumeric equity") from None
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{name} must contain finite positive account equity")
    return values


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _cagr(values: np.ndarray, initial_cash: float) -> float | None:
    annual_log_growth = (math.log(float(values[-1])) - math.log(initial_cash)) * ANNUAL_SESSIONS / len(values)
    try:
        return _finite(math.expm1(annual_log_growth))
    except OverflowError:
        return None


def extended_metrics(equity: pd.Series, benchmark: pd.Series, initial_cash: float) -> dict:
    """Compute nullable relative-risk metrics from full, exactly aligned ledgers.

    Returns JSON-native floats, integer ``benchmark_outperformance_days``, the
    corresponding fractional ``benchmark_outperformance_rate`` and
    None for undefined values. Misaligned/invalid inputs raise ValueError rather
    than manufacturing an apparent result by trimming or forward-filling data.
    ``daily_win_rate`` is a fraction, not a percentage or a winning-trade ratio.
    ``annual_excess_return`` is the difference between strategy and benchmark
    CAGRs; it is deliberately distinct from the regression ``alpha_rf0``.
    """
    strategy_values = _series(equity, "equity")
    benchmark_values = _series(benchmark, "benchmark")
    if not equity.index.equals(benchmark.index):
        raise ValueError("Strategy and benchmark must have exactly aligned daily session dates")
    if isinstance(initial_cash, bool):
        raise ValueError("initial_cash must be finite and positive")
    try:
        initial_cash = float(initial_cash)
    except (TypeError, ValueError):
        raise ValueError("initial_cash must be finite and positive") from None
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be finite and positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        returns = strategy_values / np.r_[initial_cash, strategy_values[:-1]] - 1.
        reference = benchmark_values / np.r_[initial_cash, benchmark_values[:-1]] - 1.
    if not np.isfinite(returns).all() or not np.isfinite(reference).all():
        raise ValueError("Equity changes overflow finite daily-return arithmetic")
    active = returns - reference
    mean_return, mean_reference = float(np.mean(returns)), float(np.mean(reference))
    annual_mean = ANNUAL_SESSIONS * mean_return
    downside_daily = float(np.sqrt(np.mean(np.minimum(returns, 0.) ** 2)))
    downside_annual = downside_daily * math.sqrt(ANNUAL_SESSIONS)
    alpha = beta = information_ratio = tracking_error = None
    if len(returns) >= 2:
        benchmark_std = float(np.std(reference, ddof=1))
        active_std = float(np.std(active, ddof=1))
        tracking_error = _finite(0. if active_std <= DAILY_ZERO_TOLERANCE else active_std * math.sqrt(ANNUAL_SESSIONS))
        if active_std > DAILY_ZERO_TOLERANCE:
            information_ratio = _finite(float(np.mean(active)) * math.sqrt(ANNUAL_SESSIONS) / active_std)
        if benchmark_std > DAILY_ZERO_TOLERANCE:
            # Centered sample covariance/variance; ddof factors cancel in beta.
            centered_benchmark = reference - mean_reference
            centered_strategy = returns - mean_return
            beta = _finite(float(np.dot(centered_strategy, centered_benchmark) / np.dot(centered_benchmark, centered_benchmark)))
            if beta is not None:
                alpha = _finite(ANNUAL_SESSIONS * (mean_return - beta * mean_reference))
    sortino = _finite(annual_mean / downside_annual) if downside_daily > DAILY_ZERO_TOLERANCE else None
    cagr, benchmark_cagr = _cagr(strategy_values, initial_cash), _cagr(benchmark_values, initial_cash)
    drawdowns = strategy_values / np.maximum.accumulate(np.r_[initial_cash, strategy_values])[1:] - 1.
    maximum_drawdown = max(0., float(-np.min(drawdowns)))
    calmar = _finite(cagr / maximum_drawdown) if cagr is not None and maximum_drawdown > DAILY_ZERO_TOLERANCE else None
    return {
        "alpha_rf0": alpha,
        "beta": beta,
        "information_ratio": information_ratio,
        "tracking_error": tracking_error,
        "sortino_rf0": sortino,
        "downside_volatility": _finite(0. if downside_daily <= DAILY_ZERO_TOLERANCE else downside_annual),
        "daily_win_rate": float(np.mean(returns > 0.)),
        "benchmark_outperformance_days": int(np.count_nonzero(active > OUTPERFORMANCE_TOLERANCE)),
        "benchmark_outperformance_rate": float(np.mean(active > OUTPERFORMANCE_TOLERANCE)),
        "annual_excess_return": _finite(cagr - benchmark_cagr) if cagr is not None and benchmark_cagr is not None else None,
        "calmar": calmar,
    }


def realized_trade_stats(trades: pd.DataFrame) -> dict:
    """Return FIFO sell-fill statistics from a complete long-only fill ledger.

    Each sell fill is one observation even if it closes several FIFO buy lots.
    Buy-lot cost includes its pro-rata buy commission; sale proceeds deduct the
    entire sell-fill commission. Slippage is already in ``price`` and is not
    deducted twice. Dividends and still-open lots are excluded. Profit factor is
    positive realized sell-fill P&L divided by the absolute negative sell-fill
    P&L; a zero gross loss makes the ratio undefined (None), not infinite.

    Separate symbols have separate FIFO queues. A legacy ledger without symbol
    is treated as a single asset. Missing historical buys, fractional quantities,
    negative fees or out-of-order dated fills raise ValueError instead of
    manufacturing unknown cost bases. A zero-P&L sale is included in the number
    of closed fills and is not a win; numerical P&L within 1e-12 is treated as zero.
    """
    if not isinstance(trades, pd.DataFrame):
        raise ValueError("trades must be a fill DataFrame")
    closed_pnls = []
    if not trades.empty:
        required = {"quantity", "price", "commission"}
        if not required.issubset(trades.columns):
            raise ValueError("Trade ledger needs quantity, price and commission")
        if "date" in trades:
            try:
                dates = pd.to_datetime(trades["date"], errors="raise")
            except (TypeError, ValueError):
                raise ValueError("Trade ledger contains invalid fill dates") from None
            if dates.isna().any() or not dates.is_monotonic_increasing:
                raise ValueError("Trade fills must be ordered by nonmissing dates")
        lots: dict[str, deque] = {}
        for row in trades.to_dict(orient="records"):
            symbol = row.get("symbol", "__single_asset__")
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Each trade symbol must be a nonempty string")
            try:
                quantity, price, commission = (float(row[key]) for key in ("quantity", "price", "commission"))
            except (TypeError, ValueError):
                raise ValueError("Trade quantities, prices and fees must be numeric") from None
            if any(isinstance(row[key], bool) for key in ("quantity", "price", "commission")) or not all(
                    math.isfinite(value) for value in (quantity, price, commission)):
                raise ValueError("Trade quantities, prices and fees must be finite")
            if quantity == 0 or abs(quantity - round(quantity)) > 1e-9 or price <= 0 or commission < 0:
                raise ValueError("Trades require nonzero whole shares, positive price and nonnegative commission")
            quantity = int(round(quantity))
            queue = lots.setdefault(symbol, deque())
            if quantity > 0:
                queue.append([quantity, price + commission / quantity])
                continue
            sold = -quantity
            if sold > sum(lot[0] for lot in queue):
                raise ValueError(f"Sell inventory exceeds known FIFO purchases: {symbol}")
            net_sale_per_share = price - commission / sold
            remaining = sold
            pnl_parts = []
            while remaining:
                lot = queue[0]
                matched = min(remaining, lot[0])
                pnl_parts.append(matched * (net_sale_per_share - lot[1]))
                remaining -= matched
                lot[0] -= matched
                if not lot[0]:
                    queue.popleft()
            pnl = math.fsum(pnl_parts)
            if not math.isfinite(pnl):
                raise ValueError("Realized trade P&L overflowed")
            closed_pnls.append(0. if abs(pnl) <= 1e-12 else pnl)
    gains = math.fsum(pnl for pnl in closed_pnls if pnl > 0)
    losses = -math.fsum(pnl for pnl in closed_pnls if pnl < 0)
    count = len(closed_pnls)
    return {
        "closed_fill_count": count,
        "closed_fill_win_rate": sum(pnl > 0 for pnl in closed_pnls) / count if count else None,
        "realized_profit_factor": _finite(gains / losses) if losses > 0 else None,
        "realized_gains": float(gains),
        "realized_losses": float(losses),
        "realized_net_pnl": float(math.fsum(closed_pnls)),
    }


def ledger_metrics(result, benchmark, initial_cash: float) -> dict:
    """Combine simulator metrics with full-ledger and realized trade statistics."""
    return {**result.metrics, **extended_metrics(result.daily.equity, benchmark.daily.equity, initial_cash),
            **realized_trade_stats(result.trades)}
