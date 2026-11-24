"""Causal features, pure strategy decision, and a next-open ledger simulator."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    dividend_cash_lag_days: int = 35

    def __post_init__(self):
        if not math.isfinite(self.commission_bps) or self.commission_bps < 0 or not math.isfinite(self.slippage_bps) or not 0 <= self.slippage_bps < 10000:
            raise ValueError("Costs must be nonnegative and slippage below 10000 bps")
        if self.dividend_cash_lag_days < 0:
            raise ValueError("Dividend lag must be nonnegative")


class Intent(NamedTuple):
    target_weight: float | None
    signal: float | None


@dataclass
class BacktestResult:
    metrics: dict
    daily: pd.DataFrame
    trades: pd.DataFrame


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Forward total-return prices only; no fit on the full sample or backfill."""
    p = frame["tr_close"].astype(float)
    r = np.log(p).diff()
    rv21 = r.rolling(21, min_periods=21).std(ddof=1).clip(lower=1e-6)
    rv63 = r.rolling(63, min_periods=63).std(ddof=1).clip(lower=1e-6)
    ema = p.ewm(span=21, adjust=False, min_periods=21).mean()
    return pd.DataFrame({
        "deviation": (np.log(p / ema) / (rv21 * np.sqrt(21))).clip(-3, 3),
        "momentum": (np.log(p / p.shift(63)) / (rv63 * np.sqrt(63))).clip(-3, 3),
    }, index=frame.index)


def step(kind: str, deviation: float, momentum: float, current_weight: float,
         params: dict | None = None) -> Intent:
    """Pure decision at a completed close; output can only fill next session."""
    params = params or {}
    if kind == "buy_hold":
        return Intent(1.0 if current_weight < 1e-12 else None, None)
    if kind == "buy_hold_reinvest":
        return Intent(1.0 if current_weight < 1 - 1e-12 else None, None)
    if kind == "cash":
        return Intent(0.0 if current_weight > 0 else None, None)
    if kind == "half":
        target, band, signal = .5, .05, None
    elif kind == "momentum":
        if not math.isfinite(momentum):
            return Intent(None, None)
        target, band, signal = (1.0 if momentum > 0 else 0.0), .05, -momentum
    elif kind == "sigmoid":
        if not (math.isfinite(deviation) and math.isfinite(momentum)):
            return Intent(None, None)
        signal = params["mean_reversion"] * deviation - params["momentum"] * momentum
        target = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, signal))))
        band = params["band"]
    else:
        raise ValueError(f"Unknown strategy: {kind}")
    if not 0 <= band <= 1:
        raise ValueError("Invalid rebalancing band")
    return Intent(float(target) if abs(target - current_weight) >= band else None, signal)


def metrics_from_daily(daily: pd.DataFrame, trades: pd.DataFrame,
                       initial_cash: float) -> dict:
    if daily.empty:
        raise ValueError("Empty evaluation interval")
    equity = daily.equity.to_numpy(float)
    prev = np.r_[initial_cash, equity[:-1]]
    returns = equity / prev - 1
    n = len(equity)
    std = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    drawdowns = equity / np.maximum.accumulate(np.r_[initial_cash, equity])[1:] - 1
    turnover = float(trades.notional.abs().sum() / initial_cash) if len(trades) else 0.0
    return {
        "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
        "sessions": n, "initial_cash": float(initial_cash), "final_equity": float(equity[-1]),
        "total_return": float(equity[-1] / initial_cash - 1),
        "cagr": float((equity[-1] / initial_cash) ** (252 / n) - 1),
        "annual_volatility": std * math.sqrt(252),
        "sharpe_rf0": float(np.mean(returns) / std * math.sqrt(252)) if std > 1e-12 else 0.0,
        "max_drawdown": float(-np.min(drawdowns)),
        "annual_turnover": turnover * 252 / n,
        "trade_count": int(len(trades)),
        "mean_exposure": float(daily.exposure.mean()),
        "total_commission": float(daily.commission.sum()),
        "slippage_cost": float(daily.slippage_cost.sum()),
        "dividend_entitlement": float(daily.dividend_entitlement.sum()),
    }


def backtest(frame: pd.DataFrame, kind: str, params: dict | None,
             start: str, end: str, costs: Costs = Costs(), initial_cash: float = 100000,
             feature_frame: pd.DataFrame | None = None) -> BacktestResult:
    """Fresh cash each interval; features may warm up using earlier observations.

    Orders are computed at a completed close and rebalance at next observed open.
    Ex-date dividends belong to yesterday's shares. Entitlements enter NAV at
    ex-date, but cash is made spendable after a documented 35-calendar-day proxy
    lag (not a claim about actual payment dates). Terminal receivables stay in NAV.
    Whole-share, long-only, no leverage, no cash interest, no external cash flows.
    """
    if initial_cash <= 0:
        raise ValueError("Initial cash must be positive")
    f = features(frame) if feature_frame is None else feature_frame
    if not frame.index.equals(f.index):
        raise ValueError("Feature index must match prices")
    positions = np.flatnonzero((frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end)))
    if not len(positions):
        raise ValueError("No prices in requested interval")
    op = frame.open.to_numpy(float)
    cl = frame.close.to_numpy(float)
    div = frame.dividend.to_numpy(float)
    dev = f.deviation.to_numpy(float)
    mom = f.momentum.to_numpy(float)
    dates = frame.index
    cash, shares = float(initial_cash), 0
    receivables: list[tuple[pd.Timestamp, float]] = []
    pending = Intent(None, None)
    decision_date = None
    daily_rows, trade_rows = [], []
    fee_rate = costs.commission_bps / 10000
    slip = costs.slippage_bps / 10000
    first = int(positions[0])
    if first > 0:
        pending = step(kind, dev[first - 1], mom[first - 1], 0.0, params)
        decision_date = dates[first - 1]
    for i in positions:
        date = dates[i]
        paid = sum((amount for due, amount in receivables if due <= date), 0.0)
        cash += paid
        receivables = [(due, amount) for due, amount in receivables if due > date]
        entitlement = shares * div[i]
        if entitlement:
            if costs.dividend_cash_lag_days == 0:
                cash += entitlement
                paid += entitlement
            else:
                receivables.append((date + pd.Timedelta(days=costs.dividend_cash_lag_days), entitlement))
        accrued = sum((amount for _, amount in receivables), 0.0)
        commission = slip_cost = 0.0
        if pending.target_weight is not None:
            target = pending.target_weight
            open_equity = cash + shares * op[i] + accrued
            desired_value = target * open_equity
            desired_shares = max(0, math.floor(desired_value / op[i]))
            quantity = desired_shares - shares
            direction = 1 if quantity > 0 else -1
            fill = op[i] * (1 + direction * slip)
            if quantity > 0:
                quantity = min(quantity, max(0, math.floor((cash + 1e-9) / (fill * (1 + fee_rate)))))
            else:
                quantity = max(quantity, -shares)
            if quantity:
                notional = quantity * fill
                commission = abs(notional) * fee_rate
                slip_cost = abs(quantity) * op[i] * slip
                cash -= notional + commission
                shares += quantity
                if cash < -1e-6 or shares < 0:
                    raise AssertionError("Long-only cash constraint violated")
                cash = max(cash, 0.0)
                trade_rows.append({
                    "date": date, "signal_date": decision_date,
                    "quantity": quantity, "price": fill, "open_price": op[i],
                    "notional": notional, "commission": commission,
                    "slippage_cost": slip_cost, "target_weight": target,
                    "signal": pending.signal,
                })
        equity = cash + shares * cl[i] + accrued
        exposure = shares * cl[i] / equity
        daily_rows.append({"date": date, "cash": cash, "shares": shares,
                           "receivable": accrued, "close": cl[i], "equity": equity,
                           "exposure": exposure, "commission": commission,
                           "slippage_cost": slip_cost, "dividend_entitlement": entitlement,
                           "dividend_paid_proxy": paid})
        pending = step(kind, dev[i], mom[i], exposure, params)
        decision_date = date
    daily = pd.DataFrame(daily_rows).set_index("date")
    trades = pd.DataFrame(trade_rows, columns=["date", "signal_date", "quantity", "price",
        "open_price", "notional", "commission", "slippage_cost", "target_weight", "signal"])
    return BacktestResult(metrics_from_daily(daily, trades, initial_cash), daily, trades)


def training_score(frame: pd.DataFrame, feature_frame: pd.DataFrame, params: dict,
                   folds: list[list[str]], costs: Costs, initial_cash: float = 100000) -> float:
    """Only the explicitly supplied training slices enter the search objective."""
    scores = [backtest(frame, "sigmoid", params, a, b, costs, initial_cash,
                       feature_frame=feature_frame).metrics["sharpe_rf0"] for a, b in folds]
    return float(np.mean(scores) - .25 * np.std(scores))
