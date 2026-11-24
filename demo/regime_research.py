"""A fixed, retrospective ETF trend/volatility study with a frozen protocol.

No optimizer is used. The desired weight is 0 below/equal to a causal EMA200,
otherwise min(1, 10% / annualized trailing-63-session log-return volatility).
Signals are transformed into the frozen simulator's inverse-logistic feature
contract, so all fills, cash, dividends and costs use its unchanged ledger.

CLI: python -m demo.regime_research freeze|evaluate --out results/regime_study
The freeze stage never computes performance. The evaluate stage refuses to
overwrite results or run after code/data/protocol changes. This is a current
retrospective specification, not a historical preregistration or unseen-market
experiment: the prior strategy's 2022–2025 outcomes were already inspected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from quantlab.data import load_data
from quantlab.engine import BacktestResult, Costs, backtest


ROOT = Path(__file__).resolve().parents[1]
STRATEGY = "ema200_vol10"
SPECIFICATION = {
    "study_id": "fixed_ema200_vol10_retrospective_v1",
    "strategy_id": STRATEGY,
    "assets": ["SPY", "GLD"],
    "periods": {"train": ["2010-01-01", "2018-12-31"],
                "validation": ["2019-01-01", "2021-12-31"],
                "test": ["2022-01-01", "2025-12-31"]},
    "strategy": {"ema_span": 200, "ema_adjust": False, "ema_min_periods": 200,
                 "volatility_window": 63, "volatility_ddof": 1,
                 "annual_sessions": 252, "annual_volatility_target": .10,
                 "annual_volatility_floor": 1e-6, "max_weight": 1.,
                 "rebalance_band": .05,
                 "formula": "w=0 if tr_close<=EMA200; else min(1,0.10/max(std63(log-return)*sqrt252,1e-6))",
                 "nonready": "Hold current inventory; initial state is cash",
                 "execution": "Completed-close target, next observed session open; unchanged quantlab.engine ledger",
                 "adapter": "deviation=log((1-w)/w), momentum=0, a=1,b=0; endpoint signals +60/-60; w=0 rounds to zero whole shares",
                 "exit_band": "Band also applies to zero targets; residual exposure smaller than five percentage points may persist"},
    "initial_cash": 100000,
    "costs": {"commission_bps": 1., "slippage_bps": 5., "dividend_cash_lag_days": 35},
    "benchmarks": {"half": "50% asset target with five-percentage-point band",
                   "buy_hold_reinvest": "Buy-and-hold, distributions reinvested after the same proxy payout lag"},
    "bootstrap": {"method": "paired_moving_block", "block_length": 20,
                  "replications": 5000, "confidence": .95,
                  "seeds": {"SPY": 20261123, "GLD": 20261124},
                  "sampling": "Overlapping noncircular blocks sampled with replacement; concatenate/truncate to n; same indices for both return series",
                  "estimator": "Annualized mean/simple-daily-return sample-std Sharpe difference; rf=0; percentile interval",
                  "scope": "Conditional descriptive uncertainty of already-net daily returns; no strategy re-fitting or re-execution in bootstrap"},
    "evidence_gate": {"benchmark": "half", "minimum_sharpe_delta": .20,
                      "ci_lower_strictly_greater_than": 0., "all_assets_required": True},
    "optimization_performed": False,
    "retrospective": True,
    "prior_outcomes_known": "Original Sigmoid and baseline outcomes over 2022–2025 were already inspected; this rule is now fixed without searching these outcomes",
    "selection": "No grid, hyperparameter fitting or winner selection on any period; train/validation are descriptive only",
    "post_evaluation_policy": "No parameter changes after this result; failures remain failures. Any later study is a new version on already-seen history, not new independent evidence.",
    "cash_interest": 0., "sharpe_risk_free_rate": 0.,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_exclusive(path: Path, value) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical(value))


def regime_features(frame: pd.DataFrame, strategy: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causal target series and a feature adapter for the unchanged ledger."""
    config = SPECIFICATION["strategy"] if strategy is None else strategy
    price = frame["tr_close"].astype(float)
    if price.empty or not np.isfinite(price.to_numpy()).all() or (price <= 0).any():
        raise ValueError("A finite positive total-return close series is required")
    ema = price.ewm(span=config["ema_span"], adjust=config["ema_adjust"],
                    min_periods=config["ema_min_periods"]).mean()
    log_returns = np.log(price).diff()
    vol = log_returns.rolling(config["volatility_window"], min_periods=config["volatility_window"]).std(
        ddof=config["volatility_ddof"]) * math.sqrt(config["annual_sessions"])
    ready = ema.notna() & vol.notna()
    raw = (config["annual_volatility_target"] / vol.clip(lower=config["annual_volatility_floor"])).clip(upper=config["max_weight"])
    target = raw.where(price > ema, 0.).where(ready)
    signal = pd.Series(np.nan, index=price.index, dtype=float)
    middle = ready & target.gt(0) & target.lt(1)
    signal.loc[middle] = np.log1p(-target.loc[middle]) - np.log(target.loc[middle])
    signal.loc[ready & target.le(0)] = 60.
    signal.loc[ready & target.ge(1)] = -60.
    adapted = pd.DataFrame({"deviation": signal, "momentum": 0.}, index=price.index)
    diagnostics = pd.DataFrame({"tr_close": price, "ema200": ema, "annual_volatility": vol,
                                "target_weight": target, "ready": ready}, index=price.index)
    return adapted, diagnostics


def backtest_regime(frame: pd.DataFrame, start: str, end: str, costs: Costs = Costs(),
                    initial_cash: float = 100000, strategy: dict | None = None) -> BacktestResult:
    config = SPECIFICATION["strategy"] if strategy is None else strategy
    adapted, _ = regime_features(frame, config)
    return backtest(frame, "sigmoid", {"mean_reversion": 1., "momentum": 0.,
                    "band": config["rebalance_band"]}, start, end, costs, initial_cash, adapted)


def _sharpe(returns: np.ndarray, axis=None):
    std = np.std(returns, ddof=1, axis=axis)
    mean = np.mean(returns, axis=axis)
    return np.divide(mean * math.sqrt(252), std, out=np.zeros_like(mean, dtype=float), where=std > 1e-12)


def paired_block_bootstrap(strategy_returns, benchmark_returns, *, block_length=20,
                           replications=5000, seed=0, confidence=.95) -> dict:
    """Paired overlapping moving-block percentile CI for net Sharpe difference.

    The same sampled session indices are applied to both strategies, preserving
    their contemporaneous dependence and dependence within each chosen block.
    It assumes a sufficiently stable return process, underweights endpoints,
    and neither removes retrospective model choice nor measures future alpha.
    """
    a, b = np.asarray(strategy_returns, float), np.asarray(benchmark_returns, float)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Return series must be finite, paired, one-dimensional arrays")
    if isinstance(block_length, bool) or not isinstance(block_length, int) or not 2 <= block_length <= len(a):
        raise ValueError("Block length must be an integer between 2 and the paired sample length")
    if isinstance(replications, bool) or not isinstance(replications, int) or replications < 100:
        raise ValueError("At least 100 bootstrap replications are required")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    blocks = math.ceil(len(a) / block_length)
    deltas = np.empty(replications, dtype=float)
    offsets = np.arange(block_length)
    # Batching limits memory to about 128 * n paired observations.
    for begin in range(0, replications, 128):
        count = min(128, replications - begin)
        starts = rng.integers(0, len(a) - block_length + 1, size=(count, blocks))
        indices = (starts[:, :, None] + offsets).reshape(count, -1)[:, :len(a)]
        deltas[begin:begin + count] = _sharpe(a[indices], axis=1) - _sharpe(b[indices], axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(deltas, [tail, 1 - tail])
    return {"method": "paired_moving_block", "block_length": block_length,
            "replications": replications, "seed": seed, "confidence": confidence, "sessions": len(a),
            "observed_sharpe_delta": float(_sharpe(a) - _sharpe(b)),
            "ci_lower": float(lower), "ci_upper": float(upper),
            "bootstrap_delta_median": float(np.median(deltas))}


def _snapshot_hashes(root: Path) -> dict:
    paths = [root / "demo/regime_research.py", root / "quantlab/engine.py", root / "quantlab/data.py",
             root / "data/manifest.json"]
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    for symbol in SPECIFICATION["assets"]:
        entry = next(item for item in manifest["datasets"] if item["symbol"] == symbol)
        for kind in ("csv", "raw"):
            path = root / entry[f"{kind}_path"]
            if digest(path) != entry[f"{kind}_sha256"]:
                raise ValueError(f"Original data snapshot changed: {path}")
            paths.append(path)
    return {str(path.relative_to(root)): digest(path) for path in paths}


def freeze(root: Path = ROOT, out: Path | None = None) -> dict:
    """Persist method and data hashes without computing any performance."""
    root = Path(root).resolve()
    out = Path(out or root / "results/regime_study").resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing regime study")
    hashes = _snapshot_hashes(root)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"schema_version": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "specification": SPECIFICATION, "input_sha256": hashes,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "performance_computed_during_freeze": False}
    _write_exclusive(out / "protocol_frozen.json", protocol)
    return protocol


def _net_returns(result: BacktestResult, initial_cash: float) -> np.ndarray:
    equity = result.daily.equity.to_numpy(float)
    return equity / np.r_[initial_cash, equity[:-1]] - 1


def evaluate(root: Path = ROOT, out: Path | None = None) -> dict:
    root = Path(root).resolve()
    out = Path(out or root / "results/regime_study").resolve()
    protocol_path = out / "protocol_frozen.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (out / "results.json").exists() or (out / "evaluation_started.json").exists():
        raise FileExistsError("Evaluation was already started; do not rerun or overwrite this observed-history study")
    if protocol["specification"] != SPECIFICATION or protocol["input_sha256"] != _snapshot_hashes(root):
        raise ValueError("Frozen method or data changed; preserve this study and create an explicitly new version")
    protocol_hash = digest(protocol_path)
    _write_exclusive(out / "evaluation_started.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                     "protocol_sha256": protocol_hash, "note": "Exclusive marker prevents silent reruns even after interruption"})
    spec = protocol["specification"]
    costs = Costs(**spec["costs"])
    ledger_dir = out / "ledgers"
    ledger_dir.mkdir()
    all_metrics, assets = [], []
    for symbol in spec["assets"]:
        frame = load_data(root / f"data/{symbol}.csv")
        period_results = {}
        test_results = {}
        paths = {}
        for period, (start, end) in spec["periods"].items():
            # Only causal warmup plus the current period can reach its adapter.
            available = frame.loc[:end].copy()
            strategy_result = backtest_regime(available, start, end, costs, spec["initial_cash"], spec["strategy"])
            runs = {STRATEGY: strategy_result,
                    "half": backtest(available, "half", {}, start, end, costs, spec["initial_cash"]),
                    "buy_hold_reinvest": backtest(available, "buy_hold_reinvest", {}, start, end, costs, spec["initial_cash"])}
            period_results[period] = {name: run.metrics for name, run in runs.items()}
            for name, run in runs.items():
                all_metrics.append({"symbol": symbol, "period": period, "strategy": name, **run.metrics})
                if period == "test":
                    test_results[name] = run
                    daily_path = ledger_dir / f"{symbol}_{name}_daily.csv"
                    trades_path = ledger_dir / f"{symbol}_{name}_trades.csv"
                    run.daily.to_csv(daily_path, float_format="%.12g")
                    run.trades.to_csv(trades_path, index=False, float_format="%.12g")
                    paths[name] = {"daily": str(daily_path.relative_to(out)), "trades": str(trades_path.relative_to(out))}
        chosen, half = test_results[STRATEGY], test_results["half"]
        if not chosen.daily.index.equals(half.daily.index):
            raise AssertionError("Bootstrap series must share the exact same sessions")
        bootstrap_config = spec["bootstrap"]
        bootstrap = paired_block_bootstrap(_net_returns(chosen, spec["initial_cash"]), _net_returns(half, spec["initial_cash"]),
            block_length=bootstrap_config["block_length"], replications=bootstrap_config["replications"],
            seed=bootstrap_config["seeds"][symbol], confidence=bootstrap_config["confidence"])
        delta = chosen.metrics["sharpe_rf0"] - half.metrics["sharpe_rf0"]
        reasons = []
        if delta < spec["evidence_gate"]["minimum_sharpe_delta"]:
            reasons.append("Sharpe difference is below the fixed 0.20 economic-size threshold")
        if bootstrap["ci_lower"] <= spec["evidence_gate"]["ci_lower_strictly_greater_than"]:
            reasons.append("Paired 95% moving-block interval includes zero or negative values")
        assets.append({"symbol": symbol, "periods": period_results,
                       "test_comparison": {"benchmark": "half", "sharpe_delta": delta,
                                           "bootstrap": bootstrap, "passes_evidence_gate": not reasons, "reasons": reasons},
                       "test_ledger_paths": paths})
    result = {"schema_version": 1, "study_id": spec["study_id"], "strategy_id": STRATEGY,
              "protocol_sha256": protocol_hash, "evaluated_utc": datetime.now(timezone.utc).isoformat(),
              "retrospective": True, "optimization_performed": False,
              "evidence_gate": {"min_sharpe_delta": .20, "confidence": .95,
                                "all_assets_pass": all(a["test_comparison"]["passes_evidence_gate"] for a in assets)},
              "assets": assets, "limitations": [
                  "The earlier study's 2022–2025 outcomes were known. This is a fixed new retrospective rule, not a historically preregistered or prospectively unseen strategy.",
                  "No parameter search is performed. Train and validation metrics are descriptive and do not select a candidate.",
                  "Daily bootstrap blocks assume sufficient return-process stability; noncircular blocks underweight sample endpoints. The interval is conditional and cannot remove retrospective selection bias.",
                  "The bootstrap resamples paired already-net daily returns, not executable strategy paths; it is not a future return confidence interval or a probability of profitability.",
                  "Long-only ETF exposure differs across strategies. Lower drawdown can reflect more cash rather than better market prediction.",
                  "Single public vendor, retrospective surviving ETF selection, zero cash interest and risk-free rate, fixed costs, no tax or market impact, and 35-day dividend payout proxy limit realism.",
                  "Any failure of the fixed evidence gate remains a failure. No performance guarantee, live deployment or prospective record is established."]}
    pd.DataFrame(all_metrics).to_csv(out / "metrics.csv", index=False, float_format="%.12g")
    _write_exclusive(out / "results.json", result)
    _write_report(out, protocol, result)
    files = sorted(path for path in out.rglob("*") if path.is_file())
    _write_exclusive(out / "artifact_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                     "protocol_sha256": protocol_hash,
                     "files_sha256": {str(path.relative_to(out)): digest(path) for path in files}})
    return result


def _write_report(out: Path, protocol: dict, result: dict) -> None:
    spec = protocol["specification"]
    passed = result["evidence_gate"]["all_assets_pass"]
    lines = ["# 固定趋势与波动率控制回溯研究\n",
             ("两个资产均通过本次固定证据门槛。" if passed else "本次结果未通过要求两个资产同时成立的固定证据门槛。") +
             "本研究没有参数搜索，也不会在查看测试结果后调整规则。先前项目已经查看过 2022–2025 年历史，因此本次仍是回溯分析，不能称为全新独立样本外证据。\n",
             "规则：总回报收盘价格高于 200 日 EMA 时，目标资产权重为 min(100%, 10% / 过去 63 日对数收益年化波动率)；否则目标为现金。两项指标仅使用当时已完成的收盘数据。实际仓位与目标差达 5 个百分点时，下一交易日开盘模拟成交。低于调仓带的残余仓位可能保留。\n",
             "200 日 EMA、63 日波动、10% 目标和 5% 调仓带均在本轮输出前固定；训练与验证阶段只作描述。佣金每边 1 bp、滑点每边 5 bp；整股、不加杠杆，分红应收与 35 日现金到账代理沿用原记账器。现金利息与夏普无风险利率均为零。\n",
             "## 成本后历史测试比较\n",
             "| 资产 | 策略 | 年化收益 | 净夏普 | 最大回撤 | 平均仓位 | 年换手 | 成交笔数 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    names = {STRATEGY: "固定趋势＋波动率控制", "half": "50% 仓位＋5pp 调仓带", "buy_hold_reinvest": "买入持有及分红再投"}
    for asset in result["assets"]:
        for name, metrics in asset["periods"]["test"].items():
            lines.append(f"| {asset['symbol']} | {names[name]} | {metrics['cagr']:.2%} | {metrics['sharpe_rf0']:.3f} | {metrics['max_drawdown']:.2%} | {metrics['mean_exposure']:.2%} | {metrics['annual_turnover']:.2f} 倍 | {metrics['trade_count']} |")
    lines.extend(["\n## 配对区块 bootstrap 不确定性\n",
                  "相同日期索引同时用于策略与半仓基准，按 20 个交易日的重叠移动区块抽样 5,000 次；报告净夏普差的双侧 95% 百分位区间。门槛为夏普差至少 0.20 且区间下界严格大于零，且两个资产都满足。此区间只描述现有收益样本的不确定性，不是未来收益区间，也不能消除事后选策略的偏差。\n",
                  "| 资产 | 策略减半仓夏普差 | 95% 区间 | 固定门槛 |",
                  "| --- | --- | --- | --- |"])
    for asset in result["assets"]:
        comparison = asset["test_comparison"]
        bootstrap = comparison["bootstrap"]
        lines.append(f"| {asset['symbol']} | {comparison['sharpe_delta']:+.3f} | [{bootstrap['ci_lower']:+.3f}, {bootstrap['ci_upper']:+.3f}] | {'通过' if comparison['passes_evidence_gate'] else '未通过'} |")
    lines.extend(["\n## 读取结论的边界\n",
                  "较低回撤可能来自现金仓位，须同时看平均暴露、半仓基准与满仓基准。两个存续 ETF、单一供应商、固定成本、零现金利率以及分红支付日代理均为局限。区块 bootstrap 假定收益过程足够稳定；区块边界改变序列，且不等价于重新执行策略。\n",
                  "这次结果只能支持或否定本轮预先固定的证据门槛，不能证明未来盈利。没有滚动重新训练、真实成交或前瞻纸面记录。后续任何改动都应另开研究版本，原始失败结果保留。\n",
                  "## 可追溯文件\n",
                  "[冻结协议](protocol_frozen.json) · [完整数值](results.json) · [全部时期指标](metrics.csv) · [输出哈希](artifact_manifest.json)\n",
                  f"冻结时间（UTC）：{protocol['frozen_utc']}。评估时间（UTC）：{result['evaluated_utc']}。所引用代码和原始数据哈希保存在冻结协议中；原 run_001 保持不变。\n"])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "results/regime_study")
    args = parser.parse_args()
    result = freeze(args.root, args.out) if args.stage == "freeze" else evaluate(args.root, args.out)
    if args.stage == "freeze":
        print(json.dumps({"frozen": str(args.out / "protocol_frozen.json"),
                          "performance_computed": False, "study_id": result["specification"]["study_id"]}, indent=2))
    else:
        print(json.dumps({"results": str(args.out / "results.json"), "evidence_gate": result["evidence_gate"],
                          "comparisons": [{"symbol": row["symbol"], **row["test_comparison"]} for row in result["assets"]]}, indent=2))


if __name__ == "__main__":
    main()
