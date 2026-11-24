"""Bounded, local research workflows with a trace of executed tools."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quantlab.engine import Costs, backtest, features
from verify_results import digest, verify


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _label(label: str, value: str) -> dict:
    return {"label": label, "value": value}


class ResearchAgent:
    """Executes allowlisted research tools and stops when a prerequisite fails."""

    def __init__(self, service, symbol: str):
        self.service = service
        self.symbol = symbol
        self.run = service.root / "results/run_001"
        self.trace: list[dict] = []
        self.findings: list[dict] = []
        self.protocol: dict = {}

    def _tool(self, name: str, label: str, action: Callable[[], str]) -> bool:
        try:
            summary = action()
        except (AssertionError, OSError, KeyError, ValueError) as exc:
            self.trace.append({"tool": name, "label": label, "status": "failed",
                               "summary": str(exc)[:350] or "核对失败"})
            return False
        self.trace.append({"tool": name, "label": label, "status": "completed", "summary": summary})
        return True

    def _frozen_inputs(self) -> str:
        self.frozen = _read(self.run / "protocol_frozen.json")
        self.selection = _read(self.run / "selection.json")
        self.saved = _read(self.run / "results.json")
        if (digest(self.run / "protocol_frozen.json") != self.selection["protocol_sha256"] or
                digest(self.run / "selection.json") != self.saved["selection_sha256"]):
            raise ValueError("研究协议、选参记录或结果的哈希链不一致")
        self.config = self.frozen["config"]
        self.params = self.selection["winner"]["params"]
        self.protocol.update({"source": "results/run_001", "train": self.config["train"],
                              "validation": self.config["validation"], "test": self.config["test"],
                              "params": self.params, "costs": self.config["costs"], "sharpe_rf": 0,
                              "scope": "两段研究均使用已知历史行情"})
        return f"已核对冻结协议与选参链。动态策略的调仓阈值为 {self.params['band'] * 100:g} 个百分点。"

    def _replay(self) -> str:
        request = {"symbol": self.symbol, "strategy": "sigmoid", "params": self.params,
                   "start": self.config["test"][0], "end": self.config["test"][1],
                   "initial_cash": self.config["initial_cash"], "costs": self.config["costs"]}
        self.replay = self.service._compute_backtest(request, full_replay=True)
        self.frame = self.service._frame(self.symbol)
        self.chart = pd.DataFrame(self.replay["chart"]).set_index("date")
        for series, strategy in (("equity", "selected_sigmoid"), ("half", "half"),
                                 ("benchmark", "buy_hold_reinvest")):
            path = self.run / "ledgers" / f"{self.symbol}_test_{strategy}_daily.csv"
            saved_equity = pd.read_csv(path).set_index("date")["equity"]
            if not self.chart.index.equals(saved_equity.index) or not np.allclose(
                self.chart[series], saved_equity, atol=1e-5, rtol=0
            ):
                raise ValueError(f"网页曲线 {series} 与原始账本不一致")
        if np.array_equal(self.chart.equity.to_numpy(), self.chart.half.to_numpy()):
            raise ValueError("动态策略与半仓对照输出了完全相同的数据")
        return f"复跑 {len(self.chart):,} 个交易日。网页使用的三条净值序列逐日与各自账本一致。"

    def _ledger(self) -> str:
        cash = self.config["initial_cash"]
        shares = 0
        unpaid: list[tuple[pd.Timestamp, float]] = []
        fills = {row["date"]: row for row in self.replay["trades"]}
        max_error = 0.0
        costs = Costs(**self.config["costs"])
        for session, row in self.chart.iterrows():
            day = pd.Timestamp(session)
            loc = self.frame.index.get_loc(day)
            price = self.frame.iloc[loc]
            cash += sum(amount for due, amount in unpaid if due <= day)
            unpaid = [(due, amount) for due, amount in unpaid if due > day]
            entitlement = shares * price.dividend
            if entitlement and costs.dividend_cash_lag_days:
                unpaid.append((day + pd.Timedelta(days=costs.dividend_cash_lag_days), entitlement))
            else:
                cash += entitlement
            if fill := fills.get(session):
                if loc == 0 or fill["signal_date"] != self.frame.index[loc - 1].date().isoformat():
                    raise ValueError(f"{session} 的成交没有对应上一交易日信号")
                quantity = fill["quantity"]
                expected_price = price.open * (1 + math.copysign(costs.slippage_bps / 10000, quantity))
                notional = quantity * expected_price
                fee = abs(notional) * costs.commission_bps / 10000
                if not all(math.isclose(fill[key], expected, rel_tol=0, abs_tol=1e-8) for key, expected in
                           (("price", expected_price), ("notional", notional), ("commission", fee))):
                    raise ValueError(f"{session} 的成交价格、成交额或佣金不能对账")
                cash -= notional + fee
                shares += quantity
            receivable = sum(amount for _, amount in unpaid)
            equity = cash + shares * price.close + receivable
            max_error = max(max_error, abs(equity - row.equity), abs(cash - row.cash),
                            abs(receivable - row.receivable))
            if cash < -1e-7 or shares < 0 or shares != row.shares or max_error > 1e-5:
                raise ValueError(f"{session} 的持仓、现金、分红或净值不能对账")
        self.findings.append(_label("逐日对账最大误差", f"${max_error:.2e}"))
        return f"独立按原始开盘价、{len(fills)} 笔成交和分红重建现金、份额及净值。成交日均紧接信号日。"

    def _explain(self) -> str:
        strategy = self.replay["metrics"]
        half = self.replay["baseline_metrics"]["half"]
        previous = np.vstack(([strategy["initial_cash"]] * 2, self.chart[["equity", "half"]].to_numpy()))
        daily_returns = previous[1:] / previous[:-1] - 1
        correlation = np.corrcoef(daily_returns.T)[0, 1]
        feature = features(self.frame).loc[self.config["test"][0]:self.config["test"][1]]
        signal = self.params["mean_reversion"] * feature.deviation - self.params["momentum"] * feature.momentum
        p05, p95 = (1 / (1 + np.exp(signal))).quantile([.05, .95])
        self.findings.extend([
            _label("净夏普：动态 / 半仓", f"{strategy['sharpe_rf0']:.3f} / {half['sharpe_rf0']:.3f}"),
            _label("平均实际仓位：动态 / 半仓", f"{_pct(strategy['mean_exposure'])} / {_pct(half['mean_exposure'])}"),
            _label("目标仓位的中间 90%", f"{_pct(p05)} 至 {_pct(p95)}"),
            _label("成交：动态 / 半仓", f"{strategy['trade_count']} / {half['trade_count']} 笔"),
            _label("每日收益相关性", f"{correlation:.5f}"),
            _label("期末净值差：动态减半仓", f"${strategy['final_equity'] - half['final_equity']:+,.2f}"),
        ])
        return (f"目标多在半仓附近，且目标与实际仓位须相差 {self.params['band'] * 100:g} 个百分点才会调仓。"
                f"实际平均仓位为 {_pct(strategy['mean_exposure'])}，每日收益相关性为 {correlation:.5f}。")

    def _verify_selection(self) -> str:
        audit = verify(self.run)
        if not audit["passed"]:
            raise ValueError("未通过：" + "、".join(item["name"] for item in audit["checks"] if not item["passed"]))
        winner = self.selection["winner"]
        return (f"通过 {len(audit['checks'])} 项核验。训练候选累计 {audit['counts']['unique_evaluations']:,} 次评估，"
                f"选中 {winner['method'].upper()} / 种子 {winner['seed']}，验证期夏普 {winner['validation']['sharpe_rf0']:.3f}。")

    def _benchmark(self) -> str:
        metrics = {row["strategy"]: row for row in self.saved["metrics"]
                   if row["symbol"] == self.symbol and row["period"] == "test"}
        names = (("selected_sigmoid", "寻优后的 Sigmoid"), ("fixed_sigmoid", "固定参数 Sigmoid"),
                 ("half", "固定半仓"), ("buy_hold_reinvest", "买入持有再投资"))
        for key, label in names:
            row = metrics[key]
            self.findings.append(_label(label, f"夏普 {row['sharpe_rf0']:.3f} · 年化 {_pct(row['cagr'])} · 最大回撤 {_pct(row['max_drawdown'])}"))
        return "读取同一标的、历史区间和交易成本下的四组结果，区分固定参数策略和固定半仓。"

    def _same_band_control(self) -> str:
        frame = self.service._frame(self.symbol)
        params = {"mean_reversion": 0.0, "momentum": 0.0, "band": self.params["band"]}
        control = backtest(frame, "sigmoid", params, *self.config["test"],
                           Costs(**self.config["costs"]), self.config["initial_cash"]).metrics
        strategy = next(row for row in self.saved["metrics"] if row["symbol"] == self.symbol and
                        row["period"] == "test" and row["strategy"] == "selected_sigmoid")
        delta = strategy["sharpe_rf0"] - control["sharpe_rf0"]
        self.findings.append(_label("同调仓阈值、去掉信号", f"半仓夏普 {control['sharpe_rf0']:.3f} · 加入信号后夏普差 {delta:+.3f}"))
        return (f"把两个信号权重归零，只保留 50% 目标和原 {self.params['band'] * 100:g} 个百分点的调仓阈值。"
                "用于解释已有历史表现，不用于重新选参。")

    def _regime_study(self) -> str:
        directory = self.service.root / "results/regime_study"
        result_path = directory / "results.json"
        manifest = _read(directory / "artifact_manifest.json")
        if digest(result_path) != manifest["files_sha256"]["results.json"]:
            raise ValueError("新策略结果与输出清单的哈希不一致")
        result = _read(result_path)
        if digest(directory / "protocol_frozen.json") != result["protocol_sha256"]:
            raise ValueError("新策略结果与冻结协议的哈希不一致")
        asset = next((row for row in result["assets"] if row["symbol"] == self.symbol), None)
        if asset is None:
            raise ValueError("新策略实验中没有该标的")
        stats = asset["test_comparison"]
        low, high = stats["bootstrap"]["ci_lower"], stats["bootstrap"]["ci_upper"]
        trend = asset["periods"]["test"][result["strategy_id"]]
        half = asset["periods"]["test"]["half"]
        self.protocol["new_study"] = {"id": result["study_id"], "evidence_gate": result["evidence_gate"],
                                      "historical_retrospective": result["retrospective"]}
        self.findings.extend([
            _label("趋势＋波动控制：净夏普 / 半仓", f"{trend['sharpe_rf0']:.3f} / {half['sharpe_rf0']:.3f}"),
            _label("新策略减半仓的夏普差", f"{stats['sharpe_delta']:+.3f} · 95% 区间 [{low:+.3f}, {high:+.3f}]"),
        ])
        decision = "达到" if stats["passes_evidence_gate"] else "未达到"
        return (f"已核对新策略的冻结协议。{self.symbol} {decision}事先设定的条件："
                "夏普差至少 +0.20，且 95% 区间下界大于 0。研究使用已知历史。")

    def execute(self, profile: str) -> dict:
        if profile == "curve_audit":
            tools = (("frozen_inputs", "读取并核对冻结参数", self._frozen_inputs),
                     ("replay", "复跑策略并比对三条曲线", self._replay),
                     ("ledger", "核对成交与每日账本", self._ledger),
                     ("curve_diagnostics", "计算仓位相似度与净值差", self._explain))
        else:
            tools = (("frozen_inputs", "核对冻结的时间划分与参数", self._frozen_inputs),
                     ("verify_selection", "检查选参记录和评估预算", self._verify_selection),
                     ("benchmark_comparison", "比较固定参数与固定半仓", self._benchmark),
                     ("same_band_control", "去掉信号并保留相同调仓阈值", self._same_band_control),
                     ("regime_evidence", "检查新策略的夏普证据", self._regime_study))
        passed = all(self._tool(*tool) for tool in tools)
        if not passed:
            summary = "核对中止，请查看最后一个工具的结果。"
        elif profile == "curve_audit":
            summary = f"{self.symbol} 曲线逐日核对通过。{self.trace[-1]['summary']}"
        else:
            summary = f"{self.symbol} 原实验记录核对通过。{self.trace[-1]['summary']}"
        return {"profile": profile, "symbol": self.symbol, "steps": self.trace, "passed": passed,
                "summary": summary, "findings": self.findings, "protocol": self.protocol,
                "mode": "rule_orchestrated", "run_id": "run_001"}


def run_research_agent(service, body: dict) -> dict:
    profile = body.get("profile")
    symbol = body.get("symbol")
    if profile not in ("curve_audit", "validation") or symbol not in ("SPY", "GLD"):
        raise ValueError("请选择支持的核对任务和标的（SPY 或 GLD）")
    return ResearchAgent(service, symbol).execute(profile)
