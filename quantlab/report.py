"""Create a readable, offline Chinese report from a completed frozen experiment."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


NAMES = {
    "selected_sigmoid": "验证集选定 Sigmoid",
    "fixed_sigmoid": "固定参数 Sigmoid",
    "buy_hold": "买入持有（分红留现金）",
    "buy_hold_reinvest": "买入持有（代理到账后再投）",
    "half": "50% 仓位（5 个百分点带宽）",
    "momentum": "63 日动量",
    "cash": "现金",
}
PLOT_NAMES = {
    "selected_sigmoid": "Selected sigmoid",
    "fixed_sigmoid": "Fixed sigmoid",
    "buy_hold": "Buy/hold, distributions in cash",
    "buy_hold_reinvest": "Buy/hold, reinvest proxy payouts",
    "half": "50% target, 5pp band",
    "momentum": "63-session momentum",
}
COLORS = {"selected_sigmoid": "#0f766e", "fixed_sigmoid": "#6598b6", "buy_hold": "#203852",
          "buy_hold_reinvest": "#7b629e", "half": "#b58431", "momentum": "#b45e67"}
ORDER = tuple(NAMES)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value: float, places: int = 2) -> str:
    value = 0.0 if abs(float(value)) < 1e-12 else float(value)
    return f"{100 * value:.{places}f}%"


def number(value: float, places: int = 2) -> str:
    return f"{float(value):,.{places}f}"


def pp(value: float) -> str:
    return f"{100 * value:+.2f} 个百分点"


def hong_kong_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M:%S HKT（UTC+08:00）")


class Document:
    """Keep Markdown and standalone HTML tables in one consistent representation."""
    def __init__(self, title: str):
        self.md = [f"# {title}\n"]
        self.parts = [f"<h1>{html.escape(title)}</h1>"]

    def heading(self, text: str):
        self.md.append(f"\n## {text}\n")
        self.parts.append(f"<h2>{html.escape(text)}</h2>")

    def paragraph(self, text: str):
        self.md.append(text + "\n")
        self.parts.append(f"<p>{html.escape(text)}</p>")

    def table(self, headers: list[str], rows: list[list]):
        as_text = [[str(v) for v in row] for row in rows]
        self.md.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
        self.md.extend("| " + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in row) + " |" for row in as_text)
        self.md.append("")
        head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in row) + "</tr>" for row in as_text)
        self.parts.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')

    def figure(self, path: Path, relative: str, caption: str):
        self.md.append(f"![{caption}]({relative})\n")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        self.parts.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')

    def links(self, items: list[tuple[str, str]]):
        self.md.append(" · ".join(f"[{label}]({target})" for label, target in items) + "\n")
        self.parts.append('<p class="links">' + " · ".join(f'<a href="{html.escape(target, quote=True)}">{html.escape(label)}</a>' for label, target in items) + "</p>")


def plot_test(out: Path, symbol: str, metrics: list[dict], initial_cash: float) -> tuple[Path, list[Path]]:
    fig, (ax_nav, ax_dd) = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True,
                                      gridspec_kw={"height_ratios": [2.25, 1], "hspace": .12})
    fig.patch.set_facecolor("#fbfcfd")
    used = []
    for name in ORDER:
        if name == "cash":
            continue
        path = out / "ledgers" / f"{symbol}_test_{name}_daily.csv"
        frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
        if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Invalid daily ledger: {path}")
        equity = frame["equity"].to_numpy(dtype=float)
        if not np.isfinite(equity).all() or (equity <= 0).any():
            raise ValueError(f"Invalid ledger equity: {path}")
        row = next(r for r in metrics if r["strategy"] == name)
        if not np.isclose(equity[-1], row["final_equity"], rtol=1e-9, atol=1e-6):
            raise ValueError(f"Final ledger equity does not match result: {path}")
        norm = equity / initial_cash
        drawdown = equity / np.maximum.accumulate(np.r_[initial_cash, equity])[1:] - 1
        width = 2.4 if name == "selected_sigmoid" else 1.25
        linestyle = "--" if name in ("fixed_sigmoid", "buy_hold_reinvest") else "-"
        ax_nav.plot(frame.index, norm, label=PLOT_NAMES[name], color=COLORS[name], lw=width, ls=linestyle)
        ax_dd.plot(frame.index, drawdown, color=COLORS[name], lw=width, ls=linestyle)
        used.append(path)
    ax_nav.set_title(f"{symbol} | Frozen retrospective test | 2022–2025", loc="left", fontsize=14, color="#152e49", pad=14)
    ax_nav.set_ylabel("NAV / initial cash")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(PercentFormatter(1))
    ax_nav.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)
    ax_nav.axhline(1, color="#a8b2bf", lw=.7, zorder=0)
    for ax in (ax_nav, ax_dd):
        ax.set_facecolor("#fbfcfd")
        ax.grid(axis="y", color="#dde3e9", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#bfc9d4")
        ax.tick_params(colors="#4b5a6a", labelsize=9)
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(.075, .02, "Net of modeled commission and slippage. Dividends: ex-date receivable, 35-calendar-day cash proxy. No cash interest.", fontsize=8, color="#5e6c7c")
    fig.subplots_adjust(left=.08, right=.98, top=.92, bottom=.09)
    path = out / "figures" / f"{symbol}_test.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path, used


def make_report(out: Path) -> dict:
    out = out.resolve()
    inputs = [out / name for name in ("results.json", "selection.json", "protocol_frozen.json")]
    result, selection, frozen = [json.loads(p.read_text(encoding="utf-8")) for p in inputs]
    if digest(inputs[1]) != result["selection_sha256"] or digest(inputs[2]) != selection["protocol_sha256"]:
        raise ValueError("Results, selection and frozen protocol hashes do not agree")
    config = frozen["config"]
    symbols = (config["primary_symbol"], config["transfer_symbol"])
    test_rows = {symbol: sorted([r for r in result["metrics"] if r["symbol"] == symbol and r["period"] == "test"],
                               key=lambda r: ORDER.index(r["strategy"])) for symbol in symbols}
    for symbol, rows in test_rows.items():
        if {r["strategy"] for r in rows} != set(ORDER):
            raise ValueError(f"Missing or extra test strategy metrics for {symbol}")
    selected = selection["winner"]
    doc = Document("ETF 自适应仓位策略研究报告")
    doc.paragraph("研究对象为 SPY 与 GLD，策略使用价格偏离和动量信号，经 Sigmoid 映射为目标仓位。报告呈现已经完成的历史回测、参数选择过程及稳健性诊断；全部数值来自本次运行保存的结果和逐日账本。")
    params = result["selected_params"]
    doc.paragraph(f"验证集最终选择 {result['selected_method'].upper()}、随机种子 {result['selected_seed']} 的候选；均值回归系数 {params['mean_reversion']:.4f}，动量系数 {params['momentum']:.4f}，再平衡带宽 {pct(params['band'])}。选择依据是预先固定的验证集夏普排序规则，最终测试及 GLD 迁移结果均不参与该选择。")
    comparisons = [{r["strategy"]: r for r in test_rows[symbol]} for symbol in symbols]
    if all(rows["selected_sigmoid"]["cagr"] <= rows["half"]["cagr"] and
           rows["selected_sigmoid"]["sharpe_rf0"] <= rows["half"]["sharpe_rf0"] for rows in comparisons):
        doc.paragraph("本次两个测试资产中，选定策略的年化收益和夏普均未超过 50% 带宽再平衡基准，结果不支持其具有已验证的增量优势。较满仓买入持有更低的回撤伴随更低的市场暴露，不能将这一现象直接归因为成功择时。")
    for symbol in symbols:
        rows = {r["strategy"]: r for r in test_rows[symbol]}
        s, bh, half = rows["selected_sigmoid"], rows["buy_hold"], rows["half"]
        doc.paragraph(f"{symbol} 最终测试：选定策略年化收益 {pct(s['cagr'])}、夏普 {number(s['sharpe_rf0'], 3)}、最大回撤 {pct(s['max_drawdown'])}；买入持有且分红留现金的年化收益为 {pct(bh['cagr'])}，二者年化收益差为 {pp(s['cagr'] - bh['cagr'])}。与 50% 带宽再平衡基准相比，选定策略年化收益差为 {pp(s['cagr'] - half['cagr'])}、最大回撤差为 {pp(s['max_drawdown'] - half['max_drawdown'])}。不同仓位暴露影响这些比较，不将收益差称为风险调整后 alpha。")

    doc.heading("实验范围与冻结过程")
    doc.table(["项目", "本次设定"], [
        ["训练", " ～ ".join(config["train"]) + "；三个互不重叠的三年训练段"],
        ["验证", " ～ ".join(config["validation"]) + "；仅比较六个训练优胜候选"],
        ["最终测试", " ～ ".join(config["test"]) + "；方法和候选冻结后评估"],
        ["迁移检验", f"{config['transfer_symbol']} 使用同一组固定参数，不在该资产上调参"],
        ["搜索预算", f"GA 和随机搜索各 {len(config['seeds'])} 个种子，每次 {config['search_budget_per_method_seed']} 次唯一候选评估"],
        ["初始资金", number(config["initial_cash"], 0) + " USD；各区间独立从现金开始，无外部注资"],
        ["成本", f"每边佣金 {config['costs']['commission_bps']:g} bps，滑点 {config['costs']['slippage_bps']:g} bps"],
        ["资金约束", "整股、仅做多、不加杠杆；现金利息为零，夏普的无风险利率为零"],
    ])
    doc.paragraph(f"本次协议冻结于 {hong_kong_time(frozen['frozen_utc'])}，候选选择保存于 {hong_kong_time(selection['selected_utc'])}，测试结果生成于 {hong_kong_time(result['evaluated_utc'])}。这是在既有历史数据上先冻结流程再查看保留区间的回溯实验，不是于 2022 年之前历史预注册，也不是前瞻业绩。运行结果一旦查看，该测试区间即成为已见数据；后续任何改动不得继续将其称为全新样本外证据。")

    doc.heading("策略与记账口径")
    doc.paragraph("特征仅使用已完成收盘时点之前的数据。价格偏离为收盘总回报序列相对 21 日 EMA 的对数偏离，按 21 日收益波动率与窗口长度缩放；动量为 63 日对数回报，按同周期波动率缩放，两项分别截断到 [-3, 3]。信号等于均值回归系数乘偏离、减去动量系数乘动量；目标仓位为 1 / (1 + exp(信号))。实际仓位与目标的差达到带宽才触发再平衡；未使用原设计中的仓位反馈项、宏观定投或永久封存仓。")
    doc.paragraph("t 日收盘生成目标，最早在下一交易日开盘模拟成交。特征使用向前累计的总回报收盘序列，成交使用原始开盘价。除息前持有份额在除息日记分红应收，净值包含该应收；可用现金在除息后 35 个日历日按代理规则到账。这一等待期不是实际派息日，属于明确的简化假设；末期未到账应收仍计入净值。没有外部现金流，因此回撤从相对初始资金的账户净值计算。")
    doc.table(["策略标识", "可执行含义"], [
        ["selected_sigmoid", "六个训练优胜候选中，验证集规则选定的一组固定参数"],
        ["fixed_sigmoid", "均值回归=1、动量=1、带宽=5%，作为不搜索的规则对照"],
        ["buy_hold", "首笔建仓后持有；分红代理到账后留在现金账户，不自动复投"],
        ["buy_hold_reinvest", "首笔建仓后，分红代理到账且可用现金足够整股时再次买入"],
        ["half", "目标 50% 资产、50% 现金；偏离达到 5 个百分点时再平衡"],
        ["momentum", "63 日标准化动量正时目标 100%，否则 0%；带宽 5%"],
        ["cash", "全程持有现金，利息为零"],
    ])

    doc.heading("验证集选择")
    ranked = sorted(selection["candidates"], key=lambda r: (-r["validation"]["sharpe_rf0"],
                    r["validation"]["max_drawdown"], r["validation"]["annual_turnover"], r["method"], r["seed"]))
    doc.table(["候选", "训练目标", "验证年化收益", "验证夏普", "验证最大回撤", "验证年换手", "选择"], [
        [f"{r['method'].upper()} / {r['seed']}", number(r["training_score"], 3), pct(r["validation"]["cagr"]),
         number(r["validation"]["sharpe_rf0"], 3), pct(r["validation"]["max_drawdown"]),
         number(r["validation"]["annual_turnover"]) + " 倍", "选定" if r is ranked[0] else ""] for r in ranked
    ])
    doc.paragraph("训练目标为三个训练段的成本后夏普均值减去 0.25 倍段间总体标准差。验证集依次按夏普更高、最大回撤更低、换手更低、方法名和种子排序。验证集参与了模型选择，其表现不能称为独立最终测试表现。")

    figures = []
    for symbol in symbols:
        doc.heading(f"{symbol} 最终测试结果")
        rows = test_rows[symbol]
        doc.paragraph(f"实际交易日为 {rows[0]['start']} 至 {rows[0]['end']}，共 {rows[0]['sessions']} 个交易日。收益和净值均已扣除本次模型中的佣金与滑点；年化按 252 个交易日计算，最大回撤以正数表示损失幅度。")
        doc.table(["策略", "累计收益", "年化收益", "年化波动", "夏普 rf=0", "最大回撤"], [
            [NAMES[r["strategy"]], pct(r["total_return"]), pct(r["cagr"]), pct(r["annual_volatility"]),
             number(r["sharpe_rf0"], 3), pct(r["max_drawdown"])] for r in rows
        ])
        doc.table(["策略", "末期权益 USD", "年换手", "平均仓位", "成交笔数", "佣金 USD", "滑点损耗 USD"], [
            [NAMES[r["strategy"]], number(r["final_equity"]), number(r["annual_turnover"]) + " 倍",
             pct(r["mean_exposure"]), r["trade_count"], number(r["total_commission"]), number(r["slippage_cost"])] for r in rows
        ])
        figure, used = plot_test(out, symbol, rows, config["initial_cash"])
        figures.append(figure)
        inputs.extend(used)
        doc.figure(figure, str(figure.relative_to(out)), f"{symbol} 最终测试净值与回撤；所有非现金策略均展示")
    doc.paragraph("年换手是成交名义金额绝对值之和除以初始资金，再按交易日年化，包含初始建仓；它不等于按每日平均资产净值计算的其他换手口径。")

    doc.heading("GA 与随机搜索的有限比较")
    summaries = []
    for method in ("ga", "random"):
        c = [r for r in selection["candidates"] if r["method"] == method]
        train_scores = np.array([r["training_score"] for r in c])
        val_scores = np.array([r["validation"]["sharpe_rf0"] for r in c])
        summaries.append([method.upper(), len(c), sum(r["evaluations"] for r in c),
                          sum(r["cache_hits"] for r in c), number(train_scores.mean(), 3),
                          number(val_scores.mean(), 3), f"{val_scores.min():.3f} ～ {val_scores.max():.3f}"])
    doc.table(["方法", "种子数", "实际唯一评估总数", "缓存命中", "训练目标均值", "验证夏普均值", "验证夏普范围"], summaries)
    diagnostic_rows = []
    for symbol in symbols:
        for method in ("ga", "random"):
            r = [x for x in result["seed_diagnostics"] if x["symbol"] == symbol and x["method"] == method]
            s = np.array([x["sharpe_rf0"] for x in r])
            c = np.array([x["cagr"] for x in r])
            diagnostic_rows.append([symbol, method.upper(), number(s.mean(), 3),
                                    f"{s.min():.3f} ～ {s.max():.3f}", pct(c.mean())])
    doc.table(["资产", "方法", "种子测试夏普均值", "种子测试夏普范围", "种子测试年化收益均值"], diagnostic_rows)
    doc.paragraph("每个方法只有三个固定种子，且参数空间仅三个维度；这些结果不足以证明 GA 普遍优于随机搜索。种子测试表属于冻结选择后的诊断，既没有再次选择最优种子，也没有据此更换最终参数。缓存命中不消耗唯一候选评估预算。")

    doc.heading("成本压力测试")
    doc.paragraph("下表只改变佣金与滑点的共同倍数，保持策略和全部参数固定；0 倍用于展示无摩擦对照，不代表可实现收益。")
    doc.table(["资产", "策略", "成本倍数", "年化收益", "夏普", "最大回撤"], [
        [r["symbol"], NAMES[r["strategy"]], f"{r['multiplier']:g}×", pct(r["cagr"]),
         number(r["sharpe_rf0"], 3), pct(r["max_drawdown"])] for r in result["cost_stress"]
    ])

    doc.heading("参数邻域敏感性")
    field_names = {"mean_reversion": "均值回归系数", "momentum": "动量系数", "band": "再平衡带宽"}
    doc.paragraph("每次仅将一个选定参数乘以 0.8 或 1.2，再裁剪到冻结的合法边界，其他参数保持不变。部分扰动可能因边界裁剪而等于原值；这些结果用于描述局部敏感性，未用来重新调参。")
    doc.table(["资产", "扰动参数", "原值倍数", "实际取值", "年化收益", "夏普", "最大回撤"], [
        [r["symbol"], field_names[r["field"]], f"{r['scale']:.1f}×", number(r["params"][r["field"]], 5),
         pct(r["cagr"]), number(r["sharpe_rf0"], 3), pct(r["max_drawdown"])] for r in result["parameter_sensitivity"]
    ])

    doc.heading("局限与可支持的结论")
    for text in (
        "本项目实现了可复现的研究流程、因果特征、带成本的下一开盘价记账，以及等预算的参数搜索比较。结论仅适用于本次固定资产、样本和假设；不保证策略跑赢基准，也不推断未来盈利。",
        "SPY 与 GLD 是事后选取且持续存在的 ETF，不构成无偏的历史投资范围。GLD 是直接迁移检验，市场性质与 SPY 不同；单一迁移资产不足以证明跨市场普适性。",
        "本次使用单一公共供应商，未与第二来源逐笔核验；历史行情和分红记录可能被修订。分红的 35 日到账代理、日线整股成交、固定滑点和零现金利息均是简化假设，未模拟真实支付日、税项、最小佣金、盘口深度或市场冲击。",
        "本次没有 walk-forward 滚动重新训练、严格历史时点的全市场样本、前瞻纸面交易或实盘成交；已冻结的历史保留测试不能替代这些验证。训练段间惩罚和三个种子也不能消除多次尝试带来的选择偏差。",
        "本次策略为规则与遗传算法参数优化框架，没有 LLM、深度学习或经过验证的机器学习预测模块。不能把 AI 辅助编程过程当作策略预测能力。",
    ):
        doc.paragraph(text)

    verification_path = out.parent / "verification.json"
    audit_path = out.parent / "audit.json"
    tests_path = out.parent / "tests.log"
    if verification_path.exists() and audit_path.exists() and tests_path.exists():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        test_log = tests_path.read_text(encoding="utf-8")
        test_count = re.search(r"Ran (\d+) tests", test_log)
        doc.heading("计算验证与独立审计")
        doc.table(["验证", "保存的证据"], [
            ["单元测试", f"{test_count.group(1) if test_count else '未识别数量'} 项；日志状态 " + ("OK" if test_log.rstrip().endswith("OK") else "请检查日志")],
            ["同数据独立重跑", ("通过；" if verification["passed"] else "未全部通过；") +
             f"比较 {verification['search_runs_compared']} 份完整搜索历史、候选选择及指标；{verification['ledger_files_compared']} 份账本 CSV 逐字节比较"],
            ["独立审计", ("通过；" if audit["passed"] else "未全部通过；") +
             f"核验 {audit['counts']['test_ledger_pairs']} 对测试账本、{audit['counts']['test_daily_rows']:,} 个逐日记录及 {audit['counts']['test_fills']} 笔成交；检验冻结哈希、预算、选参和资金记账不变量"],
        ])
        doc.paragraph("这些检查验证程序可复现、保存结果一致及账本约束成立，不增加新的样本外数据，也不证明交易策略具有超额收益。")
        doc.links([("重跑验证记录", "../verification.json"), ("独立审计记录", "../audit.json"), ("单元测试日志", "../tests.log")])
        inputs.extend([verification_path, audit_path, tests_path])

    doc.heading("结果文件与复现")
    doc.paragraph("输入 JSON、协议及逐日账本是报告的数值真源。report_manifest.json 保存本报告、独立图和离线页面的 SHA256，并记录报告依赖的输入文件哈希。图中的标签采用英文以避免不同电脑缺少中文绘图字体；正文与页面均为中文，页面图片已内嵌，无需网络或 CDN。")
    doc.links([("原始结果", "results.json"), ("候选选择", "selection.json"), ("冻结协议", "protocol_frozen.json"),
               ("指标 CSV", "metrics.csv"), ("成本诊断 CSV", "cost_stress.csv"), ("报告哈希", "report_manifest.json")])
    doc.paragraph("重新生成报告：在仓库根目录运行 python -m quantlab.report --out results/run_001。该命令只读取已有实验结果，不重跑搜索或修改测试选择。重新执行完整实验应使用新的输出目录，并保留本次冻结结果。")

    markdown = out / "REPORT.md"
    dashboard = out / "dashboard.html"
    markdown.write_text("\n".join(doc.md), encoding="utf-8")
    css = """
    :root{color-scheme:light;--ink:#16304b;--muted:#596b7e;--teal:#0f766e;--line:#dbe3ea}
    *{box-sizing:border-box}body{margin:0;background:#f7f8f8;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;font-size:15px;line-height:1.8}
    main{max-width:1180px;margin:0 auto;padding:56px 38px 72px;background:#fff}h1{font-size:34px;letter-spacing:-.03em;line-height:1.3;margin:8px 0 24px;font-weight:750}h2{font-size:22px;margin:46px 0 14px;color:var(--ink);line-height:1.4}p{margin:12px 0;color:#334b62;max-width:1080px}.eyebrow{font-size:12px;letter-spacing:.12em;font-weight:700;color:var(--teal);text-transform:uppercase}
    .table-wrap{overflow-x:auto;margin:18px 0 22px;border:1px solid var(--line);border-radius:4px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.65;font-variant-numeric:tabular-nums}th{background:#ecf2f5;font-weight:650;color:var(--ink);text-align:left;white-space:nowrap}td,th{padding:11px 13px;border-bottom:1px solid var(--line);vertical-align:top}tbody tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafcfd}td:not(:first-child){white-space:nowrap}figure{margin:24px 0 30px}img{width:100%;height:auto;display:block}figcaption{font-size:12px;color:var(--muted);margin-top:9px}a{color:var(--teal);text-decoration-thickness:1px;text-underline-offset:3px}.links{font-size:13px}footer{margin-top:40px;color:var(--muted);font-size:12px}@media(max-width:700px){main{padding:30px 18px}h1{font-size:28px}h2{font-size:20px}td,th{padding:9px 10px}body{font-size:14px}}
    """
    dashboard.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ETF 自适应仓位策略研究报告</title><style>' + css + '</style></head><body><main><div class="eyebrow">Quantitative research · Reproducible experiment</div>' + "\n".join(doc.parts) + '<footer>离线研究报告 · 历史回溯结果 · 生成自已保存的实验数据</footer></main></body></html>', encoding="utf-8")
    generated = [markdown, dashboard, *figures]
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "generator_sha256": digest(Path(__file__)),
                "input_sha256": {os.path.relpath(p, out): digest(p) for p in sorted(set(inputs))},
                "output_sha256": {str(p.relative_to(out)): digest(p) for p in generated}}
    (out / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results/run_001")
    args = parser.parse_args()
    result = make_report(args.out)
    print(json.dumps({"outputs": list(result["output_sha256"]), "inputs_verified": len(result["input_sha256"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
