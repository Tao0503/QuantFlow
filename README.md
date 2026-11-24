# QuantFlow 量化策略研究终端

QuantFlow 是一个 ETF 策略研究应用，包含 React 中文界面、Python 回测服务和 SQLite 工作区。项目用 Sigmoid 动态仓位检验成本后的策略表现，并比较遗传算法与等预算的随机搜索。

[公开演示站](https://tao0503.github.io/QuantFlow/)展示原始历史实验的汇总数据与方法。完整的策略回测、参数优化、Agent 工具和账本在本地应用中运行。新克隆可以生成带标签的合成行情来体验功能。[部署与运行指南](GITHUB_DEPLOYMENT.md)

「研究 Agent」可以选 SPY 或 GLD，再触发曲线审计或验证检查。Agent 以规则编排本地工具，逐步核对冻结参数、重跑策略、重建账本、检查对照结果，遇到失败就停止并返回工具记录。验证检查也展示与原策略相同调仓阈值的零信号对照，以及预先固定的趋势＋波动控制实验。

## 三参数联合调参

进入「参数优化」，默认打开「三参数联合调参」。输入每轮预算后，遗传算法和随机搜索各跑三个固定种子，同时搜索均值回归权重、动量权重和调仓阈值。每轮在 2010–2018 年的三个独立训练段评分，以相对半仓的净夏普差均值扣减 0.25 倍段间标准差。六轮都结束后再计算 2019–2021 年验证结果，并按净夏普、最大回撤、年换手排序。2022 年起的历史行情不参与选参。

页面保留每轮真实预算、进度、候选排序、半仓和当前参数对照。选定候选会保存在参数库，可用按钮带入 SPY 验证期回测配置。已有实例参数由手动晋升与应用决定。原始历史实验采用每轮 1,024 次评估，汇总结果见公开演示站。

## 本地运行

新克隆先完成一次初始化。需要 Python 3.12 和 Node.js 22。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m demo.bootstrap_public
(cd web && npm ci && npm run build)
```

初始化生成合成 OHLCV、分红事件和回测产物，适合体验流程。原始历史实验的汇总结果可在演示站查阅。

## 打开交互终端

```bash
./start_demo.sh
```

浏览器打开 **http://127.0.0.1:8766**。下次启动仍可运行同一命令，实例、参数和任务保存在 `demo/state/workbench.sqlite3`。

可操作的流程为：创建策略实例 → 提交 GA 或随机搜索任务 → 查看真实进度和收敛曲线 → 审批候选并显式绑定实例 → 选择日期和成本运行回测 → 查看基准对照、导出交易 CSV → 逐个交易日回放成交。详见 [现场演示说明](DEMO_GUIDE.md)。历史回放不连接券商。

首次打开可查看初始化生成的 GA／随机搜索任务和 SPY／GLD 回测记录，也可现场重新计算。

从源码重建界面：

```bash
cd web
npm ci
npm run build
```

开发时可在一个终端运行 `./start_demo.sh`，另一个终端进入 `web` 执行 `npm run dev`。正式展示使用 8766 端口上的构建版，界面资源全部在本地。

## 多标的回测

打开左侧的「多标的回测」，勾选两个或更多标的，填写合计 100% 的预算比例。比如 SPY 60%、GLD 40%，固定半仓的目标就是整个账户的 30% 投 SPY、20% 投 GLD、另外 50% 留现金。动态策略给每个标的算出信号，再把信号建议的投资比例乘以该标的预算比例。

一次运行用相同的标的、预算、区间和交易费用计算两种规则。各标的共用现金，各自记录份额。交易发生在下一交易日开盘。卖出先结算，现金不够时按同一比例缩小买单，再按整股成交。结果展示组合总曲线、期末每项持仓和现金，以及带标的代码的成交明细。选「主策略」可以换报表使用的逐笔成交和期末持仓。

组合实验中的动态规则使用当前工作区的 SPY 参数版本。组合引擎在 `demo/portfolio.py`，早期行情先用于计算指标，两种规则从各标的指标都齐备后开始决策。标的须使用共同交易日，缺少区间内行情会报错。首页展示由本地数据生成的单标的冻结实验。

内置标的有 SPY、GLD 和 IEF。以后可用 `.venv/bin/python -m demo.add_asset 标的代码 --name '显示名'` 添加适用的美国 ETF，再重启本地服务。导入程序会校验美元计价、共同交易日、行情覆盖和公司行动，行情和校验值保存在 `data/portfolio/`。这个版本的组合账户统一以美元计价，覆盖美国共同交易日。

## 回测指标

单标的、组合回测和联合调参验证页都提供指标对照。分类参考[聚宽公开风险指标](https://cdn.joinquant.com/help/img/JoinQuantAPI.pdf)。QuantFlow 用成本后每日完整净值计算收益、波动、回撤、夏普和索提诺，以固定半仓的同日收益作为 Alpha、Beta 和信息比率的参考。年化基数为 252 个交易日，无风险利率取零。卖出胜率按 FIFO 与买入批次匹配，盈利因子取已实现盈利和亏损的比值，未卖出份额与分红另计。没有足够样本或分母为零时，比率显示为 `—`。旧记录缺少的新指标需要重新运行才能生成。

外部 Alpha 提交与回测指标是两类事。WorldQuant 的公开资料确认 [BRAIN 研究顾问可以在平台提交 Alpha](https://www.worldquant.com/ideas/worldquant-launches-brain-platform-and-inaugural-global-alphathon-competition/)，具体门槛以该平台的现行提交规则为准。此处的 ETF 回测表使用本地计算口径。

完整范围和参考关系见 [复现说明](REPRODUCTION_NOTES.md)，数学、时序和实验边界见 [研究协议](RESEARCH_PROTOCOL.md)。这不是原 Go SaaS 全部模块的逐项复刻，也不是盈利承诺。没有实现的模块逐项列在复现说明中。

## 查看实际结果

初始化之后，可以打开本地的 `results/run_001/dashboard.html` 或阅读 `results/run_001/REPORT.md`。报告包含基准与成本检验。原始历史研究的汇总数据由公开演示站提供。

本轮已运行并验证：31 项测试通过、28 项独立账本与冻结检查通过，完整复跑结果一致。2022–2025 年 SPY 上选定策略年化 5.53%、Sharpe 0.618、最大回撤 13.39%；半仓基准为 5.82%、0.672、12.47%。本轮没有建立自适应信号优于简单仓位配置的证据，GLD 迁移表现也接近半仓基准。

## 运行

完成本地初始化后，验证缓存数据和运行测试：

```bash
.venv/bin/python -m quantlab.data
.venv/bin/python -m unittest discover -s tests -v
```

在全新目录复现一轮完整实验：

```bash
.venv/bin/python -m quantlab.run fit --out results/my_run
.venv/bin/python -m quantlab.run evaluate --out results/my_run
.venv/bin/python -m quantlab.report --out results/my_run
```

`fit` 先冻结代码、数据与配置摘要，再只用训练期搜索，并只用验证期从六个训练冠军中选一个。`evaluate` 检查摘要未变后运行冻结参数与基准。命令拒绝覆盖已有结果。重跑相同历史期能证明计算确定性，不能把已见历史重新变成新的样本外数据。不要为了更好结果修改测试期规则后沿用原样本外表述。

`quantlab.data` 默认复用本次快照，校验 CSV 和原始 JSON 哈希。只有显式 `--refresh` 才重新下载；供应商修订历史后不能与原运行混为一谈。

## 实现

```text
数据快照 → 因果总回报特征 → 纯 step → 次日开盘成交账本 → 指标
                               ↑
       GA / 随机搜索 → 训练评分 → 验证期选参 → 冻结 → 历史测试与 GLD 迁移
```

| 模块 | 职责 |
|---|---|
| `quantlab/data.py` | 下载并冻结 Yahoo chart、校验价格与分红、构建因果特征价格 |
| `quantlab/engine.py` | 纯策略决策、整股成交、现金与应收分红账本、指标 |
| `quantlab/search.py` | 等预算 GA／随机搜索、重复候选缓存、完整历史 |
| `quantlab/run.py` | 分割隔离、协议冻结、候选选择、压力与邻域诊断 |
| `quantlab/report.py` | 研究报告、图片和无需外部服务的 HTML 看板 |
| `tests/` | 因果性、成交时序、分红归属、资金约束、预算和流程测试 |

## 本轮固定设计

- 2010–2018 训练，2019–2021 验证，2022–2025 历史测试；训练内分三段独立评分，不称滚动 WFO。
- 两个特征：21 日价格偏离、63 日动量；Sigmoid 转为 0–100% 目标权重；只搜索两个非负信号系数与一个调仓带宽。
- GA 和随机搜索，各三个固定种子、每次 256 个不同参数，总计 1,536 次候选评估。每个候选运行三个训练区间，共 4,608 次训练区间回放。
- SPY 按验证期 Sharpe 选出的参数原样应用于 GLD，GLD 不调参。
- 基准：买入持有留存分红现金、分红代理到账后再投资、50% 带宽再平衡、简单动量、现金、固定 Sigmoid。
- 收盘信号下一开盘成交；1 bp 佣金、5 bp 单边滑点；0/1/2/5 倍成本压力检验。

## 阅读结果的边界

本次“冻结”是当前回溯工作流在查看相应输出前锁定，并非在 2022 年之前完成的历史注册。研究者可能知道历史行情，不能消除先验选择偏差。两个存续 ETF 不构成无偏资产全集。

数据来自单一供应商。分红除息日建立应收，35 个日历日后转成可用现金，是支付日代理；不是精确券商流水。未实现拆股记账，出现拆股直接报错。默认费用为假设，现金利息和无风险利率均为零，未建模税费、最小佣金、冲击、流动性及真实开盘撮合。

低仓位带来的低回撤不等于风险调整 alpha；报告同时给出暴露、波动、全仓及半仓基准。没有 LLM／新闻信号、机器学习预测、实盘部署或前瞻纸面业绩，也没有把三个种子当成统计显著性证据。

## 可追溯文件

`data/manifest.json` 记录数据来源与摘要；`protocol_frozen.json` 记录运行时和方法摘要；`search_*.json` 保存全部候选；`selection.json` 保存查看测试前的选择；`metrics.csv`、`cost_stress.csv`、`ledgers/` 保存指标和账本。验证日志与复跑比较另存于 `results/verification.json`。

复核已有结果，无需调参或重跑选择：

```bash
.venv/bin/python verify_results.py --run results/run_001
.venv/bin/python check_reproducibility.py --first results/run_001 --second results/run_002
```

`results/audit.json` 是独立审计记录，`results/tests.log` 保存测试输出；`run_002` 是同数据同规则的确定性复跑。其存在不代表对新的未见数据作过验证。

参考数据入口：[Yahoo SPY chart](https://query1.finance.yahoo.com/v8/finance/chart/SPY)、[Yahoo GLD chart](https://query1.finance.yahoo.com/v8/finance/chart/GLD)。完整请求参数与抓取时间以 manifest 为准。数据仅在本地用于本次研究，公开发布前需要另行核实供应商的使用和再分发条件。
