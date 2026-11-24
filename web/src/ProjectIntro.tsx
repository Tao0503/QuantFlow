import { useEffect, useState } from 'react';
import { ArrowRight, ArrowUpRight } from 'lucide-react';

type Metric = {
  symbol: string;
  period: string;
  strategy: string;
  cagr: number;
  max_drawdown: number;
};

const symbols = [
  { ticker: 'SPY', name: '美国大盘股票' },
  { ticker: 'GLD', name: '黄金' },
];
const strategies = [
  { id: 'selected_sigmoid', label: '动态调整仓位' },
  { id: 'half', label: '固定半仓对照' },
];
const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

function StudyResults({ synthetic }: { synthetic: boolean }) {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');

  useEffect(() => {
    const controller = new AbortController();
    const load = async () => {
      try {
        const response = await fetch('/artifacts/results.json', { signal: controller.signal });
        if (!response.ok) throw new Error('保存的结果暂不可用');
        const study: { metrics: Metric[] } = await response.json();
        const rows = symbols.flatMap(({ ticker }) => strategies.map(({ id }) =>
          study.metrics.find(row => row.symbol === ticker && row.period === 'test' && row.strategy === id)
        ));
        if (!rows.every(row => row && Number.isFinite(row.cagr) && Number.isFinite(row.max_drawdown))) {
          throw new Error('保存的结果不完整');
        }
        setMetrics(rows.filter((row): row is Metric => row !== undefined));
        setStatus('ready');
      } catch {
        if (!controller.signal.aborted) setStatus('error');
      }
    };
    void load();
    return () => controller.abort();
  }, []);

  const spyStrategy = metrics.find(row => row.symbol === 'SPY' && row.strategy === 'selected_sigmoid');
  const spyHalf = metrics.find(row => row.symbol === 'SPY' && row.strategy === 'half');
  const lowerOnBoth = symbols.every(({ ticker }) => {
    const strategy = metrics.find(row => row.symbol === ticker && row.strategy === 'selected_sigmoid');
    const half = metrics.find(row => row.symbol === ticker && row.strategy === 'half');
    return strategy && half && strategy.cagr < half.cagr;
  });

  return <section className="walkthrough-results" aria-labelledby="study-results-title">
    <div>
      <h2 id="study-results-title">跑出了什么？</h2>
      <p className="walkthrough-period">2022–2025 年{synthetic ? '合成行情测试' : '历史测试'} · 已扣佣金和滑点</p>
      {status === 'ready' && spyStrategy && spyHalf && lowerOnBoth && <p className="walkthrough-finding">
        动态调整在两只 ETF 上的年化收益都低于半仓对照。以 SPY 为例，动态调整为 <strong>{percent(spyStrategy.cagr)}</strong>，半仓为 <strong>{percent(spyHalf.cagr)}</strong>。
      </p>}
    </div>
    {status === 'ready' ? <>
      <table className="walkthrough-table" aria-label="动态仓位策略与固定半仓比较">
        <thead><tr><th scope="col">标的与规则</th><th scope="col">年化收益</th><th scope="col">最大回撤</th></tr></thead>
        <tbody>{metrics.map(row => <tr key={`${row.symbol}-${row.strategy}`}>
          <th scope="row"><span>{row.symbol}</span> {strategies.find(item => item.id === row.strategy)?.label}</th>
          <td>{percent(row.cagr)}</td><td>{percent(row.max_drawdown)}</td>
        </tr>)}</tbody>
      </table>
      <p className="walkthrough-period">年化收益把多年的涨跌换算成每年水平。最大回撤是资金从一个高点往下跌得最多的一次。</p>
    </> : <p className="walkthrough-period" role="status">
      {status === 'loading' ? '正在读取保存的实验结果…' : '暂时无法读取对照数据，可以打开完整实验报告查看。'}
    </p>}
    <a className="walkthrough-report" href="/artifacts/dashboard.html" target="_blank" rel="noreferrer">
      查看完整实验报告 <ArrowUpRight size={15}/>
    </a>
  </section>;
}

export default function ProjectIntro({ synthetic = false }: { synthetic?: boolean }) {
  return <>
    <div className="walkthrough-grid">
      <section className="walkthrough-question" aria-labelledby="study-question-title">
        <h2 id="study-question-title">根据行情调整投资比例，<br/>能比一直维持半仓更好吗？</h2>
        <p>QuantFlow 用历史行情试跑投资规则，比较不同规则下资金怎么涨跌、最深跌了多少，以及每笔交易为什么发生。</p>
        <p>程序根据近期价格离均线有多远、总体在涨还是跌，决定拿多少钱买入，其余留现金。先用较早的行情选参数，再用 2022–2025 年的行情和「一半投资、一半现金」的规则比较。</p>
        <div className="walkthrough-assets" aria-label="研究标的">
          {symbols.map(({ ticker, name }) => <span key={ticker}><b>{ticker}</b> {name}</span>)}
        </div>
      </section>
      <StudyResults synthetic={synthetic}/>
    </div>
    <section className="walkthrough-actions" aria-labelledby="walkthrough-actions-title">
      <h2 id="walkthrough-actions-title">现场怎么演示？</h2>
      <div className="walkthrough-links">
        <a href="#/research-agent"><span><strong>让研究 Agent 核对回测</strong><small>查看两条曲线为什么相近，以及新策略是否通过验证。</small></span><ArrowRight size={17}/></a>
        <a href="#/backtest"><span><strong>自己跑一次回测</strong><small>用默认设置重算，查看资金曲线和对照。</small></span><ArrowRight size={17}/></a>
        <a href="#/portfolio"><span><strong>把多个标的放到一起</strong><small>分配同一笔资金，查看组合持仓和每笔买卖。</small></span><ArrowRight size={17}/></a>
        <a href="#/executor"><span><strong>看一笔交易</strong><small>跳到下一笔交易，查看哪天发信号、哪天成交。</small></span><ArrowRight size={17}/></a>
        <a href="#/optimization"><span><strong>看参数怎么选</strong><small>查看试过多少组参数、训练分数怎么变。</small></span><ArrowRight size={17}/></a>
      </div>
    </section>
  </>;
}
