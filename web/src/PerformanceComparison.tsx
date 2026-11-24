import type { Row } from './api';
import './performance-comparison.css';

type Strategy = { id: string; label: string; metrics?: Row; primary?: boolean };
type Format = 'percent' | 'signedPercent' | 'points' | 'decimal' | 'number' | 'usd' | 'multiple';
type Metric = { key: string; name: string; detail: string; format: Format };

const sections: { title: string; metrics: Metric[] }[] = [
  { title: '收益与风险', metrics: [
    { key: 'total_return', name: '累计收益', detail: '期末净值相对初始资金的涨跌幅。', format: 'signedPercent' },
    { key: 'cagr', name: '年化收益', detail: '将区间收益折算为按 252 个交易日复利增长的年收益。', format: 'signedPercent' },
    { key: 'max_drawdown', name: '最大回撤', detail: '期间净值从高点到之后低点的最大跌幅，按亏损幅度显示正数。', format: 'percent' },
    { key: 'annual_volatility', name: '年化波动率', detail: '日收益样本标准差乘以 √252。', format: 'percent' },
    { key: 'sharpe_rf0', name: '夏普比率', detail: '日均收益除以日收益样本标准差，再乘以 √252。无风险利率取 0。', format: 'decimal' },
    { key: 'sortino_rf0', name: '索提诺比率', detail: '以年化日均收益除以年化下行波动率，下行目标取 0。', format: 'decimal' },
    { key: 'downside_volatility', name: '下行波动率', detail: '只将低于 0 的日收益计为下行偏差，按全部交易日计算并年化。', format: 'percent' },
    { key: 'calmar', name: '卡玛比率', detail: '复合年化收益除以最大回撤。最大回撤为零时不计算。', format: 'decimal' },
  ] },
  { title: '相对基准', metrics: [
    { key: 'annual_excess_return', name: '年化收益差', detail: '策略复合年化收益减去参考基准的复合年化收益。', format: 'points' },
    { key: 'alpha_rf0', name: 'Alpha', detail: '日收益对参考基准日收益回归后的年化截距，无风险利率取 0。', format: 'signedPercent' },
    { key: 'beta', name: 'Beta', detail: '策略日收益相对于参考基准日收益的回归斜率。', format: 'decimal' },
    { key: 'information_ratio', name: '信息比率', detail: '年化日均超额收益除以年化跟踪误差。', format: 'decimal' },
    { key: 'tracking_error', name: '跟踪误差', detail: '策略与参考基准日收益差的年化标准差。', format: 'percent' },
    { key: 'benchmark_outperformance_rate', name: '日跑赢基准比例', detail: '策略当日收益高于参考基准的交易日占比，持平不计入。', format: 'percent' },
  ] },
  { title: '交易与资金', metrics: [
    { key: 'annual_turnover', name: '年换手', detail: '买卖成交额绝对值之和除以初始资金，再按 252 个交易日折算。', format: 'multiple' },
    { key: 'mean_exposure', name: '平均仓位', detail: '各交易日收盘时持仓市值占账户净值的平均比例。', format: 'percent' },
    { key: 'trade_count', name: '成交笔数', detail: '成交记录条数，买、卖分别计入。', format: 'number' },
    { key: 'closed_fill_count', name: '已卖出笔数', detail: '每次卖出成交计为一笔，即使匹配多个买入批次。', format: 'number' },
    { key: 'closed_fill_win_rate', name: '已卖出胜率', detail: '已卖出成交中，扣除费用后盈利的笔数占比。买入成本按先进先出匹配。', format: 'percent' },
    { key: 'realized_profit_factor', name: '已实现盈亏比', detail: '已卖出成交的盈利之和除以亏损绝对值之和。计入佣金和滑点，分红与未卖出份额单列。', format: 'multiple' },
    { key: 'daily_win_rate', name: '盈利日比例', detail: '账户收盘净值比上一交易日上升的日数占比。', format: 'percent' },
    { key: 'total_commission', name: '累计佣金', detail: '所有成交支付的佣金，已计入净值。', format: 'usd' },
    { key: 'slippage_cost', name: '累计滑点成本', detail: '实际模拟成交价格相对参考开盘价的不利差额，已计入净值。', format: 'usd' },
    { key: 'dividend_entitlement', name: '累计分红应计额', detail: '除息日确认应收的分红总额，含尚未到账部分。', format: 'usd' },
    { key: 'initial_cash', name: '初始资金', detail: '该策略及对照用同一起始资金回测。', format: 'usd' },
    { key: 'final_equity', name: '期末净值', detail: '期末持仓市值、现金和应收分红合计。', format: 'usd' },
  ] },
];

function formatValue(value: unknown, format: Format): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  if (format === 'percent' || format === 'signedPercent' || format === 'points') {
    const prefix = format !== 'percent' && value > 0 ? '+' : '';
    const units = format === 'points' ? ' 个百分点' : '%';
    return `${prefix}${(value * 100).toFixed(2)}${units}`;
  }
  if (format === 'usd') return value.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
  if (format === 'multiple') return `${value.toFixed(2)} 倍`;
  if (format === 'number') return value.toLocaleString('en-US', { maximumFractionDigits: 0 });
  return value.toFixed(3);
}

function sessionText(metrics: Row): string {
  const sessions = formatValue(metrics.sessions, 'number');
  return metrics.start && metrics.end ? `${metrics.start} 至 ${metrics.end} · ${sessions} 个交易日` : '';
}

export default function PerformanceComparison({ title = '策略表现对比', strategies }: {
  title?: string;
  strategies: Strategy[];
}) {
  const available = strategies.filter((strategy): strategy is Strategy & { metrics: Row } =>
    strategy.metrics != null && typeof strategy.metrics.total_return === 'number');
  if (!available.length) return null;

  const extended = available.some(({ metrics }) => Object.hasOwn(metrics, 'downside_volatility'));
  const periods = [...new Set(available.map(strategy => sessionText(strategy.metrics)))];
  return <section className="panel performance-comparison">
    <div className="panel-head"><div><h2>{title}</h2>
      {periods.length === 1 && periods[0] && <p className="performance-caption">{periods[0]}</p>}
    </div></div>
    <div className="table-wrap" role="region" tabIndex={0} aria-label={`${title}，可横向滚动`}>
      <table>
        <thead><tr><th scope="col">指标</th>{available.map(strategy =>
          <th scope="col" key={strategy.id} className={strategy.primary ? 'performance-primary' : ''}>
            {strategy.label}{periods.length > 1 && <small>{sessionText(strategy.metrics)}</small>}
          </th>)}</tr></thead>
        {sections.map(section => <tbody key={section.title}>
          <tr className="performance-category"><th scope="rowgroup" colSpan={available.length + 1}>{section.title}</th></tr>
          {section.metrics.map(metric => <tr key={metric.key}>
            <th scope="row" title={metric.detail}>{metric.name}</th>
            {available.map(strategy => <td key={strategy.id} className={`mono ${strategy.primary ? 'performance-primary' : ''}`}>
              {formatValue(strategy.metrics[metric.key], metric.format)}
            </td>)}
          </tr>)}
        </tbody>)}
      </table>
    </div>
    <p className="performance-caption performance-note">
      {extended ? '相对指标统一以固定半仓为参考。— 表示缺少数据，或指标在本区间无定义。' : '重新运行这份历史配置可计算相对指标和下行风险。— 表示该记录未保存指标。'}
    </p>
    <details className="performance-definitions">
      <summary>指标口径与来源</summary>
      <p>指标类别参考聚宽公开的回测风险指标。这里用 QuantFlow 的 252 日年化、零无风险利率和成本后每日净值。外部 Alpha 提交条件以平台自己的规则为准。</p>
      <dl>{sections.flatMap(section => section.metrics).map(metric =>
        <div key={metric.key}><dt>{metric.name}</dt><dd>{metric.detail}</dd></div>)}</dl>
      <a href="https://cdn.joinquant.com/help/img/JoinQuantAPI.pdf" target="_blank" rel="noreferrer">聚宽公开 API 文档 · 风险指标</a>
    </details>
  </section>;
}
