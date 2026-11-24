import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from 'react';
import {
  ArrowDownLeft,
  ArrowRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Download,
  Layers3,
  Loader2,
  Play,
  Scale,
  SlidersHorizontal,
  Wallet,
} from 'lucide-react';
import { api, type Row } from './api';
import PerformanceComparison from './PerformanceComparison';
import './portfolio.css';

type Dataset = {
  symbol: string;
  name?: string;
  first_session?: string;
  last_session?: string;
};
export type Props = {
  datasets: Dataset[];
  genomes: Row[];
  backtests: Row[];
  notify: (text: string, error?: boolean) => void;
  refresh: () => Promise<void>;
  renderChart: (rows: Row[]) => ReactNode;
};

const number = (value: any, digits = 2) =>
  value !== undefined && value !== null && Number.isFinite(Number(value))
    ? Number(value).toLocaleString('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : '—';
const percent = (value: any, signed = false) =>
  value !== undefined && value !== null && Number.isFinite(Number(value))
    ? `${signed && Number(value) > 0 ? '+' : ''}${(Number(value) * 100).toFixed(2)}%`
    : '—';
const date = (value: any) => (value ? String(value).slice(0, 10) : '—');
const waiting = (row: Row | null) =>
  !!row && ['pending', 'queued', 'running'].includes(row.status);
const statuses: Record<string, string> = {
  pending: '等待计算',
  queued: '排队中',
  running: '正在回测',
  completed: '已完成',
  done: '已完成',
  failed: '计算失败',
  cancelled: '已取消',
  interrupted: '已中断',
};
const labels: Record<string, string> = { SPY: '标普 500 ETF', GLD: '黄金 ETF' };

function equalWeights(symbols: string[]) {
  if (!symbols.length) return {};
  const each = Math.floor(10000 / symbols.length) / 100;
  return Object.fromEntries(
    symbols.map((symbol, index) => [
      symbol,
      index === symbols.length - 1
        ? Number((100 - each * index).toFixed(2))
        : each,
    ]),
  );
}
function seedGenome(genomes: Row[]) {
  return genomes.find((row) => row.role === 'champion') || genomes[0];
}
function Label({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}
function Status({ row }: { row: Row }) {
  return (
    <span
      className={`pill ${row.status === 'completed' || row.status === 'done' ? 'teal' : row.status === 'failed' ? 'red' : ''}`}
    >
      {waiting(row) ? <Loader2 size={12} className="spin" /> : <i />}
      {statuses[row.status] || row.status || '历史结果'}
    </span>
  );
}
export default function PortfolioBacktest({
  datasets,
  genomes,
  backtests,
  notify,
  refresh,
  renderChart,
}: Props) {
  const available = useMemo(
    () =>
      datasets.filter(
        (row, index, all) =>
          row.symbol &&
          all.findIndex((other) => other.symbol === row.symbol) === index,
      ),
    [datasets],
  );
  const [weights, setWeights] = useState<Record<string, number>>(() =>
    equalWeights(available.slice(0, 2).map((row) => row.symbol)),
  );
  const assetName = (symbol: string) =>
    available.find((row) => row.symbol === symbol)?.name ||
    labels[symbol] ||
    '已保存的历史数据';
  const [strategy, setStrategy] = useState('sigmoid');
  const [genomeId, setGenomeId] = useState(() =>
    String(seedGenome(genomes)?.id || ''),
  );
  const [form, setForm] = useState({
    start: '2022-01-01',
    end: '2025-12-31',
    initial_cash: 100000,
    commission_bps: 1,
    slippage_bps: 5,
  });
  const [result, setResult] = useState<Row | null>(null);
  const [resultId, setResultId] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [filterSymbol, setFilterSymbol] = useState('all');
  const [filterSide, setFilterSide] = useState('all');
  const [tradePage, setTradePage] = useState(0);
  const [localError, setLocalError] = useState('');
  const finished = useRef(new Set<string>());
  const refreshRef = useRef(refresh);
  const notifyRef = useRef(notify);
  useEffect(() => {
    refreshRef.current = refresh;
    notifyRef.current = notify;
  }, [refresh, notify]);

  useEffect(() => {
    if (!genomeId && genomes.length)
      setGenomeId(String(seedGenome(genomes)?.id || ''));
  }, [genomes, genomeId]);
  useEffect(() => {
    if (!resultId && backtests.length) setResultId(String(backtests[0].id));
  }, [backtests, resultId]);

  useEffect(() => {
    if (!resultId) return;
    let alive = true;
    let timer: number | undefined;
    setDetailLoading(true);
    const poll = async () => {
      try {
        const row = await api.backtestDetail(resultId);
        if (!alive) return;
        setResult(row);
        setLocalError('');
        setDetailLoading(false);
        if (waiting(row)) timer = window.setTimeout(poll, 1000);
        else if (!finished.current.has(resultId)) {
          finished.current.add(resultId);
          await refreshRef.current().catch(() => undefined);
        }
      } catch (error) {
        if (!alive) return;
        setDetailLoading(false);
        const message = (error as Error).message;
        setLocalError(message);
        notifyRef.current(message, true);
      }
    };
    void poll();
    return () => {
      alive = false;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [resultId]);

  const selectedSymbols = available
    .filter((row) => Object.hasOwn(weights, row.symbol))
    .map((row) => row.symbol);
  const totalWeight = selectedSymbols.reduce(
    (sum, symbol) => sum + (Number(weights[symbol]) || 0),
    0,
  );
  const totalValid = Math.abs(totalWeight - 100) < 0.00001;
  const selectedDatasets = available.filter((row) =>
    selectedSymbols.includes(row.symbol),
  );
  const firstDate =
    selectedDatasets
      .map((row) => row.first_session || '')
      .sort()
      .at(-1) || undefined;
  const lastDate =
    selectedDatasets.map((row) => row.last_session || '9999-12-31').sort()[0] ||
    undefined;
  const selectedGenome = genomes.find((row) => String(row.id) === genomeId);
  const switchSymbol = (symbol: string) => {
    const next = selectedSymbols.includes(symbol)
      ? selectedSymbols.filter((value) => value !== symbol)
      : [...selectedSymbols, symbol];
    setWeights(equalWeights(next));
  };
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (selectedSymbols.length < 2) {
      notify('请至少选择两个标的。', true);
      return;
    }
    if (
      !totalValid ||
      selectedSymbols.some(
        (symbol) => !Number.isFinite(weights[symbol]) || weights[symbol] <= 0,
      )
    ) {
      notify('每个标的份额应大于 0，合计为 100%。', true);
      return;
    }
    if (
      form.start > form.end ||
      (firstDate && form.start < firstDate) ||
      (lastDate && form.end > lastDate)
    ) {
      notify('请选择所有所选标的都有历史数据的日期区间。', true);
      return;
    }
    if (!genomeId) {
      notify('请先选择一份动态策略参数，便于同时计算对照结果。', true);
      return;
    }
    setSubmitting(true);
    setLocalError('');
    try {
      const allocations = Object.fromEntries(
        selectedSymbols.map((symbol) => [
          symbol,
          Number(weights[symbol]) / 100,
        ]),
      );
      const row = await api.backtest({
        allocations,
        strategy,
        genome_id: genomeId,
        ...form,
      });
      setResult(row);
      setResultId(String(row.id));
      await refreshRef.current();
      notify('组合回测已提交，将同时比较动态策略与固定半仓。');
    } catch (error) {
      const message = (error as Error).message;
      setLocalError(message);
      notify(message, true);
    } finally {
      setSubmitting(false);
    }
  };

  const resultAllocations: Row = result?.allocations || {};
  const resultSymbols = Object.keys(resultAllocations);
  const allTrades: Row[] = result?.trades || [];
  const filteredTrades = useMemo(
    () =>
      allTrades
        .filter(
          (row) =>
            (filterSymbol === 'all' || row.symbol === filterSymbol) &&
            (filterSide === 'all' ||
              (filterSide === 'buy'
                ? Number(row.quantity) > 0
                : Number(row.quantity) < 0)),
        )
        .map((row, index) => ({ row, index }))
        .sort(
          (a, b) =>
            String(b.row.date).localeCompare(String(a.row.date)) ||
            b.index - a.index,
        )
        .map((item) => item.row),
    [allTrades, filterSymbol, filterSide],
  );
  const totalPages = Math.max(1, Math.ceil(filteredTrades.length / 8));
  const currentTradePage = Math.min(tradePage, totalPages - 1);
  useEffect(() => {
    setTradePage(0);
  }, [filterSymbol, filterSide]);
  useEffect(() => {
    setTradePage(0);
    setFilterSymbol('all');
    setFilterSide('all');
  }, [resultId]);
  const dynamicMetrics =
    result?.baseline_metrics?.sigmoid ||
    (result?.strategy === 'sigmoid' ? result?.metrics : undefined);
  const halfMetrics =
    result?.baseline_metrics?.half ||
    (result?.strategy === 'half' ? result?.metrics : undefined);
  const portfolio: Row | undefined = result?.portfolio;
  const holdings: Row = portfolio?.holdings || {};
  const resultStrategy = result?.strategy === 'half' ? '固定半仓' : '动态策略';
  const chart: Row[] = result?.chart || [];
  const ready = !!result && !waiting(result) && !!result.metrics;
  const busy = submitting || waiting(result);
  const allocationNote = useCallback(
    (allocations: Row) =>
      Object.entries(allocations)
        .map(
          ([symbol, allocation]) =>
            `${symbol} ${Number((Number(allocation) * 100).toFixed(2))}%`,
        )
        .join(' / '),
    [],
  );

  return (
    <div className="portfolio-backtest">
      <section className="portfolio-intro">
        <span className="portfolio-intro-icon">
          <Layers3 size={23} />
        </span>
        <div>
          <h2>把一笔资金，分给多个标的一起研究</h2>
          <p>
            先分配各标的的预算份额，再让策略决定实际买入多少。所有标的共用一个现金账户。
          </p>
        </div>
        <span className="pill teal">多标的 · 共享现金</span>
      </section>

      <section className="panel portfolio-config">
        <div className="panel-head">
          <div>
            <span className="eyebrow">PORTFOLIO BACKTEST</span>
            <h2>设置组合与交易规则</h2>
          </div>
          <button
            type="button"
            className="text-button"
            onClick={() => setHistoryOpen((value) => !value)}
            aria-expanded={historyOpen}
          >
            <Clock3 size={15} />
            组合历史记录<span className="count-badge">{backtests.length}</span>
            <ChevronDown size={13} />
          </button>
        </div>
        {historyOpen && (
          <div className="portfolio-history">
            {backtests.length ? (
              backtests.map((row) => (
                <button
                  key={row.id}
                  type="button"
                  className={resultId === String(row.id) ? 'selected' : ''}
                  onClick={() => {
                    setResultId(String(row.id));
                    setHistoryOpen(false);
                  }}
                >
                  <span>
                    <strong>
                      {allocationNote(row.allocations || {}) ||
                        row.symbol ||
                        '多标的组合'}
                    </strong>
                    <small>
                      {date(row.start || row.metrics?.start)} —{' '}
                      {date(row.end || row.metrics?.end)} ·{' '}
                      {row.strategy === 'half' ? '固定半仓' : '动态策略'}
                    </small>
                  </span>
                  <Status row={row} />
                  <ChevronRight size={15} />
                </button>
              ))
            ) : (
              <p>组合回测完成后会自动保存到这里。</p>
            )}
          </div>
        )}
        <form onSubmit={submit}>
          <div className="portfolio-form-section">
            <div className="portfolio-section-heading">
              <span className="portfolio-step">1</span>
              <div>
                <h3>选择标的，分配预算</h3>
                <p>至少选择两个已保存的标的，份额合计 100%。</p>
              </div>
              <button
                type="button"
                className="text-button"
                disabled={!selectedSymbols.length}
                onClick={() => setWeights(equalWeights(selectedSymbols))}
              >
                <Scale size={14} />
                等权分配
              </button>
            </div>
            <div className="portfolio-symbols">
              {available.map((dataset) => {
                const selected = selectedSymbols.includes(dataset.symbol);
                return (
                  <div
                    key={dataset.symbol}
                    className={`portfolio-symbol ${selected ? 'selected' : ''}`}
                  >
                    <label className="portfolio-symbol-toggle">
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => switchSymbol(dataset.symbol)}
                      />
                      <span className="portfolio-checkbox">
                        {selected && <Check size={13} />}
                      </span>
                      <span>
                        <strong>{dataset.symbol}</strong>
                        <small>{assetName(dataset.symbol)}</small>
                      </span>
                    </label>
                    {selected ? (
                      <div className="portfolio-weight">
                        <label>
                          <span className="sr-only">
                            {dataset.symbol} 预算份额百分比
                          </span>
                          <input
                            type="number"
                            min={0.01}
                            max={100}
                            step={0.01}
                            required
                            value={
                              Number.isNaN(weights[dataset.symbol])
                                ? ''
                                : weights[dataset.symbol]
                            }
                            onChange={(event) =>
                              setWeights((value) => ({
                                ...value,
                                [dataset.symbol]:
                                  event.target.value === ''
                                    ? NaN
                                    : Number(event.target.value),
                              }))
                            }
                          />
                          <span>%</span>
                        </label>
                        <small>
                          初始预算 $
                          {number(
                            (Number(form.initial_cash) *
                              (weights[dataset.symbol] || 0)) /
                              100,
                          )}
                        </small>
                      </div>
                    ) : (
                      <span className="portfolio-symbol-inactive">
                        点击加入组合
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
            {!available.length && (
              <p className="portfolio-inline-error">
                尚无已保存的数据集，请先准备数据。
              </p>
            )}
            <div
              className={`portfolio-weight-total ${totalValid ? 'valid' : 'invalid'}`}
              role="status"
            >
              <span>{selectedSymbols.length} 个标的 · 预算合计</span>
              <strong>
                {number(totalWeight)}%{' '}
                {totalValid ? <Check size={14} /> : <small>需为 100%</small>}
              </strong>
            </div>
            <div className="portfolio-example">
              <strong>预算份额不是实际持仓</strong>
              <p>
                例如预算为 SPY 50%、GLD 50%，若动态规则让两者分别投入各自预算的
                60% 和 40%，整个账户就是 <b>SPY 30% + GLD 20% + 现金 50%</b>
                。固定半仓则各投入一半预算，即 SPY 25%、GLD 25%、现金
                50%。这是计算示例，实际结果以回测账本为准。
              </p>
            </div>
          </div>
          <div className="portfolio-form-section">
            <div className="portfolio-section-heading">
              <span className="portfolio-step">2</span>
              <div>
                <h3>设置规则、时间与费用</h3>
                <p>两套规则使用相同预算、时间区间和交易成本，同时计算对照。</p>
              </div>
            </div>
            <div className="portfolio-strategy-row">
              <Label label="主策略">
                <select
                  value={strategy}
                  onChange={(event) => setStrategy(event.target.value)}
                >
                  <option value="sigmoid">动态策略 · 根据信号调整仓位</option>
                  <option value="half">固定半仓 · 各预算投入一半</option>
                </select>
              </Label>
              <Label
                label="动态策略使用的参数"
                hint="这组参数只用于动态策略，两种规则都会计算。"
              >
                <select
                  required
                  value={genomeId}
                  onChange={(event) => setGenomeId(event.target.value)}
                >
                  {!genomes.length && (
                    <option value="">暂无参数，请先完成参数实验</option>
                  )}
                  {genomes.map((row) => (
                    <option key={row.id} value={String(row.id)}>
                      {row.name || row.id}
                      {row.role === 'champion' ? ' · 当前冠军' : ''}
                    </option>
                  ))}
                </select>
              </Label>
            </div>
            {selectedGenome?.params && (
              <div className="portfolio-params">
                <SlidersHorizontal size={14} />
                <span>
                  本次动态参数：均值回归{' '}
                  {number(selectedGenome.params.mean_reversion, 3)} · 动量{' '}
                  {number(selectedGenome.params.momentum, 3)} · 调仓带{' '}
                  {percent(selectedGenome.params.band)}
                </span>
              </div>
            )}
            <div className="portfolio-settings">
              <Label label="初始总资金（USD）">
                <input
                  required
                  type="number"
                  min={1000}
                  max={10000000}
                  step={100}
                  value={form.initial_cash}
                  onChange={(event) =>
                    setForm((value) => ({
                      ...value,
                      initial_cash: Number(event.target.value),
                    }))
                  }
                />
              </Label>
              <Label label="开始日期">
                <input
                  required
                  type="date"
                  min={firstDate}
                  max={lastDate}
                  value={form.start}
                  onChange={(event) =>
                    setForm((value) => ({
                      ...value,
                      start: event.target.value,
                    }))
                  }
                />
              </Label>
              <Label label="结束日期">
                <input
                  required
                  type="date"
                  min={form.start > (firstDate || '') ? form.start : firstDate}
                  max={lastDate}
                  value={form.end}
                  onChange={(event) =>
                    setForm((value) => ({ ...value, end: event.target.value }))
                  }
                />
              </Label>
              <Label label="每边佣金（基点）">
                <input
                  required
                  type="number"
                  min={0}
                  max={100}
                  step={0.1}
                  value={form.commission_bps}
                  onChange={(event) =>
                    setForm((value) => ({
                      ...value,
                      commission_bps: Number(event.target.value),
                    }))
                  }
                />
              </Label>
              <Label label="每边滑点（基点）">
                <input
                  required
                  type="number"
                  min={0}
                  max={200}
                  step={0.1}
                  value={form.slippage_bps}
                  onChange={(event) =>
                    setForm((value) => ({
                      ...value,
                      slippage_bps: Number(event.target.value),
                    }))
                  }
                />
              </Label>
            </div>
            <div className="portfolio-submit">
              <span>1 基点 = 0.01% · 整股交易 · 无杠杆、无追加资金</span>
              <button
                className="button primary"
                disabled={
                  busy || !totalValid || selectedSymbols.length < 2 || !genomeId
                }
              >
                {busy ? (
                  <Loader2 size={16} className="spin" />
                ) : (
                  <Play size={16} />
                )}
                运行组合回测
              </button>
            </div>
          </div>
        </form>
      </section>

      {localError && (
        <div className="notice error" role="alert">
          {localError}
        </div>
      )}
      {detailLoading && !result && (
        <div className="portfolio-empty">
          <Loader2 size={25} className="spin" />
          <strong>正在读取已保存的组合账本</strong>
        </div>
      )}
      {result && (
        <div className="portfolio-result-context">
          <div>
            <span className="live-square" />
            <strong>
              {allocationNote(resultAllocations) || result.symbol || '组合回测'}
            </strong>
            <Status row={result} />
          </div>
          <span>
            {date(result.start || result.metrics?.start)} —{' '}
            {date(result.end || result.metrics?.end)} · 历史模拟
          </span>
        </div>
      )}
      {result?.error && (
        <div className="notice error" role="alert">
          {result.error}
        </div>
      )}
      {waiting(result) && (
        <div className="portfolio-calculating">
          <Loader2 size={22} className="spin" />
          <div>
            <strong>正在计算两套规则的真实历史账本</strong>
            <p>完成后会显示组合权益、分标的持仓和逐笔成交，记录保存在本地。</p>
          </div>
        </div>
      )}

      {ready && (
        <>
          <div className="portfolio-result-title">
            <div>
              <h2>同一笔资金，两种规则会怎样？</h2>
              <p>
                下列数值均扣除本次设置的费用；持仓明细和成交表对应你选择的“
                {resultStrategy}”。
              </p>
            </div>
          </div>
          <PerformanceComparison
            title="组合策略表现对比"
            strategies={[
              {
                id: 'sigmoid',
                label: '动态仓位组合',
                metrics: dynamicMetrics,
                primary: result.strategy === 'sigmoid',
              },
              {
                id: 'half',
                label: '固定半仓',
                metrics: halfMetrics,
                primary: result.strategy === 'half',
              },
            ]}
          />
          <section className="panel portfolio-result-chart">
            <div className="panel-head">
              <div>
                <h2>组合资金变化</h2>
                <span className="portfolio-panel-sub">
                  同一初始本金、同一交易区间与成本
                </span>
              </div>
              <span className="pill">动态策略 / 固定半仓</span>
            </div>
            {chart.length ? (
              renderChart(chart)
            ) : (
              <div className="portfolio-empty">
                <Layers3 size={24} />
                <strong>结果尚无曲线数据</strong>
              </div>
            )}
          </section>

          <section className="panel portfolio-holdings">
            <div className="panel-head">
              <div>
                <h2>期末的钱，分别在哪里？</h2>
                <span className="portfolio-panel-sub">
                  {resultStrategy} ·{' '}
                  {date(portfolio?.date || result.metrics?.end)} ·
                  预算份额与实际持仓分开显示
                </span>
              </div>
              <Wallet size={20} />
            </div>
            {portfolio ? (
              <>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>资产</th>
                        <th>预算份额</th>
                        <th>期末份额</th>
                        <th>期末价格</th>
                        <th>期末市值</th>
                        <th>占总资产</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(holdings).map(
                        ([symbol, item]: [string, any]) => (
                          <tr key={symbol}>
                            <td>
                              <strong>{symbol}</strong>
                              <span className="cell-sub">
                                {assetName(symbol)}
                              </span>
                            </td>
                            <td className="mono">
                              {percent(resultAllocations[symbol])}
                            </td>
                            <td className="mono">{number(item.shares, 0)}</td>
                            <td className="mono">
                              ${number(item.close ?? item.price)}
                            </td>
                            <td className="mono">
                              ${number(item.market_value)}
                            </td>
                            <td className="mono">{percent(item.weight)}</td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="portfolio-cash">
                  <div>
                    <Wallet size={18} />
                    <span>共享可用现金</span>
                    <strong>${number(portfolio.cash)}</strong>
                  </div>
                  <div>
                    <span>分红应收</span>
                    <strong>${number(portfolio.receivable)}</strong>
                  </div>
                  <div>
                    <span>总资产</span>
                    <strong>${number(portfolio.equity)}</strong>
                  </div>
                </div>
                <p className="portfolio-account-note">
                  总资产 = 各标的市值 + 共享现金 +
                  分红应收。没有买入的预算留在共享现金账户内。
                </p>
              </>
            ) : (
              <div className="portfolio-empty">
                <Wallet size={24} />
                <strong>此结果未包含期末持仓明细</strong>
              </div>
            )}
          </section>

          <section className="panel portfolio-trades">
            <div className="panel-head">
              <div>
                <h2>各标的实际买卖记录</h2>
                <span className="portfolio-panel-sub">
                  {resultStrategy} · 按最新成交在前显示
                </span>
              </div>
              <a
                className="button secondary small-button"
                href={`/api/backtests/${encodeURIComponent(String(result.id))}/export`}
                download
              >
                <Download size={14} />
                导出全部成交 CSV
              </a>
            </div>
            <div className="portfolio-trade-filters">
              <label>
                <span>标的</span>
                <select
                  aria-label="筛选成交标的"
                  value={filterSymbol}
                  onChange={(event) => setFilterSymbol(event.target.value)}
                >
                  <option value="all">所有标的</option>
                  {resultSymbols.map((symbol) => (
                    <option key={symbol} value={symbol}>
                      {symbol}
                    </option>
                  ))}
                </select>
              </label>
              <div className="segmented">
                {[
                  ['all', '全部'],
                  ['buy', '买入'],
                  ['sell', '卖出'],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    className={filterSide === value ? 'selected' : ''}
                    onClick={() => setFilterSide(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <span>共 {filteredTrades.length} 笔</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>成交日期</th>
                    <th>标的</th>
                    <th>方向</th>
                    <th>份额</th>
                    <th>成交价格</th>
                    <th>成交金额</th>
                    <th>佣金</th>
                    <th>信号日期</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTrades
                    .slice(currentTradePage * 8, currentTradePage * 8 + 8)
                    .map((row, index) => (
                      <tr key={`${row.date}-${row.symbol}-${index}`}>
                        <td className="mono">{date(row.date)}</td>
                        <td>
                          <strong>{row.symbol}</strong>
                        </td>
                        <td>
                          <span
                            className={`pill ${Number(row.quantity) > 0 ? 'teal' : 'coral'}`}
                          >
                            {Number(row.quantity) > 0 ? (
                              <ArrowDownLeft size={12} />
                            ) : (
                              <ArrowUpRight size={12} />
                            )}
                            {Number(row.quantity) > 0 ? '买入' : '卖出'}
                          </span>
                        </td>
                        <td className="mono">
                          {number(Math.abs(Number(row.quantity)), 0)}
                        </td>
                        <td className="mono">${number(row.price)}</td>
                        <td className="mono">
                          ${number(Math.abs(Number(row.notional)))}
                        </td>
                        <td className="mono">${number(row.commission)}</td>
                        <td className="mono muted">{date(row.signal_date)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
            {!filteredTrades.length && (
              <div className="portfolio-empty">
                <ArrowRight size={24} />
                <strong>当前筛选下没有成交</strong>
                <p>
                  未成交的日期也会记录每日资产，账户可能保持现金或原有持仓。
                </p>
              </div>
            )}
            <div className="pagination">
              <span>每页 8 笔 · 共 {filteredTrades.length} 笔</span>
              <div>
                <button
                  type="button"
                  className="icon-button bordered"
                  aria-label="上一页成交"
                  disabled={currentTradePage === 0}
                  onClick={() =>
                    setTradePage((value) => Math.max(0, value - 1))
                  }
                >
                  <ChevronLeft size={16} />
                </button>
                <span>
                  {currentTradePage + 1} / {totalPages}
                </span>
                <button
                  type="button"
                  className="icon-button bordered"
                  aria-label="下一页成交"
                  disabled={currentTradePage >= totalPages - 1}
                  onClick={() => setTradePage((value) => value + 1)}
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          </section>
          <div className="portfolio-footnote">
            <Clock3 size={14} />
            <span>
              使用所选标的的共同交易日。收盘后生成信号，下一交易日开盘模拟成交；分红现金使用
              35 天到账代理。以上为历史研究结果。
            </span>
          </div>
        </>
      )}
      {!result && !detailLoading && (
        <div className="portfolio-empty portfolio-first-run">
          <span>
            <Layers3 size={27} />
          </span>
          <strong>先分配资金，再看两种规则的实际差别</strong>
          <p>
            运行后会保存两套策略的对照结果，并展示所选主策略的分标的持仓、共享现金和成交记录。
          </p>
        </div>
      )}
    </div>
  );
}
