import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Check, CheckCircle2, ChevronRight, Clock3, Copy, FlaskConical, Gauge, History, Layers3, Loader2, Play, ShieldCheck, SlidersHorizontal, Square, TrendingUp } from 'lucide-react';
import { api, request, type Row } from './api';
import PerformanceComparison from './PerformanceComparison';
import './joint-tuning.css';

export type JointTuningProps = {
  runs: Row[];
  refresh: () => Promise<void>;
  openBacktest: (genomeId: string) => void;
};

const BUDGETS = [64, 128, 256, 512, 1024];
const isRunning = (row?: Row | null) => !!row && ['queued', 'pending', 'running', 'cancelling'].includes(row.status);
const numeric = (value: unknown) => value !== null && value !== undefined && typeof value !== 'boolean' && Number.isFinite(Number(value));
const num = (value: unknown, digits = 3) => numeric(value) ? Number(value).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '—';
const pct = (value: unknown) => numeric(value) ? `${(Number(value) * 100).toFixed(2)}%` : '—';
const signed = (value: unknown, digits = 3) => numeric(value) ? `${Number(value) > 0 ? '+' : ''}${num(value, digits)}` : '—';
const methodName = (value: unknown) => value === 'ga' ? '遗传算法' : value === 'random' ? '随机搜索' : String(value || '等待分配');
const period = (value: unknown, fallback: string) => Array.isArray(value) ? value.map(item => String(item).slice(0, 10)).join(' — ') : typeof value === 'string' ? value : fallback;
const statusLabels: Record<string, string> = { queued: '等待开始', pending: '等待开始', running: '计算中', completed: '已完成', done: '已完成', failed: '失败', cancelled: '已取消', cancelling: '正在停止' };
const phaseLabels: Record<string, string> = { queued: '等待计算', training: '准备训练数据', train: '搜索训练参数', validation: '比较验证表现', completed: '计算完成', done: '计算完成' };

function Status({ task }: { task: Row }) {
  const running = isRunning(task);
  return <span className={`pill ${['completed', 'done'].includes(task.status) ? 'teal' : task.status === 'failed' ? 'red' : ''}`}>
    {running ? <Loader2 className="spin" size={12} /> : task.status === 'completed' ? <Check size={12} /> : <i />}
    {statusLabels[task.status] || task.status || '未知状态'}
  </span>;
}

function Params({ params }: { params: Row }) {
  const fields = [
    { key: 'mean_reversion', title: '均值回归力度', value: num(params.mean_reversion, 6), help: '价格偏离均线时，反向调整仓位的力度。' },
    { key: 'momentum', title: '趋势跟随力度', value: num(params.momentum, 6), help: '近期涨跌趋势对目标仓位的影响力度。' },
    { key: 'band', title: '调仓门槛', value: pct(params.band), help: '实际仓位与目标相差达到这个比例，才调整仓位。' },
  ];
  return <div className="joint-tuning-params">{fields.map(field => <div key={field.key}><span>{field.title}</span><strong>{field.value}</strong><p>{field.help}</p><code>{field.key}</code></div>)}</div>;
}

function BenchmarkCard({ label, value, help, highlight = false }: { label: string; value: string; help: string; highlight?: boolean }) {
  return <div className={`joint-tuning-benchmark ${highlight ? 'highlight' : ''}`}><span>{label}</span><strong>{value}</strong><small>{help}</small></div>;
}

export default function JointTuning({ runs = [], refresh, openBacktest }: JointTuningProps) {
  const [budget, setBudget] = useState(256);
  const [selectedId, setSelectedId] = useState('');
  const [task, setTask] = useState<Row | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [copied, setCopied] = useState(false);
  const [retry, setRetry] = useState(0);
  const finished = useRef(new Set<string>());
  const refreshRef = useRef(refresh);
  const mounted = useRef(true);
  useEffect(() => { refreshRef.current = refresh; }, [refresh]);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => { if (!selectedId && runs.length) setSelectedId(String(runs[0].id)); }, [runs, selectedId]);

  useEffect(() => {
    if (!selectedId) return;
    let alive = true;
    let timer: number | undefined;
    const row = runs.find(item => String(item.id) === selectedId);
    setTask(row || null); setLoading(true); setError(''); setCopied(false);
    const poll = async () => {
      try {
        const detail = await api.task(selectedId);
        if (!alive) return;
        setTask(detail); setLoading(false); setError('');
        if (isRunning(detail)) timer = window.setTimeout(poll, 1000);
        else if (!finished.current.has(selectedId)) {
          finished.current.add(selectedId);
          await refreshRef.current().catch(() => undefined);
        }
      } catch (reason) {
        if (!alive) return;
        setLoading(false); setError((reason as Error).message || '无法读取调参任务。');
      }
    };
    void poll();
    return () => { alive = false; if (timer !== undefined) window.clearTimeout(timer); };
    // Bootstrap polling must not restart this task poll or clear its progress.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, retry]);

  const active = runs.find(isRunning) || (isRunning(task) ? task : null);
  const start = async () => {
    if (submitting || active) return;
    setSubmitting(true); setError(''); setMessage(''); setCopied(false);
    try {
      const value = await request<Row>('/optimization/tuning', { method: 'POST', body: JSON.stringify({ budget_per_run: budget }) });
      if (!mounted.current) return;
      setTask(value); setSelectedId(String(value.id));
      setMessage(`已提交联合调参：6 轮搜索，每轮 ${budget} 个唯一候选，共 ${6 * budget} 个搜索候选。`);
      await refreshRef.current().catch(() => undefined);
    } catch (reason) { if (mounted.current) setError((reason as Error).message || '联合调参未能启动。'); }
    finally { if (mounted.current) setSubmitting(false); }
  };
  const cancel = async () => {
    if (!task || !isRunning(task) || cancelling) return;
    setCancelling(true); setError(''); setMessage('');
    try {
      await api.cancelTask(String(task.id));
      if (!mounted.current) return;
      setMessage('停止请求已提交，服务会在可中断的计算步骤结束后停止。');
      setRetry(value => value + 1);
      await refreshRef.current().catch(() => undefined);
    } catch (reason) { if (mounted.current) setError((reason as Error).message || '暂时无法停止任务。'); }
    finally { if (mounted.current) setCancelling(false); }
  };
  const winner: Row | undefined = task?.winner;
  const finalists: Row[] = task?.finalists || [];
  const benchmarks: Row = task?.benchmarks || {};
  const metric = (row?: Row): Row => row?.validation || row?.metrics || row || {};
  const half = metric(benchmarks.half), incumbent = metric(benchmarks.incumbent);
  const validation = winner?.validation || {};
  const validationStrategies = [
    { id: 'winner', label: '选定候选', metrics: validation, primary: true },
    { id: 'half', label: '固定半仓', metrics: half },
    { id: 'incumbent', label: '当前参数', metrics: incumbent },
    { id: 'fixed', label: '固定参数 Sigmoid', metrics: metric(benchmarks.fixed_sigmoid) },
    { id: 'same_band', label: '同阈值半仓', metrics: metric(benchmarks.same_band_half) },
  ].filter(strategy => ['cagr', 'total_return', 'sharpe_rf0', 'max_drawdown', 'annual_volatility']
    .some(key => numeric(strategy.metrics[key])));
  const delta = winner && numeric(validation.sharpe_rf0) && numeric(half.sharpe_rf0)
    ? Number(validation.sharpe_rf0) - Number(half.sharpe_rf0) : winner?.validation_delta;
  const progress: Row = task?.progress || {};
  const completed = Number(progress.completed) || 0;
  const total = Number(progress.total) || 0;
  const percentDone = total > 0 ? Math.max(0, Math.min(100, completed / total * 100)) : 0;
  const paramsText = winner?.params ? JSON.stringify(winner.params, null, 2) : '';
  const copy = async () => {
    try { await navigator.clipboard.writeText(paramsText); setCopied(true); setMessage('已复制完整精度的参数 JSON。'); }
    catch { setMessage('浏览器未允许自动复制，可选中下方参数文本手动复制。'); }
  };
  const isWinner = (candidate: Row) => !!winner && candidate.method === winner.method && candidate.seed === winner.seed;
  const validWinner = !!winner && !!winner.params && ['completed', 'done'].includes(task?.status);

  return <div className="joint-tuning">
    <section className="joint-tuning-intro"><span><SlidersHorizontal size={27} /></span><div><h2>一起搜索三个参数，直接看选出了哪一组</h2><p>让遗传算法和随机搜索各跑 3 次，再按验证期表现选出一组候选。你可以复制数值，或带着这组参数打开回测。</p></div><span className="pill teal">SPY · 联合调参</span></section>
    <section className="panel joint-tuning-config"><div className="panel-head"><div><span className="eyebrow">SEARCH BUDGET</span><h2>每轮比较多少组参数？</h2></div><span className="pill">2 种算法 × 3 个种子</span></div><div className="joint-tuning-budget-row"><div className="joint-tuning-budget"><span>每轮唯一候选数</span><div className="joint-tuning-presets" role="group" aria-label="选择每轮候选预算">{BUDGETS.map(value => <button type="button" key={value} className={budget === value ? 'selected' : ''} aria-pressed={budget === value} disabled={submitting || !!active} onClick={() => setBudget(value)}>{value}</button>)}</div><p>本次搜索候选总数：<strong>6 × {budget} = {(6 * budget).toLocaleString('en-US')}</strong>。另外还有验证比较步骤，候选数越多，计算越久。</p></div><button type="button" className="button primary" disabled={submitting || !!active} onClick={() => void start()}>{submitting ? <Loader2 size={16} className="spin" /> : <Play size={16} />}开始联合调参</button></div>{active && String(active.id) !== selectedId && <div className="joint-tuning-active-note"><Loader2 size={14} className="spin" /><span>另一轮联合调参正在计算，完成或停止后可以再启动。</span><button type="button" className="text-button" onClick={() => setSelectedId(String(active.id))}>查看运行任务<ArrowRight size={13} /></button></div>}<div className="joint-tuning-protocol"><ShieldCheck size={16} /><p>只用 <strong>{period(task?.train, '2010 — 2018')}</strong> 搜索，用 <strong>{period(task?.validation_period, '2019 — 2021')}</strong> 比较候选。<strong>2022 年及之后的数据不参与这次调参。</strong>验证期参与选择，不能称为独立最终测试。</p></div></section>

    {error && <div className="joint-tuning-feedback error" role="alert"><span>{error}</span>{selectedId && <button type="button" className="text-button" onClick={() => setRetry(value => value + 1)}>重新读取</button>}</div>}
    {message && <div className="joint-tuning-feedback" role="status"><CheckCircle2 size={15} /><span>{message}</span></div>}

    <div className="joint-tuning-workspace"><section className="panel joint-tuning-current"><div className="panel-head"><div><span className="eyebrow">CURRENT EXPERIMENT</span><h2>{task ? '本轮真实计算进度' : '等待开始第一轮联合调参'}</h2></div>{task && <Status task={task} />}</div>{task ? <><div className="joint-tuning-progress"><div><strong>{isRunning(task) ? phaseLabels[progress.phase] || progress.phase || '等待计算' : statusLabels[task.status] || task.status}</strong><span>{loading && <Loader2 size={12} className="spin" />}{total ? `${completed.toLocaleString('en-US')} / ${total.toLocaleString('en-US')} 计算步骤` : '等待服务返回进度'}</span></div><div className="progress-track"><div style={{ width: `${percentDone}%` }} /></div><div className="joint-tuning-phase"><span>{progress.method ? methodName(progress.method) : 'GA / 随机搜索'}{progress.seed != null ? ` · 种子 ${progress.seed}` : ''}</span><span>{total ? `${percentDone.toFixed(0)}%` : '—'}</span></div></div><div className="joint-tuning-run-facts"><div><span>每轮候选数</span><strong>{num(task.budget_per_run, 0)}</strong></div><div><span>当前最优训练分数</span><strong>{numeric(progress.best_score) && Number(progress.best_score) > -1e100 ? num(progress.best_score, 4) : '—'}</strong></div><div><span>唯一评估 / 缓存命中</span><strong>{num(task.evaluations, 0)} <small>/ {num(task.cache_hits, 0)}</small></strong></div></div>{task.error && <p className="joint-tuning-task-error" role="alert">{task.error}</p>}<div className="joint-tuning-task-bottom"><span>{String(task.created_at || '').replace('T', ' ').slice(0, 19)}{task.seeds && ` · 种子 ${task.seeds.join(' / ')}`}</span>{isRunning(task) && <button type="button" className="text-button danger" disabled={cancelling} onClick={() => void cancel()}>{cancelling ? <Loader2 size={12} className="spin" /> : <Square size={12} />}停止本轮</button>}</div></> : <div className="joint-tuning-empty"><FlaskConical size={29} /><strong>准备好预算，点击上方按钮开始</strong><p>计算进度、六个候选和选定参数都由服务实际返回，不会提前填入示例结果。</p></div>}</section>
      <section className="panel joint-tuning-history"><div className="panel-head"><div><h2>已保存的联合调参</h2><span className="joint-tuning-panel-note">点击查看历史轮次</span></div><History size={18} /></div>{runs.length ? <div>{runs.map(row => <button type="button" key={row.id} className={String(row.id) === selectedId ? 'selected' : ''} onClick={() => setSelectedId(String(row.id))}><span className="joint-tuning-history-icon"><Layers3 size={16} /></span><span><strong>{num(row.budget_per_run, 0)} / 轮 · {row.symbol || 'SPY'}</strong><small>{String(row.created_at || row.id).replace('T', ' ').slice(0, 19)}</small></span><Status task={row} /><ChevronRight size={14} /></button>)}</div> : <div className="joint-tuning-history-empty"><Clock3 size={22} /><p>启动后的任务会保存在本地，刷新页面后也能继续查看。</p></div>}</section></div>

    {validWinner && <>
      <section className="panel joint-tuning-winner"><div className="panel-head"><div><span className="eyebrow">SELECTED CANDIDATE</span><h2>这轮选出的三个数值</h2><span className="joint-tuning-panel-note">{methodName(winner.method)} · 种子 {winner.seed} · 按验证期表现选择</span></div><span className="pill teal"><CheckCircle2 size={12} />选定候选</span></div><Params params={winner.params} /><div className="joint-tuning-copy-row"><details><summary><Copy size={14} />查看并复制完整参数 JSON</summary><textarea aria-label="完整参数 JSON" readOnly value={paramsText} rows={5} onFocus={event => event.target.select()} /><button type="button" className="button secondary" onClick={() => void copy()}>{copied ? <Check size={13} /> : <Copy size={13} />}{copied ? '已复制' : '复制完整数值'}</button></details><div><button type="button" className="button primary" disabled={!task?.genome_id} onClick={() => openBacktest(String(task?.genome_id))}>用这组参数打开回测<ArrowRight size={15} /></button><small>{task?.genome_id ? '只打开回测配置，由你决定何时运行。' : '候选尚未保存为参数版本，暂不能打开回测。'}</small></div></div></section>
      <section className="panel joint-tuning-comparison"><div className="panel-head"><div><h2>验证期里，真的比简单规则更好吗？</h2><span className="joint-tuning-panel-note">同一区间、同一记账与成本口径；夏普反映收益相对于波动的大小，假设无风险利率为 0。</span></div><Gauge size={19} /></div><div className="joint-tuning-benchmarks"><BenchmarkCard label="选定候选 · 验证夏普" value={num(validation.sharpe_rf0)} help={`年化 ${pct(validation.cagr)} · 最大回撤 ${pct(validation.max_drawdown)}`} highlight /><BenchmarkCard label="固定半仓 · 验证夏普" value={num(half.sharpe_rf0)} help={`年化 ${pct(half.cagr)} · 最大回撤 ${pct(half.max_drawdown)}`} /><BenchmarkCard label="当前参数 · 验证夏普" value={num(incumbent.sharpe_rf0)} help="本轮开始时的参数对照" /><BenchmarkCard label="选定候选 − 固定半仓" value={signed(delta)} help="验证夏普差；正数表示本区间较高" highlight={numeric(delta) && Number(delta) > 0} /></div><PerformanceComparison title="验证期指标明细" strategies={validationStrategies} /><p className="joint-tuning-panel-note">年化收益看增长速度，最大回撤看历史跌幅，夏普看收益相对波动，换手和费用看交易代价。这些指标帮助理解取舍；选参仍先比较验证夏普，再依次比较回撤、换手和固定排序。</p><p className="joint-tuning-conclusion">{numeric(delta) ? Number(delta) > 0 ? `这组候选在本验证区间的夏普比固定半仓高 ${num(delta)}。这是选择候选所用的数据，仍需要后续独立验证，不能据此保证未来表现。` : `这组候选在本验证区间没有超过固定半仓的夏普。搜索选出一组参数，并不表示已经找到了比简单规则更好的策略。` : '服务未提供足够的基准数值，当前无法判断是否超过固定半仓。'}</p></section>
    </>}

    {finalists.length > 0 && <section className="panel joint-tuning-finalists"><div className="panel-head"><div><h2>本轮候选逐一比较</h2><span className="joint-tuning-panel-note">每轮训练搜索保留一个候选，再在验证期对照；选中标记由服务结果决定。</span></div><span className="pill">{finalists.length} 个真实候选</span></div><div className="table-wrap"><table><thead><tr><th>搜索方法 / 种子</th><th>均值回归力度</th><th>趋势跟随力度</th><th>调仓门槛</th><th>训练分数</th><th>验证夏普</th><th>相对半仓 Δ</th><th>验证年化收益</th><th>验证最大回撤</th><th>选择</th></tr></thead><tbody>{finalists.map((candidate, index) => {
        const difference = numeric(candidate.validation_delta) ? candidate.validation_delta : numeric(candidate.validation?.sharpe_rf0) && numeric(half.sharpe_rf0) ? Number(candidate.validation.sharpe_rf0) - Number(half.sharpe_rf0) : undefined;
        return <tr key={`${candidate.method}-${candidate.seed}-${index}`} className={isWinner(candidate) ? 'joint-tuning-selected-row' : ''}><td><strong>{methodName(candidate.method)}</strong><span className="cell-sub">种子 {candidate.seed}</span></td><td className="mono">{num(candidate.params?.mean_reversion, 6)}</td><td className="mono">{num(candidate.params?.momentum, 6)}</td><td className="mono">{pct(candidate.params?.band)}</td><td className="mono">{num(candidate.training_score, 4)}</td><td className="mono">{num(candidate.validation?.sharpe_rf0)}</td><td className={`mono ${numeric(difference) && Number(difference) > 0 ? 'positive' : ''}`}>{signed(difference)}</td><td className="mono">{pct(candidate.validation?.cagr)}</td><td className="mono">{pct(candidate.validation?.max_drawdown)}</td><td>{isWinner(candidate) ? <span className="pill teal"><Check size={11} />选定</span> : <span className="muted">对照候选</span>}</td></tr>;
      })}</tbody></table></div></section>}
      <div className="joint-tuning-footnote"><TrendingUp size={15} /><span>本页只搜索和比较参数，不自动晋升、不更改实例、不发送订单。复制或打开回测后，仍由你决定下一步。</span></div>
  </div>;
}
