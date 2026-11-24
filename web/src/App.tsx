import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { Activity, ArrowDown, ArrowDownLeft, ArrowRight, ArrowUpRight, BarChart3, Boxes, Check, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, CircleDot, Clock3, Cpu, Database, Download, FlaskConical, FolderKanban, Gauge, Layers3, LayoutDashboard, Loader2, Pause, Play, Plus, RefreshCw, Search, Settings2, ShieldCheck, SkipForward, SlidersHorizontal, Sparkles, Square, TrendingUp, Waves, X, Zap, type LucideIcon } from 'lucide-react';
import { api, type Row } from './api';
import ProjectIntro from './ProjectIntro';
import PortfolioBacktest from './PortfolioBacktest';
import ResearchAgent from './ResearchAgent';
import JointTuning from './JointTuning';
import PerformanceComparison from './PerformanceComparison';

type Page = 'overview' | 'templates' | 'instances' | 'optimization' | 'executor' | 'backtest' | 'portfolio' | 'research-agent';
const NAV: { id: Page; name: string; icon: LucideIcon; sub: string }[] = [
  { id: 'overview', name: '项目总览', icon: LayoutDashboard, sub: '研究问题、测试结果和现场演示。' },
  { id: 'templates', name: '策略模板', icon: Layers3, sub: '查看每条规则在什么情况下会买入或卖出。' },
  { id: 'instances', name: '策略实例', icon: Boxes, sub: '保存一套配置，包括买什么、用哪条规则和投入多少钱。' },
  { id: 'optimization', name: '参数优化', icon: FlaskConical, sub: '用较早的行情试不同参数，再用另一段行情检查结果。' },
  { id: 'executor', name: '历史回放', icon: Cpu, sub: '从每笔模拟成交追溯到触发它的交易信号。' },
  { id: 'backtest', name: '回测分析', icon: BarChart3, sub: '用过去的价格模拟一次投资，并和简单规则比较。' },
  { id: 'portfolio', name: '多标的回测', icon: Layers3, sub: '用同一笔现金，同时投资多个标的。' },
  { id: 'research-agent', name: '研究 Agent', icon: Cpu, sub: '按步骤复跑策略、核对账本和检查验证结果。' },
];
const money = (v: any, digits = 2) => Number.isFinite(Number(v)) ? Number(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '—';
const pct = (v: any, signed = false) => Number.isFinite(Number(v)) ? `${signed && Number(v) > 0 ? '+' : ''}${(Number(v) * 100).toFixed(2)}%` : '—';
const dateLabel = (value: any) => value ? String(value).slice(0, 10) : '—';
const statusText: Row = { running: '运行中', paused: '已暂停', stopped: '未启动', completed: '已完成', done: '已完成', failed: '失败', cancelled: '已取消', queued: '排队中', pending: '等待中', interrupted: '已中断' };
const activeTask = (task: Row | null) => task && ['running', 'queued', 'pending'].includes(task.status);
const paramsLabel = (params?: Row) => params ? `MR ${Number(params.mean_reversion || 0).toFixed(3)}  ·  MOM ${Number(params.momentum || 0).toFixed(3)}  ·  BAND ${pct(params.band || 0)}` : '固定规则';

function Pill({ children, tone = '', dot = false }: { children: ReactNode; tone?: string; dot?: boolean }) {
  return <span className={`pill ${tone}`}>{dot && <i />}{children}</span>;
}
function Status({ value, instance = false }: { value?: string; instance?: boolean }) { return <Pill tone={value === 'running' || value === 'completed' ? 'teal' : value === 'failed' ? 'red' : ''} dot>{instance && value==='running' ? '已启用' : statusText[value || ''] || value || '等待中'}</Pill>; }
function Empty({ icon: Icon = Database, title, description, action }: { icon?: LucideIcon; title: string; description?: string; action?: ReactNode }) {
  return <div className="empty"><div className="empty-icon"><Icon size={25}/></div><h3>{title}</h3>{description && <p>{description}</p>}{action}</div>;
}
function Panel({ title, eyebrow, action, children, className = '' }: { title?: string; eyebrow?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{(title || action) && <div className="panel-head"><div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2></div>{action}</div>}{children}</section>;
}
function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return <label className="field"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>;
}
function Kpi({ label, value, unit, note, icon: Icon, tone = '', children }: { label: string; value: string; unit?: string; note?: string; icon: LucideIcon; tone?: string; children?: ReactNode }) {
  return <div className={`kpi ${tone}`}><div className="kpi-label">{label}<Icon size={17}/></div><div className="kpi-value">{unit && <span>{unit}</span>}{value}</div><div className="kpi-foot">{children || note || '历史回测数据'}</div><div className="kpi-decor"/></div>;
}

function NavChart({ rows = [], height = 280, title = '账户资金随时间变化', compact = false, portfolio = false, sameAs }: { rows?: Row[]; height?: number; title?: string; compact?: boolean; portfolio?: boolean; sameAs?: 'half' | 'benchmark' }) {
  const [range, setRange] = useState('ALL');
  const [hover, setHover] = useState<number | null>(null);
  const [visible, setVisible] = useState({ equity: true, benchmark: sameAs === 'half', half: true });
  const container = useRef<HTMLDivElement>(null);
  const allRows = rows.filter(row => Number.isFinite(Number(row.equity)));
  const months: Record<string, number> = { '1M': 1, '3M': 3, '1Y': 12 };
  let cutoff = '';
  if (months[range] && allRows.length) {
    const end = new Date(`${dateLabel(allRows[allRows.length - 1].date)}T00:00:00Z`);
    const start = new Date(end); start.setUTCDate(1); start.setUTCMonth(start.getUTCMonth() - months[range]);
    const monthDays = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + 1, 0)).getUTCDate();
    start.setUTCDate(Math.min(end.getUTCDate(), monthDays)); cutoff = start.toISOString().slice(0, 10);
  }
  const data = cutoff ? allRows.filter(row => dateLabel(row.date) >= cutoff) : allRows;
  const width = 900, plotHeight = height, left = 65, right = 18, top = 20, bottom = 35;
  const equityLabel = portfolio ? '动态仓位组合' : sameAs === 'half' ? '固定半仓（当前策略）' : sameAs === 'benchmark' ? '买入持有（当前策略）' : '策略账户';
  const series = [{ key: 'equity', label: equityLabel, color: '#36d9bf' }, { key: 'benchmark', label: '买入持有（分红再投资）', color: '#8590b3' }, { key: 'half', label: '固定半仓对照', color: '#f3aa78' }].filter(series => series.key !== sameAs && (!portfolio || series.key !== 'benchmark'));
  const values = data.flatMap(row => series.filter(s => visible[s.key as keyof typeof visible]).map(s => Number(row[s.key])).filter(Number.isFinite));
  let min = Math.min(...values), max = Math.max(...values);
  if (!values.length) { min = 0; max = 1; }
  const span = max - min || Math.max(max * .1, 1); min -= span * .12; max += span * .12;
  const x = (index: number) => left + index / Math.max(1, data.length - 1) * (width - left - right);
  const y = (value: number) => top + (max - value) / (max - min) * (plotHeight - top - bottom);
  const path = (key: string) => data.filter((_, i) => i % Math.max(1, Math.floor(data.length / 550)) === 0 || i === data.length - 1).map(row => {
    const index = data.indexOf(row); return `${index === 0 ? 'M' : 'L'}${x(index).toFixed(2)},${y(Number(row[key])).toFixed(2)}`;
  }).join(' ');
  const hovered = hover !== null ? data[Math.min(hover, data.length - 1)] : null;
  const last = data[data.length - 1];
  return <div className={`nav-chart ${compact ? 'compact-chart' : ''}`} ref={container}>
    {!compact && <div className="chart-toolbar"><div><h3>{title}</h3><span className="chart-subtitle">以美元计价 · 扣除佣金与滑点</span></div><div className="segmented">{['1M','3M','1Y','ALL'].map(value => <button key={value} className={range === value ? 'selected' : ''} onClick={() => {setRange(value);setHover(null);}}>{value === 'ALL' ? '全部' : value}</button>)}</div></div>}
    <div className="chart-legend">{series.map(s => <button key={s.key} aria-pressed={visible[s.key as keyof typeof visible]} className={!visible[s.key as keyof typeof visible] ? 'muted' : ''} onClick={() => setVisible(v => ({ ...v, [s.key]: !v[s.key as keyof typeof v] }))}><span style={{ background: s.color }}/>{s.label}</button>)}{last && <span className="chart-date">{dateLabel(data[0].date)} — {dateLabel(last.date)}</span>}</div>
    {!data.length ? <Empty title="等待研究数据" description="选择实例或运行一次回测，查看真实权益曲线。" icon={TrendingUp}/> : <div className="chart-surface">
      <svg viewBox={`0 0 ${width} ${plotHeight}`} role="img" aria-label={title} onMouseLeave={() => setHover(null)} onMouseMove={event => {
        const box = event.currentTarget.getBoundingClientRect(); const point = (event.clientX - box.left) / box.width * width;
        setHover(Math.max(0, Math.min(data.length - 1, Math.round((point - left) / (width - left - right) * (data.length - 1)))));
      }}>
        <defs><linearGradient id="equity-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#2dd4bf" stopOpacity=".17"/><stop offset="100%" stopColor="#2dd4bf" stopOpacity="0"/></linearGradient></defs>
        {[0,1,2,3,4].map(i => { const value = min + (max-min) * i / 4; return <g key={i}><line x1={left} x2={width-right} y1={y(value)} y2={y(value)} stroke="#1a233b" strokeDasharray="3 5"/><text x={left-12} y={y(value)+4} textAnchor="end" fill="#64728e" fontSize="11">{money(value / 1000, 1)}k</text></g>; })}
        {[0,.25,.5,.75,1].map(fraction => {const i = Math.round((data.length-1) * fraction);return <text key={fraction} x={x(i)} y={plotHeight-10} textAnchor="middle" fill="#64728e" fontSize="11">{dateLabel(data[i].date)}</text>;})}
        {visible.equity && <path d={`${path('equity')} L${x(data.length-1)},${plotHeight-bottom} L${left},${plotHeight-bottom} Z`} fill="url(#equity-fill)"/>}
        {series.filter(s => visible[s.key as keyof typeof visible] && data.some(row => Number.isFinite(Number(row[s.key])))).map(s => <path key={s.key} d={path(s.key)} stroke={s.color} strokeWidth={s.key === 'equity' ? 2.3 : 1.5} strokeDasharray={s.key === 'equity' ? undefined : '5 5'} fill="none" strokeLinejoin="round"/>)}
        {hovered && hover !== null && <g><line x1={x(hover)} x2={x(hover)} y1={top} y2={plotHeight-bottom} stroke="#677995" strokeDasharray="3 4"/><circle cx={x(hover)} cy={y(Number(hovered.equity))} r="4" fill="#36d9bf" stroke="#082529" strokeWidth="3"/></g>}
      </svg>
      {hovered && hover !== null && <div className="chart-tooltip" style={{left: `${Math.max(16, Math.min(76, x(hover)/width*100))}%`}}><strong>{dateLabel(hovered.date)}</strong>{series.map(s => hovered[s.key] != null && <div key={s.key}><span><i style={{background:s.color}}/>{s.label}</span><b>${money(hovered[s.key])}</b></div>)}</div>}
    </div>}
  </div>;
}

function Convergence({ history = [] }: { history?: Row[] }) {
  if (!history.length) return <Empty title="开始一轮可复现实验" description="搜索历史将显示每次真实评估的最佳得分。" icon={FlaskConical}/>;
  const points = history.map((row, index) => ({ x: Number(row.evaluation ?? index + 1), y: Number(row.best_score ?? row.score ?? 0) }));
  const finite = points.filter(point => Number.isFinite(point.y) && point.y > -1e100);
  if (!finite.length) return <Empty title="正在评估初始参数" icon={Loader2}/>;
  const low = Math.min(...finite.map(p => p.y)), high = Math.max(...finite.map(p => p.y));
  const path = finite.map((p, i) => `${i ? 'L' : 'M'}${30 + (p.x-finite[0].x)/Math.max(1,finite[finite.length-1].x-finite[0].x)*650},${145-(p.y-low)/(high-low || 1)*105}`).join(' ');
  return <div className="convergence"><div className="chart-subtitle">训练目标分数 · 累计最优 <span className="mono">{finite[finite.length-1].y.toFixed(4)}</span></div><svg viewBox="0 0 710 180"><line x1="30" x2="680" y1="145" y2="145" stroke="#1a233b"/><line x1="30" x2="680" y1="90" y2="90" stroke="#1a233b" strokeDasharray="3 5"/><line x1="30" x2="680" y1="40" y2="40" stroke="#1a233b" strokeDasharray="3 5"/><path d={path} stroke="#36d9bf" strokeWidth="2.5" fill="none"/><text x="30" y="171" fill="#64728e" fontSize="11">首次评估</text><text x="680" y="171" fill="#64728e" textAnchor="end" fontSize="11">{finite[finite.length-1].x} 次评估</text></svg></div>;
}

export default function App() {
  const [page, setPage] = useState<Page>(() => (window.location.hash.slice(2) as Page) || 'overview');
  const [data, setData] = useState<Row | null>(null);
  const [loading, setLoading] = useState(true);
  const [connectionError, setConnectionError] = useState('');
  const [toast, setToast] = useState<{ text: string; error?: boolean } | null>(null);
  const [selectedId, setSelectedId] = useState('');
  const [replay, setReplay] = useState<Row | null>(null);
  const [replayLoading, setReplayLoading] = useState(false);
  const [instanceModal, setInstanceModal] = useState<Row | null>(null);
  const [backtestPreset, setBacktestPreset] = useState<{ genomeId: string; validation?: string[] }>({ genomeId: '' });
  const [mobileOpen, setMobileOpen] = useState(false);
  const notify = useCallback((text: string, error = false) => setToast({text,error}), []);
  const load = useCallback(async () => {try { const value = await api.bootstrap(); setData(value); setConnectionError(''); } catch (error) {setConnectionError((error as Error).message);} finally {setLoading(false);} }, []);
  useEffect(() => {load(); const id = window.setInterval(load, 4000); return () => window.clearInterval(id);}, [load]);
  useEffect(() => {
    const update = () => {
      const next = window.location.hash.slice(2) as Page;
      setPage(NAV.some(item => item.id === next) ? next : 'overview');
      setMobileOpen(false);
      window.scrollTo(0, 0);
    };
    update();
    window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);
  useEffect(() => {if (!toast) return;const id = window.setTimeout(() => setToast(null),4500);return () => window.clearTimeout(id);},[toast]);
  const instances: Row[] = data?.instances || [];
  const genomes: Row[] = data?.genomes || [];
  const tasks: Row[] = data?.tasks || [];
  const tuningRuns: Row[] = data?.tuning_runs || [];
  const backtests: Row[] = (data?.backtests || []).filter((row: Row) => !row.allocations);
  const portfolioBacktests: Row[] = (data?.backtests || []).filter((row: Row) => row.allocations);
  useEffect(() => {if (!instances.length) {setSelectedId('');return;}if (!instances.some(row => row.id === selectedId)) setSelectedId((instances.find(row => row.symbol === 'SPY') || instances[0]).id);},[instances,selectedId]);
  const currentInstance = instances.find(row => row.id === selectedId);
  const replayKey = JSON.stringify([selectedId,currentInstance?.symbol,currentInstance?.template,currentInstance?.initial_cash,currentInstance?.genome_id]);
  const needsReplay = page === 'overview' || page === 'executor';
  useEffect(() => {if (!selectedId || !needsReplay) {setReplay(null);return;}let alive = true;setReplay(null);setReplayLoading(true);api.replay(selectedId).then(value => {if (alive) setReplay(value);}).catch(error => {if (alive) {setReplay(null);notify(error.message,true);}}).finally(() => {if(alive)setReplayLoading(false);});return () => {alive=false;};},[replayKey,needsReplay,notify]);
  const navigate = (next: Page) => {window.location.hash = `/${next}`;};
  const applyGenome = async (genome: Row) => {if(currentInstance && currentInstance.template!=='sigmoid'){notify('请先选择 Sigmoid 动态均衡实例以应用参数',true);return;}try {await api.promote(genome.id,selectedId || undefined);await load();notify(selectedId ? '参数已晋升并应用到所选实例' : '参数已设为当前冠军');if(selectedId)api.replay(selectedId).then(setReplay).catch(()=>{});}catch(error){notify((error as Error).message,true);}};
  const toggleInstance = async (instance: Row) => {try {await api.updateInstance(instance.id,{status:instance.status === 'running' ? 'paused':'running'});await load();notify(instance.status === 'running' ? '研究实例已暂停':'研究实例已启用，可进入历史回放页播放');}catch(error){notify((error as Error).message,true);}};
  const openBacktest = (genomeId = '', validation?: string[]) => {
    setBacktestPreset({ genomeId, validation });
    navigate('backtest');
  };
  const shared = { data, instances, genomes, tasks, backtests, selectedId, setSelectedId, currentInstance, replay, replayKey, replayLoading, notify, load, navigate, toggleInstance, setInstanceModal, openBacktest, applyGenome };
  const currentPage = NAV.find(item => item.id === page) || NAV[0];
  return <div className="app"><div className="ambient ambient-coral"/><div className="ambient ambient-teal"/>
    {mobileOpen && <button className="mobile-shade" aria-label="关闭导航" onClick={()=>setMobileOpen(false)}/>}
    <aside className={`sidebar ${mobileOpen ? 'mobile-open':''}`}>
      <a className="brand" href="#/overview"><span className="brand-mark"><Activity size={24} strokeWidth={2.4}/></span><span>Quant<span className="brand-light">Flow</span><small>RESEARCH WORKSPACE</small></span></a>
      <div className="workspace-switch"><span className="workspace-icon"><FlaskConical size={17}/></span><div><strong>个人研究空间</strong><small>Local workspace</small></div><Pill>DEV</Pill></div>
      <span className="nav-label">工作空间</span>
      <nav>{NAV.map((item,index) => <a key={item.id} href={`#/${item.id}`} className={page===item.id?'active':''}><item.icon size={19}/><span>{item.name}</span>{index===3 && <span className="nav-count">{[...tasks, ...tuningRuns].filter(task=>activeTask(task)).length || 'LAB'}</span>}{page===item.id&&<span className="nav-active-dot"/>}</a>)}</nav>
      <div className="sidebar-bottom"><div className="environment-card"><ShieldCheck size={18}/><div><strong>本地研究环境</strong><small>历史数据 · 无真实交易</small></div><span className="status-dot"/></div><div className="profile"><span className="avatar">QF</span><div><strong>策略研究员</strong><small>个人工作台</small></div><span className="profile-dot"/></div></div>
    </aside>
    <div className="workspace"><header className="topbar"><div className="breadcrumb"><button className="mobile-menu icon-button" aria-label="打开导航" onClick={()=>setMobileOpen(true)}><Layers3 size={20}/></button><span>工作空间</span><ChevronRight size={13}/><strong>{currentPage.name}</strong></div><div className="topbar-status"><span className={`service-status ${connectionError?'offline':''}`}><i/>{connectionError?'研究服务连接中断':'本地研究服务'}</span><span className="topbar-divider"/><Clock3 size={14}/><span>历史数据截至 2025-12-31</span><button className="icon-button" aria-label="刷新数据" onClick={()=>{load();notify('已刷新研究数据');}}><RefreshCw size={16}/></button><span className="topbar-avatar">Q</span></div></header>
      <main><div className="page-heading"><div><h1>{page==='overview'?'自研 Agent 量化回测平台与遗传优化算法':currentPage.name}</h1><p>{currentPage.sub}</p></div><div className="heading-actions">{page==='instances'&&<button className="button primary" onClick={()=>setInstanceModal({})}><Plus size={17}/>新建实例</button>}</div></div>
        {connectionError&&<div className="notice error"><CircleDot size={17}/><span>{connectionError}。请确认本地研究服务正在运行。</span><button onClick={load}>重试</button></div>}
        {data?.synthetic_data && <div className="walkthrough-tip" role="status"><strong>合成行情演示</strong> · 当前行情与回测结果由程序生成，用于体验流程。</div>}
        {page==='overview'&&<ProjectIntro synthetic={data?.synthetic_data === true}/>}
        {loading&&!data?<div className="initial-loading"><Loader2 className="spin" size={30}/><h3>正在连接研究工作空间</h3><p>读取真实实验、参数与回测账本</p></div>:data?<>
          {page==='overview'&&<details className="walkthrough-details"><summary>查看所选策略的资金曲线和账户详情</summary><Overview {...shared}/></details>}
          {page==='templates'&&<Templates {...shared}/>}
          {page==='instances'&&<Instances {...shared}/>}
          {page==='optimization'&&<Optimization {...shared} tuningRuns={tuningRuns}/>}
          {page==='backtest'&&<Backtesting {...shared} key={backtestPreset.genomeId}
            initialGenome={backtestPreset.genomeId} validationPreset={backtestPreset.validation}/>}
          {page==='portfolio'&&<PortfolioBacktest
            datasets={data.overview.datasets} genomes={genomes} backtests={portfolioBacktests}
            notify={notify} refresh={load}
            renderChart={rows => <NavChart portfolio rows={rows.map(row => ({ ...row, equity: row.sigmoid }))}/>}
          />}
          {page==='executor'&&<Executor {...shared}/>}
          {page==='research-agent'&&<ResearchAgent/>}
        </>:<Empty title="研究服务尚未就绪" description="服务连接成功后，会显示已保存的真实研究记录。" action={<button className="button primary" onClick={load}>重新连接</button>}/>}
        <footer className="main-footer"><span><ShieldCheck size={12}/> 仅用于历史研究与模拟回放</span><span>QuantFlow Research · 数据与结果均可追溯</span></footer>
      </main>
    </div>
    {toast&&<div className={`toast ${toast.error?'toast-error':''}`}>{toast.error?<CircleDot size={18}/>:<CheckCircle2 size={18}/>}<span>{toast.text}</span><button onClick={()=>setToast(null)} aria-label="关闭提示"><X size={15}/></button></div>}
    {instanceModal&&<InstanceModal instance={instanceModal} genomes={genomes} onClose={()=>setInstanceModal(null)} onSave={async body=>{try{if(instanceModal.id)await api.updateInstance(instanceModal.id,body);else await api.createInstance(body);await load();setInstanceModal(null);notify(instanceModal.id?'实例配置已更新':'研究实例已创建');}catch(error){notify((error as Error).message,true);throw error;}}}/>}
  </div>;
}

type Shared = Row;

function InstanceSelect({ instances, value, onChange }: { instances: Row[]; value: string; onChange: (id:string)=>void }) {
  return <div className="instance-select"><span className="select-icon"><Boxes size={16}/></span><select aria-label="选择研究实例" value={value} onChange={event=>onChange(event.target.value)}>{!instances.length&&<option value="">暂无研究实例</option>}{instances.map(instance=><option value={instance.id} key={instance.id}>{instance.name} · {instance.symbol}</option>)}</select><ChevronDown size={13}/></div>;
}

function Overview({data,instances,selectedId,setSelectedId,replay,replayLoading,currentInstance,navigate,setInstanceModal,toggleInstance}:Shared) {
  const metrics = replay?.metrics || {};
  const chart:Row[] = replay?.chart || [];
  const daily:Row[] = replay?.daily || [];
  const portfolio = replay?.portfolio || daily[daily.length-1] || {};
  const finalEquity = portfolio.equity ?? metrics.final_equity;
  const overview = data?.overview || {};
  const events:Row[] = (data?.events || []).slice(0,5);
  return <>
    <div className="context-row"><div className="context-label"><span className="live-square"/>研究面板 <span className="context-slash">/</span><InstanceSelect instances={instances} value={selectedId} onChange={setSelectedId}/></div><span className="muted small">{replayLoading?<><Loader2 size={12} className="spin"/> 正在读取账本</>:<><Database size={12}/> SPY / GLD · 日频历史数据</>}</span></div>
    <div className="kpi-grid"><Kpi label="期末总资产" value={finalEquity!=null?money(finalEquity):'—'} unit="$" icon={Gauge} note="所选实例 · 历史测试期末"/><Kpi label="累计收益率" value={metrics.total_return!=null?pct(metrics.total_return,true):'—'} icon={TrendingUp} tone="highlight"><span className={metrics.total_return>=0?'positive':'negative'}><ArrowUpRight size={12}/> {metrics.cagr!=null?`年化 ${pct(metrics.cagr)}`:'等待回测'}</span><span>扣除交易成本</span></Kpi><Kpi label="最大回撤" value={metrics.max_drawdown!=null?pct(metrics.max_drawdown):'—'} icon={Waves} note="基于每日权益峰谷计算"/><Kpi label="研究实例" value={String(overview.instance_count??instances.length)} icon={Boxes}><span className="positive"><i className="tiny-dot"/>{overview.running_instances??instances.filter((r:Row)=>r.status==='running').length} 个实例已启用</span><span>{overview.completed_optimizations??0} 次参数实验</span></Kpi></div>
    <div className="overview-main"><Panel className="equity-panel" title="投资组合表现" eyebrow="PORTFOLIO PERFORMANCE" action={<Pill>历史测试区间</Pill>}><NavChart key={currentInstance?.template || 'strategy'} rows={chart} sameAs={currentInstance?.template === 'half' ? 'half' : currentInstance?.template === 'buy_hold_reinvest' ? 'benchmark' : undefined}/><div className="chart-summary"><div><span>初始资金</span><strong>${money(currentInstance?.initial_cash??currentInstance?.capital??metrics.initial_cash)}</strong></div><div><span>夏普比率 <small>rf = 0</small></span><strong>{metrics.sharpe_rf0!=null?Number(metrics.sharpe_rf0).toFixed(3):'—'}</strong></div><div><span>平均资产敞口</span><strong>{metrics.mean_exposure!=null?pct(metrics.mean_exposure):'—'}</strong></div><div><span>成交笔数</span><strong>{metrics.trade_count??'—'}<small> 笔</small></strong></div></div></Panel>
      <Panel title="研究概况" eyebrow="WORKSPACE STATUS" className="research-status"><div className="status-orbit"><div className="orbit outer"/><div className="orbit inner"/><div className="orbit-icon"><Activity size={34}/></div><span className="orbit-point"/></div><div className="research-status-title">研究环境已就绪<Pill tone="teal" dot>本地计算</Pill></div><p>从策略假设、参数搜索到历史回放，<br/>每一份结果都有完整账本。</p><div className="research-status-list"><div><span><Database size={14}/> 数据集</span><strong>{(overview.datasets||[]).length||2} 个 ETF</strong></div><div><span><FlaskConical size={14}/> 参数实验</span><strong>{overview.completed_optimizations??data.tasks?.filter((t:Row)=>t.status==='completed').length??0} 次完成</strong></div><div><span><BarChart3 size={14}/> 回测记录</span><strong>{overview.completed_backtests??data.backtests?.length??0} 份</strong></div></div><button className="button secondary full" onClick={()=>navigate('optimization')}>进入参数实验室<ArrowRight size={15}/></button></Panel></div>
    <div className="overview-bottom"><Panel title="我的策略实例" action={<button className="text-button" onClick={()=>navigate('instances')}>查看全部<ArrowRight size={14}/></button>}><div className="overview-instances">{instances.length?instances.slice(0,3).map((instance:Row)=><div key={instance.id} className={`overview-instance ${instance.id===selectedId?'selected-instance':''}`}><button className="instance-main" onClick={()=>setSelectedId(instance.id)}><span className={`strategy-avatar ${instance.symbol==='GLD'?'gold':''}`}><Activity size={20}/></span><div><strong>{instance.name}</strong><span>{instance.symbol} <i/> {templateDetails[instance.template]?.title || '策略实例'}</span></div></button><div className="instance-side"><Status value={instance.status} instance/><button className="icon-button bordered" aria-label={instance.status==='running'?'暂停研究实例':'启用研究实例'} onClick={()=>toggleInstance(instance)}>{instance.status==='running'?<Pause size={14}/>:<Play size={14}/>}</button></div></div>):<Empty title="创建你的第一个实例" action={<button className="button primary" onClick={()=>setInstanceModal({})}><Plus size={15}/>新建实例</button>}/>}</div></Panel>
      <Panel title="最近动态" action={<span className="small muted">实际操作记录</span>}><div className="activity-list">{events.length?events.map((event:Row,index:number)=><div className="activity-item" key={event.id||index}><span className={`activity-icon ${String(event.kind).includes('optimization')?'coral':''}`}>{String(event.kind).includes('optimization')?<FlaskConical size={14}/>:String(event.kind).includes('backtest')?<BarChart3 size={14}/>:<Check size={14}/>}</span><div><strong>{event.message||event.kind}</strong><span>{String(event.created_at||'').replace('T',' ').slice(0,19)}</span></div></div>):<Empty title="准备开始新的研究" description="创建实例或运行实验后，操作会记录在这里。" icon={Clock3}/>}</div></Panel></div>
    <div className="method-note"><ShieldCheck size={15}/><span>所有表现均来自历史数据回测，非实时账户收益。日频信号于收盘生成，下一交易日开盘模拟成交。</span><a href="/artifacts/dashboard.html" target="_blank" rel="noreferrer">查看研究报告<ArrowUpRight size={13}/></a></div>
  </>;
}

const templateDetails:Row = {
  sigmoid:{icon:Activity,color:'teal',category:'动态仓位',title:'Sigmoid 动态均衡',description:'将价格偏离与趋势信号映射为连续目标仓位，在收益与风险之间寻找平衡。',formula:'w = 1 / (1 + exp(a × 偏离 − b × 动量))',tags:['可进化参数','连续目标仓位','调仓带过滤'],items:['21 日指数均线偏离，按历史波动标准化','63 日动量信号，只使用当时可得数据','仓位差超过 band 时，下一交易日调仓']},
  momentum:{icon:TrendingUp,color:'coral',category:'趋势跟踪',title:'时序动量',description:'以标准化的中期动量判断持有或空仓，提供一个易于解释的趋势基线。',formula:'w = 1  if MOM₆₃ > 0, otherwise 0',tags:['固定规则','趋势基线','长仓 / 现金'],items:['63 日历史动量，正向信号持有资产','非正信号回到现金，禁止卖空和杠杆','与主策略共享费用和成交时序']},
  half:{icon:Layers3,color:'blue',category:'风险对照',title:'固定半仓均衡',description:'始终以一半资产敞口参与市场，用于衡量主动择时是否带来实际增量。',formula:'w = 50%  ·  rebalance band = 5%',tags:['固定规则','敞口对照','带宽再平衡'],items:['目标为 50% 标的资产与 50% 现金','偏离目标超过 5% 时进行再平衡','同样使用整股、现金约束与交易成本']},
  buy_hold_reinvest:{icon:TrendingUp,color:'blue',category:'被动基准',title:'买入持有再投资',description:'初始买入可负担的整股份额，分红代理到账后再投资，作为全仓被动基准。',formula:'Buy available whole shares; reinvest paid dividends',tags:['固定规则','全仓基准','分红再投资'],items:['初始信号下一交易日开盘买入','分红除息日记为应收，35 天后转为现金','到账现金用于后续整股再投资，计入相同成本']},
};
function Templates({data,setInstanceModal}:Shared) {
  const templates:Row[] = data.templates?.length?data.templates:Object.keys(templateDetails).map(id=>({id}));
  return <><div className="section-intro"><div><Pill tone="teal"><Layers3 size={12}/>策略库</Pill><h2>把想法变成可验证的策略</h2><p>清晰的信号、统一的执行规则。为每个假设建立独立实例。</p></div><span className="small muted">{templates.length} 个研究模板 · 日频 / 只做多</span></div><div className="template-grid">{templates.map((template:Row)=>{const detail=templateDetails[template.id]||templateDetails.sigmoid;const Icon=detail.icon;return <Panel key={template.id} className={`template-card template-${detail.color}`}><div className="template-top"><span className="template-icon"><Icon size={27}/></span><Pill>{detail.category}</Pill></div><span className="eyebrow">{template.id.toUpperCase()} STRATEGY</span><h2>{detail.title}</h2><p>{detail.description}</p><div className="template-tags">{detail.tags.map((tag:string)=><span key={tag}>{tag}</span>)}</div><div className="formula"><span>核心决策逻辑</span><code>{detail.formula}</code></div><ul>{detail.items.map((item:string)=><li key={item}><Check size={14}/>{item}</li>)}</ul><div className="template-supported"><span>支持标的</span><strong>SPY <i/> GLD</strong><span>日线</span></div><button className="button secondary full" onClick={()=>setInstanceModal({template:template.id})}><Plus size={16}/>使用此模板创建实例<ArrowRight size={15}/></button></Panel>;})}</div><div className="notice"><ShieldCheck size={16}/><span>策略模板定义的是研究规则，收益取决于市场与参数。历史回测不构成未来表现承诺。</span></div></>;
}

function Instances({instances,genomes,selectedId,setSelectedId,setInstanceModal,toggleInstance,navigate}:Shared) {
  const [filter,setFilter]=useState('all');
  const [query,setQuery]=useState('');
  const filtered:Row[]=instances.filter((instance:Row)=>(filter==='all'||instance.status===filter)&&`${instance.name} ${instance.symbol}`.toLowerCase().includes(query.toLowerCase()));
  return <><div className="list-toolbar"><div className="tabs">{[['all','全部实例'],['running','已启用'],['paused','已暂停']].map(([value,label])=><button key={value} className={filter===value?'active':''} onClick={()=>setFilter(value)}>{label}{value==='all'&&<span>{instances.length}</span>}</button>)}</div><div className="search-input"><Search size={15}/><input placeholder="搜索名称或标的…" value={query} onChange={e=>setQuery(e.target.value)}/></div></div><div className="instance-grid">{filtered.map(instance=>{const genome=genomes.find((g:Row)=>g.id===instance.genome_id);return <Panel key={instance.id} className={`instance-card ${instance.id===selectedId?'chosen':''}`}><div className="instance-card-top"><span className={`strategy-avatar ${instance.symbol==='GLD'?'gold':''}`}><Activity size={24}/></span><Status value={instance.status} instance/></div><h2>{instance.name}</h2><p>{instance.symbol} <span className="muted">/</span> {templateDetails[instance.template]?.title||'Sigmoid 动态均衡'}</p><div className="instance-capital"><span>初始研究资金</span><strong><small>$</small>{money(instance.initial_cash??instance.capital)}</strong></div><div className="instance-detail"><span>参数版本</span><strong>{instance.template==='sigmoid'?(genome?.name||instance.genome_id||'默认参数'):'模板固定规则'}</strong></div><div className="instance-detail"><span>创建日期</span><span className="mono">{dateLabel(instance.created_at)}</span></div><div className="instance-card-actions"><button className="button secondary" onClick={()=>setInstanceModal(instance)}><Settings2 size={14}/>配置</button><button className={`button ${instance.status==='running'?'secondary':'primary'}`} onClick={()=>toggleInstance(instance)}>{instance.status==='running'?<Pause size={14}/>:<Play size={14}/ >}{instance.status==='running'?'暂停实例':'启用实例'}</button></div><button className="instance-open" onClick={()=>{setSelectedId(instance.id);navigate('executor');}}>查看历史回放<ArrowRight size={14}/></button></Panel>})}<button className="create-instance-card" onClick={()=>setInstanceModal({})}><span><Plus size={27}/></span><strong>新建研究实例</strong><p>选择策略、标的和参数，<br/>开启一组独立实验。</p></button></div>{!filtered.length&&instances.length>0&&<Empty title="没有符合条件的实例" description="尝试其他名称或切换筛选条件。" icon={Search}/>}<div className="notice"><Database size={15}/><span>实例配置保存在本地研究服务中。启用状态保存在本地；逐日播放进度由历史回放页控制，不会向券商提交订单。</span></div></>;
}

function InstanceModal({instance,genomes,onClose,onSave}:{instance:Row;genomes:Row[];onClose:()=>void;onSave:(body:Row)=>Promise<void>}) {
  const [form,setForm]=useState({name:instance.name||'',symbol:instance.symbol||'SPY',template:instance.template||'sigmoid',initial_cash:instance.initial_cash??instance.capital??100000,genome_id:instance.genome_id||genomes.find(g=>g.role==='champion')?.id||''});
  const [saving,setSaving]=useState(false);
  const submit=async(event:FormEvent)=>{event.preventDefault();setSaving(true);try{await onSave({...form,initial_cash:Number(form.initial_cash),genome_id:form.genome_id||undefined});}catch{/* Toast is displayed by parent. */}finally{setSaving(false);}};
  return <div className="modal-backdrop" onClick={onClose}><div className="modal" role="dialog" aria-modal="true" aria-labelledby="instance-title" onClick={e=>e.stopPropagation()}><div className="modal-head"><div><span className="eyebrow">RESEARCH INSTANCE</span><h2 id="instance-title">{instance.id?'配置研究实例':'创建研究实例'}</h2></div><button className="icon-button" onClick={onClose} aria-label="关闭弹窗"><X size={20}/></button></div><p className="modal-description">每个实例使用独立资金和参数进行历史研究。</p><form onSubmit={submit}><Field label="实例名称"><input autoFocus required maxLength={80} placeholder="例如：SPY 动态均衡研究" value={form.name} onChange={e=>setForm({...form,name:e.target.value})}/></Field><div className="form-grid"><Field label="策略模板"><select value={form.template} onChange={e=>setForm({...form,template:e.target.value})}><option value="sigmoid">Sigmoid 动态均衡</option><option value="momentum">时序动量</option><option value="half">固定半仓</option><option value="buy_hold_reinvest">买入持有再投资</option></select></Field><Field label="研究标的"><select value={form.symbol} onChange={e=>setForm({...form,symbol:e.target.value})}><option value="SPY">SPY · 标普 500 ETF</option><option value="GLD">GLD · 黄金 ETF</option></select></Field></div><Field label="初始资金（USD）" hint="单一初始本金，无后续注资；范围 $1,000 — $10,000,000。"><input type="number" required min={1000} max={10000000} step={100} value={form.initial_cash} onChange={e=>setForm({...form,initial_cash:Number(e.target.value)})}/></Field>{form.template==='sigmoid'&&<Field label="参数版本"><select value={form.genome_id} onChange={e=>setForm({...form,genome_id:e.target.value})}>{!genomes.length&&<option value="">使用默认参数</option>}{genomes.map(g=><option key={g.id} value={g.id}>{g.name||g.id} {g.role==='champion'?'· 当前冠军':''}</option>)}</select></Field>}<div className="modal-note"><ShieldCheck size={15}/>本地历史研究实例 · 不连接真实券商</div><div className="modal-actions"><button type="button" className="button secondary" onClick={onClose}>取消</button><button className="button primary" disabled={saving}>{saving?<Loader2 className="spin" size={16}/>:<Check size={16}/>}保存实例</button></div></form></div></div>;
}

function Optimization({data,tasks,tuningRuns,genomes,instances,selectedId,setSelectedId,notify,load,openBacktest,applyGenome}:Shared) {
  const [tab,setTab]=useState('joint');
  const [form,setForm]=useState({method:'ga',population:16,generations:8,seed:42,spawn_mode:'current',mean_reversion:1,momentum:1,band:.05});
  const [submitting,setSubmitting]=useState(false);
  const [taskId,setTaskId]=useState('');
  const [task,setTask]=useState<Row|null>(null);
  useEffect(()=>{if(!taskId&&tasks.length)setTaskId(tasks[0].id);},[tasks,taskId]);
  useEffect(()=>{if(!taskId||tab!=='optimize')return;let alive=true;const poll=()=>api.task(taskId).then(row=>{if(alive)setTask(row);}).catch(error=>{if(alive)notify(error.message,true);});poll();const interval=window.setInterval(poll,1200);return()=>{alive=false;window.clearInterval(interval);};},[taskId,notify,tab]);
  const submit=async(event:FormEvent)=>{event.preventDefault();setSubmitting(true);try{const body:Row={method:form.method,population:Number(form.population),generations:Number(form.generations),seed:Number(form.seed),spawn_mode:form.spawn_mode};if(form.spawn_mode==='manual')body.params={mean_reversion:Number(form.mean_reversion),momentum:Number(form.momentum),band:Number(form.band)};const row=await api.optimize(body);setTaskId(row.id);setTask(row);await load();notify('参数实验已提交，正在进行真实历史评估');}catch(error){notify((error as Error).message,true);}finally{setSubmitting(false);}};
  const completed=Number(task?.progress?.completed??task?.completed??0), total=Number(task?.progress?.total??task?.total??task?.budget??0);
  const progress=total?Math.min(100,completed/total*100):0;
  const requestBudget=Math.min(512,form.population*form.generations);
  return <>
    <div className="list-toolbar">
      <div className="tabs">
        <button className={tab==='joint'?'active':''} onClick={()=>setTab('joint')}><SlidersHorizontal size={15}/>三参数联合调参</button>
        <button className={tab==='optimize'?'active':''} onClick={()=>setTab('optimize')}><FlaskConical size={15}/>单轮搜索</button>
        <button className={tab==='genomes'?'active':''} onClick={()=>setTab('genomes')}><Layers3 size={15}/>参数库<span>{genomes.length}</span></button>
      </div>
      <Pill>SPY · 2010 — 2018 训练</Pill>
    </div>
    {tab==='joint' && <JointTuning runs={tuningRuns} refresh={load}
      openBacktest={genomeId=>openBacktest(genomeId, data.protocol.validation)}/>}
    {tab==='optimize'?<><p className="walkthrough-tip">每轮只运行一种搜索算法和一个随机种子，可单独查看训练分数变化。</p><div className="optimization-layout"><Panel title="新建参数实验" eyebrow="EXPERIMENT CONFIGURATION" className="optimization-form"><form onSubmit={submit}><Field label="搜索算法"><div className="algorithm-choices"><button type="button" onClick={()=>setForm({...form,method:'ga'})} className={form.method==='ga'?'selected':''}><Sparkles size={17}/>遗传算法<small>GA</small></button><button type="button" onClick={()=>setForm({...form,method:'random'})} className={form.method==='random'?'selected':''}><SlidersHorizontal size={17}/>随机搜索<small>RANDOM</small></button></div></Field><div className="form-grid"><Field label="候选预算批量"><input required type="number" min={16} max={128} step={1} value={form.population} onChange={e=>setForm({...form,population:Number(e.target.value)})}/></Field><Field label="预算轮数"><input required type="number" min={1} max={32} step={1} value={form.generations} onChange={e=>setForm({...form,generations:Number(e.target.value)})}/></Field></div><div className="budget-summary"><span>唯一评估预算</span><strong>{requestBudget}<small> 次</small></strong></div><p className="field-note">预算 = 批量 × 轮数，上限 512。批量与轮数只控制候选评估预算；GA 内部种群固定为 32。</p><Field label="随机种子" hint="相同数据、参数与种子可重复同一轮实验。"><input required type="number" min={0} max={2147483647} step={1} value={form.seed} onChange={e=>setForm({...form,seed:Number(e.target.value)})}/></Field><Field label="参数起点" hint="继承／手动模式：以锚点为中心，在全空间宽度 ±25% 的局部范围搜索；随机探索使用全范围。锚点不保证出现在首代。"><select value={form.spawn_mode} onChange={e=>setForm({...form,spawn_mode:e.target.value})}><option value="current">继承当前冠军</option><option value="random">完全随机探索</option><option value="manual">手动指定参数</option></select></Field>{form.spawn_mode==='manual'&&<div className="manual-params"><Field label="均值回归权重 a"><input type="number" min={0} max={2} step={.01} value={form.mean_reversion} onChange={e=>setForm({...form,mean_reversion:Number(e.target.value)})}/></Field><Field label="动量权重 b"><input type="number" min={0} max={2} step={.01} value={form.momentum} onChange={e=>setForm({...form,momentum:Number(e.target.value)})}/></Field><Field label="调仓带 band"><input type="number" min={.02} max={.2} step={.01} value={form.band} onChange={e=>setForm({...form,band:Number(e.target.value)})}/></Field></div>}<button className="button primary full" disabled={submitting||!!activeTask(task)}>{submitting?<Loader2 size={16} className="spin"/>:<Play size={16}/>}启动参数优化</button><div className="form-bottom-note"><ShieldCheck size={12}/>仅训练区间用于寻优，不读取测试区间</div></form></Panel>
      <div className="optimization-results"><Panel title="实验运行状态" action={task?<Status value={task.status}/>:<Pill>等待任务</Pill>}>{task?<><div className="task-meta"><div><span>实验编号</span><strong className="mono">{String(task.id).slice(0,24)}</strong></div><div><span>搜索方法</span><strong>{task.method==='ga'?'遗传算法':'随机搜索'} <small>SEED {task.seed}</small></strong></div></div><div className="task-score"><div><span>当前最优训练分数</span><strong>{task.best_score!=null&&Number(task.best_score)>-1e100?Number(task.best_score).toFixed(4):'—'}</strong></div><div className="task-evaluation"><span className="mono">{completed}<small> / {total||'—'}</small></span><span>唯一参数评估</span></div></div><div className="progress-track"><div style={{width:`${progress}%`}}/></div><div className="progress-caption"><span>{activeTask(task)?'正在评估历史数据…':statusText[task.status]||task.status}</span><strong>{progress.toFixed(0)}%</strong></div><Convergence history={task.history||[]}/>{task.error&&<div className="notice error">{task.error}</div>}<div className="task-footer"><span><Clock3 size={13}/>{task.created_at?String(task.created_at).replace('T',' ').slice(0,19):'任务已提交'}</span>{activeTask(task)&&<button className="text-button danger" onClick={async()=>{try{await api.cancelTask(task.id);load();notify('已请求停止实验');}catch(error){notify((error as Error).message,true);}}}><Square size={12}/>停止实验</button>}{task.status==='completed'&&<button className="text-button" onClick={()=>{setTab('genomes');load();}}>查看候选参数<ArrowRight size={14}/></button>}</div></>:<Empty icon={FlaskConical} title="实验室已准备就绪" description="配置搜索算法和评估预算，启动第一轮参数实验。"/>}</Panel><div className="objective-note"><span className="objective-icon"><Gauge size={19}/></span><div><strong>在训练数据中寻找稳定性</strong><p>目标为三个训练分段的净夏普均值 − 0.25 × 分段标准差。分数不代表未来收益。</p></div></div></div></div>
      <Panel title="实验记录" action={<span className="small muted">{tasks.length} 次实验</span>}><div className="table-wrap"><table><thead><tr><th>实验</th><th>算法 / 种子</th><th>状态</th><th>评估进度</th><th>最优分数</th><th>创建时间</th><th/></tr></thead><tbody>{tasks.map((row:Row)=><tr key={row.id}><td><span className="table-id">{String(row.id).slice(0,19)}</span></td><td>{row.method==='ga'?'遗传算法':'随机搜索'}<span className="cell-sub">SEED {row.seed}</span></td><td><Status value={row.status}/></td><td className="mono">{row.completed??row.progress?.completed??0} / {row.total??row.progress?.total??row.budget??'—'}</td><td className="mono positive">{row.best_score!=null&&row.best_score>-1e100?Number(row.best_score).toFixed(4):'—'}</td><td className="muted mono small">{dateLabel(row.created_at)}</td><td><button className="text-button" onClick={()=>setTaskId(row.id)}>查看<ArrowRight size={12}/></button></td></tr>)}</tbody></table>{!tasks.length&&<Empty title="暂无实验记录" description="启动实验后，进度和结果会保存在本地。"/>}</div></Panel></>:tab==='genomes'?<><div className="genome-context"><p>候选参数可设为当前冠军，并应用至所选研究实例。</p><InstanceSelect instances={instances} value={selectedId} onChange={setSelectedId}/></div><div className="genome-grid">{genomes.map((genome:Row)=><Panel key={genome.id} className={`genome-card ${genome.role==='champion'?'champion':''}`}><div className="genome-top"><span className="genome-icon">{genome.role==='champion'?<Sparkles size={22}/>:<Layers3 size={22}/>}</span><Pill tone={genome.role==='champion'?'teal':''}>{genome.role==='champion'?'当前冠军':genome.role==='retired'?'历史参数':'候选参数'}</Pill></div><h3>{genome.name||genome.id}</h3><p className="small muted">{genome.method==='ga'?'遗传算法':genome.method==='random'?'随机搜索':'冻结研究参数'} · SEED {genome.seed??'—'}</p><div className="genome-metrics"><div><span>训练分数</span><strong>{genome.training_score!=null?Number(genome.training_score).toFixed(4):'—'}</strong></div><div><span>验证夏普</span><strong>{genome.validation?.sharpe_rf0!=null?Number(genome.validation.sharpe_rf0).toFixed(3):'—'}</strong></div></div><div className="param-list">{[['mean_reversion','均值回归权重'],['momentum','动量权重'],['band','调仓带']].map(([key,label])=><div key={key}><span>{label}</span><strong className="mono">{genome.params?.[key]!=null?Number(genome.params[key]).toFixed(4):'—'}</strong></div>)}</div><div className="genome-actions"><button className="button secondary" onClick={()=>openBacktest(genome.id)}><BarChart3 size={14}/>回测</button><button className={`button ${genome.role==='champion'?'secondary':'primary'}`} onClick={()=>applyGenome(genome)}><Check size={14}/>{genome.role==='champion'?'应用到实例':'晋升并应用'}</button></div></Panel>)}</div>{!genomes.length&&<Empty title="参数库尚无记录" description="完成参数实验后，候选会保存到这里。"/>}<div className="notice"><ShieldCheck size={15}/><span>人工应用参数不会修改已经冻结的原始研究结果。后续新回测会保存为独立记录。</span></div></>:null}
  </>;
}

function downloadTrades(trades:Row[],id:string) {
  if(!trades.length)return;
  const keys=Object.keys(trades[0]);
  const cell=(value:any)=>`"${String(value??'').replaceAll('"','""')}"`;
  const csv='\uFEFF'+[keys.map(cell).join(','),...trades.map(row=>keys.map(key=>cell(row[key])).join(','))].join('\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const anchor=document.createElement('a');anchor.href=url;anchor.download=`${id}_trades.csv`;anchor.click();URL.revokeObjectURL(url);
}
function TradeTable({trades=[]}:{trades?:Row[]}) {
  const [direction,setDirection]=useState('all');const [page,setPage]=useState(0);
  const filtered=trades.filter(row=>direction==='all'||(direction==='buy'?Number(row.quantity)>0:Number(row.quantity)<0));
  const pages=Math.max(1,Math.ceil(filtered.length/8));
  useEffect(()=>setPage(0),[direction,trades]);
  return <><div className="trade-filter"><div className="segmented">{[['all','全部成交'],['buy','买入'],['sell','卖出']].map(([value,label])=><button key={value} className={direction===value?'selected':''} onClick={()=>setDirection(value)}>{label}</button>)}</div><span className="small muted">共 {filtered.length} 笔</span></div><div className="table-wrap"><table><thead><tr><th>成交日期</th><th>信号日期</th><th>方向</th><th>份额</th><th>成交价格</th><th>成交金额</th><th>佣金</th><th>目标仓位</th></tr></thead><tbody>{filtered.slice(page*8,page*8+8).map((row,index)=><tr key={`${row.date}-${index}`}><td className="mono">{dateLabel(row.date)}</td><td className="mono muted">{dateLabel(row.signal_date)}</td><td><Pill tone={Number(row.quantity)>0?'teal':'coral'}>{Number(row.quantity)>0?'买入':'卖出'}</Pill></td><td className="mono">{money(Math.abs(Number(row.quantity)),0)}</td><td className="mono">${money(row.price)}</td><td className="mono">${money(Math.abs(Number(row.notional)))}</td><td className="mono muted">${money(row.commission)}</td><td className="mono">{pct(row.target_weight)}</td></tr>)}</tbody></table>{!filtered.length&&<Empty title="当前区间没有成交" description="策略保持现金或现有仓位；账本仍记录每日资产状态。" icon={ArrowDownLeft}/>}</div><div className="pagination"><span>每页 8 笔 · {filtered.length?`${page*8+1}–${Math.min((page+1)*8,filtered.length)}`:'0'} / {filtered.length}</span><div><button className="icon-button bordered" aria-label="上一页" disabled={page===0} onClick={()=>setPage(page-1)}><ChevronLeft size={16}/></button><span>{page+1} / {pages}</span><button className="icon-button bordered" aria-label="下一页" disabled={page>=pages-1} onClick={()=>setPage(page+1)}><ChevronRight size={16}/></button></div></div></>;
}

function Backtesting({data,instances,genomes,selectedId,currentInstance,notify,load,backtests,initialGenome,validationPreset}:Shared & {initialGenome:string; validationPreset?:string[]}) {
  const [form,setForm]=useState({instance_id:initialGenome?'':selectedId||'',genome_id:initialGenome||currentInstance?.genome_id||genomes.find((g:Row)=>g.role==='champion')?.id||'',symbol:validationPreset?'SPY':currentInstance?.symbol||'SPY',start:validationPreset?.[0]||'2022-01-01',end:validationPreset?.[1]||'2025-12-31',initial_cash:validationPreset?data.protocol.initial_cash:currentInstance?.initial_cash??currentInstance?.capital??100000,commission_bps:validationPreset?data.protocol.costs.commission_bps:1,slippage_bps:validationPreset?data.protocol.costs.slippage_bps:5});
  const [result,setResult]=useState<Row|null>(null);const [busy,setBusy]=useState(false);const [resultId,setResultId]=useState('');const [historyOpen,setHistoryOpen]=useState(false);
  useEffect(()=>{if(initialGenome)setForm(form=>({...form,instance_id:'',genome_id:initialGenome}));},[initialGenome]);
  useEffect(()=>{if(validationPreset||resultId||!backtests.length)return;setResultId(backtests[0].id);},[backtests,resultId,validationPreset]);
  useEffect(()=>{if(!resultId)return;let alive=true;let interval:number|undefined;const poll=()=>api.backtestDetail(resultId).then(row=>{if(!alive)return;setResult(row);if(!activeTask(row)&&interval)window.clearInterval(interval);}).catch(error=>{if(alive)notify(error.message,true);});poll();interval=window.setInterval(poll,1200);return()=>{alive=false;if(interval)window.clearInterval(interval);};},[resultId,notify]);
  const submit=async(event:FormEvent)=>{event.preventDefault();if(form.start>form.end){notify('开始日期应早于结束日期',true);return;}setBusy(true);try{const row=await api.backtest({...form,strategy,initial_cash:Number(form.initial_cash),commission_bps:Number(form.commission_bps),slippage_bps:Number(form.slippage_bps),genome_id:strategy==='sigmoid'?form.genome_id||undefined:undefined});setResult(row);setResultId(row.id);await load();notify('历史回测已提交，结果将保存到本地');}catch(error){notify((error as Error).message,true);}finally{setBusy(false);}};
  const strategy = form.instance_id ? (instances.find((row:Row)=>row.id===form.instance_id)?.template || 'sigmoid') : 'sigmoid';
  const trades:Row[]=result?.trades||[];
  const chart:Row[]=result?.chart||[];
  const metrics = result?.metrics || {};
  const sameAs = result?.strategy === 'half' ? 'half' : result?.strategy === 'buy_hold_reinvest' ? 'benchmark' : undefined;
  const performanceStrategies = [
    { id: 'strategy', label: templateDetails[result?.strategy]?.title || '所选策略', metrics, primary: true },
    ...(sameAs !== 'half' ? [{ id: 'half', label: '固定半仓', metrics: result?.baseline_metrics?.half }] : []),
    ...(sameAs !== 'benchmark' ? [{ id: 'benchmark', label: '买入持有（分红再投）', metrics: result?.baseline_metrics?.buy_hold_reinvest }] : []),
  ];
  const displayVersion = genomes.find((genome: Row) => genome.id === result?.genome_id)?.name || result?.genome_id;
  const matchingSetup = result && result.symbol === form.symbol && result.start === form.start && result.end === form.end
    && result.strategy === strategy && (strategy !== 'sigmoid' || result.genome_id === form.genome_id)
    && result.initial_cash === form.initial_cash && result.costs?.commission_bps === form.commission_bps
    && result.costs?.slippage_bps === form.slippage_bps;
  return <>
    {validationPreset && <p className="walkthrough-tip">已带入本轮参数和 SPY 验证期设置。点击「运行回测」可查看对应资金曲线和成交记录。</p>}
    <p className="walkthrough-tip">这里每次回测一个标的，可以在下方换标的。要在同一个账户内同时持有多个标的，请打开 <a className="walkthrough-report" href="#/portfolio">多标的回测 <ArrowRight size={13}/></a>。</p>
    <Panel title="回测配置" eyebrow="BACKTEST CONFIGURATION" action={<button className="text-button" onClick={()=>setHistoryOpen(!historyOpen)}><Clock3 size={14}/>历史记录<span className="count-badge">{backtests.length}</span></button>}><div className="field-note">当前策略：{templateDetails[strategy]?.title || strategy} · 选择实例会使用该实例的策略规则。</div><form onSubmit={submit} className="backtest-form"><Field label="研究实例"><select value={form.instance_id} onChange={event=>{const instance=instances.find((r:Row)=>r.id===event.target.value);setForm({...form,instance_id:event.target.value,symbol:instance?.symbol||form.symbol,initial_cash:instance?.initial_cash??instance?.capital??form.initial_cash,genome_id:instance?.genome_id||form.genome_id});}}><option value="">独立回测</option>{instances.map((instance:Row)=><option key={instance.id} value={instance.id}>{instance.name}</option>)}</select></Field><Field label="参数版本" hint={strategy!=='sigmoid'?'固定规则策略不使用参数版本':undefined}><select disabled={strategy!=='sigmoid'} value={form.genome_id} onChange={e=>setForm({...form,genome_id:e.target.value})}><option value="">默认 Sigmoid 参数</option>{genomes.map((g:Row)=><option key={g.id} value={g.id}>{g.name||g.id}{g.role==='champion'?' · 冠军':''}</option>)}</select></Field><Field label="标的"><select value={form.symbol} onChange={e=>setForm({...form,symbol:e.target.value})}>{(data.overview.datasets as Row[]).map(dataset => <option key={dataset.symbol} value={dataset.symbol}>{dataset.symbol}{dataset.name ? ` · ${dataset.name}` : ''}</option>)}</select></Field><Field label="开始日期"><input required type="date" min="2005-01-01" max="2025-12-31" value={form.start} onChange={e=>setForm({...form,start:e.target.value})}/></Field><Field label="结束日期"><input required type="date" min={form.start} max="2025-12-31" value={form.end} onChange={e=>setForm({...form,end:e.target.value})}/></Field><Field label="初始资金（USD）"><input required type="number" min={1000} max={10000000} value={form.initial_cash} onChange={e=>setForm({...form,initial_cash:Number(e.target.value)})}/></Field><Field label="佣金（基点）" hint="1 基点 = 成交金额的 0.01%"><input required type="number" min={0} max={100} step={.1} value={form.commission_bps} onChange={e=>setForm({...form,commission_bps:Number(e.target.value)})}/></Field><Field label="滑点（基点）" hint="模拟成交价比开盘价更不利的幅度"><input required type="number" min={0} max={200} step={.1} value={form.slippage_bps} onChange={e=>setForm({...form,slippage_bps:Number(e.target.value)})}/></Field><button className="button primary" disabled={busy||!!activeTask(result)}>{busy||activeTask(result)?<Loader2 className="spin" size={16}/>:<Play size={16}/>}运行回测</button></form>{historyOpen&&<div className="backtest-history">{backtests.length?backtests.map((row:Row)=><button key={row.id} className={resultId===row.id?'selected':''} onClick={()=>{setResultId(row.id);setHistoryOpen(false);}}><span><strong>{row.symbol} · {row.name||String(row.id).slice(0,18)}</strong><small>{dateLabel(row.start)} — {dateLabel(row.end)}</small></span><Status value={row.status}/><ChevronRight size={14}/></button>):<p className="muted small">暂无保存的回测记录。</p>}</div>}</Panel>
    {result?<><div className="result-context"><div><span className="live-square"/><strong>{result.symbol||form.symbol} · {templateDetails[result.strategy]?.title || result.strategy || 'Sigmoid 动态均衡'}</strong><Pill>HISTORICAL</Pill><Status value={result.status}/></div><span className="mono small muted">{String(result.id).slice(0,28)}</span></div>{result.error&&<div className="notice error">{result.error}</div>}<div className="walkthrough-tip">
      <strong>{matchingSetup ? '结果配置' : '下方显示已保存的结果，按「运行回测」可应用上面的配置'}</strong>
      <div>保存的区间 {dateLabel(result.start)} 至 {dateLabel(result.end)} · 初始资金 ${money(result.initial_cash)} · 佣金 {result.costs?.commission_bps ?? '—'} bp · 滑点 {result.costs?.slippage_bps ?? '—'} bp</div>
      {result.strategy === 'sigmoid' && displayVersion && <div>参数版本 {displayVersion}{result.params?.band != null && ` · 调仓阈值 ${(result.params.band * 100).toFixed(0)} 个百分点`}</div>}
      {sameAs && <div>当前策略本身就是一条基准，图中合并了对应曲线。</div>}
    </div>
    {result.status === 'completed' && <PerformanceComparison title="策略与基准指标对比" strategies={performanceStrategies}/>}
    <Panel title="资金曲线" action={<Pill>{dateLabel(metrics.start||result.start)} — {dateLabel(metrics.end||result.end)}</Pill>}>
      <NavChart key={sameAs || 'strategy'} rows={chart} sameAs={sameAs}/>
      <a className="walkthrough-report" href="#/research-agent">核对原始研究的账本和曲线 <ArrowRight size={13}/></a>
    </Panel><Panel title="逐笔成交记录" action={<button className="button secondary small-button" disabled={!trades.length} onClick={()=>downloadTrades(trades,result.id)}><Download size={14}/>导出 CSV</button>}><TradeTable trades={trades}/></Panel><div className="method-note"><ShieldCheck size={15}/><span>真实历史账本 · 收盘生成信号、下一交易日开盘模拟成交 · 分红以 35 天现金到账代理处理。</span></div></>:<Empty title="让策略接受数据的检验" description="选择参数和时间区间，运行一次完整的历史回测。" icon={BarChart3}/>}
  </>;
}

function Executor({instances,selectedId,setSelectedId,currentInstance,replay,replayKey,replayLoading,notify,load}:Shared) {
  const [cursor,setCursor]=useState(0);const [speed,setSpeed]=useState(1);const [playing,setPlaying]=useState(false);const [playBusy,setPlayBusy]=useState(false);
  const daily:Row[]=replay?.daily||[];const chart:Row[]=replay?.chart||[];const fills:Row[]=replay?.fills||replay?.trades||[];
  useEffect(()=>{setCursor(0);setPlaying(false);},[replayKey]);
  useEffect(()=>{if(currentInstance?.status!=='running')setPlaying(false);},[currentInstance?.status]);
  useEffect(()=>{if(!playing||!daily.length)return;const interval=window.setInterval(()=>setCursor(index=>{if(index>=daily.length-1){setPlaying(false);return index;}return Math.min(index+1,daily.length-1);}),Math.max(40,1000/speed));return()=>window.clearInterval(interval);},[playing,speed,daily.length]);
  const row=daily[Math.min(cursor,daily.length-1)]||{};const date=dateLabel(row.date||chart[cursor]?.date);
  const pastFills=fills.filter(fill=>dateLabel(fill.date)<=date).slice(-8).reverse();
  const nextFill = fills.find(fill => dateLabel(fill.date) > date);
  const jumpFill = nextFill || fills[0];
  const jumpIndex = jumpFill ? daily.findIndex(day => dateLabel(day.date) === dateLabel(jumpFill.date)) : -1;
  const jumpToFill = () => {
    if (jumpIndex < 0) return;
    setPlaying(false);
    setCursor(jumpIndex);
  };
  const toggle=async()=>{if(playBusy)return;setPlayBusy(true);const next=!playing;try{await api.updateInstance(selectedId,{status:next?'running':'paused'});if(next&&cursor>=daily.length-1)setCursor(0);setPlaying(next);await load();}catch(error){notify((error as Error).message,true);}finally{setPlayBusy(false);}};
  return <><div className="context-row"><InstanceSelect instances={instances} value={selectedId} onChange={setSelectedId}/><Pill tone="coral"><Cpu size={12}/>确定性历史回放 · 非券商执行端</Pill></div><p className="walkthrough-tip">点「下一笔成交」定位一笔交易。右侧会显示策略在哪一天收盘后发出信号、在哪一天开盘时模拟成交，以及买卖数量和费用。</p>{replayLoading?<div className="initial-loading"><Loader2 className="spin" size={25}/><p>读取历史执行账本…</p></div>:daily.length?<><div className="kpi-grid"><Kpi label="回放日期" value={date} icon={Clock3} note={`${cursor+1} / ${daily.length} 个交易日`}/><Kpi label="当前权益" value={money(row.equity)} unit="$" icon={Gauge} note={`初始资金 $${money(currentInstance?.initial_cash??currentInstance?.capital??100000)}`}/><Kpi label="持有份额" value={money(row.shares,0)} icon={Layers3} note={`${currentInstance?.symbol||'—'} · 敞口 ${pct(row.exposure)}`}/><Kpi label="可用现金" value={money(row.cash)} unit="$" icon={Database} note={`分红应收 $${money(row.receivable??0)}`}/></div><div className="execution-layout"><Panel title="历史执行轨迹" action={<Pill tone={playing?'teal':''} dot>{playing?'正在播放':'已暂停'}</Pill>}><NavChart rows={chart.slice(0,cursor+1)} compact height={260}/><div className="replay-controls"><button className="play-button" disabled={!selectedId||playBusy} aria-label={playing?'暂停回放':'播放回放'} onClick={toggle}>{playBusy?<Loader2 className="spin" size={21}/>:playing?<Pause size={21}/>:<Play size={21}/>}</button><button className="icon-button bordered" aria-label="前进一个交易日" disabled={cursor>=daily.length-1} onClick={()=>setCursor(Math.min(cursor+1,daily.length-1))}><SkipForward size={18}/></button><div className="replay-timeline"><input aria-label="历史回放进度" type="range" min={0} max={daily.length-1} value={cursor} onChange={e=>setCursor(Number(e.target.value))}/><div><span>{dateLabel(daily[0].date)}</span><strong>{date}</strong><span>{dateLabel(daily[daily.length-1].date)}</span></div></div><select aria-label="播放速度" value={speed} onChange={e=>setSpeed(Number(e.target.value))}><option value={1}>1×</option><option value={5}>5×</option><option value={20}>20×</option></select></div><div className="walkthrough-jump"><button className="button secondary" onClick={jumpToFill} disabled={jumpIndex < 0 || jumpIndex === cursor || playBusy}>
    <SkipForward size={15}/>{nextFill ? '下一笔成交' : '回到首笔成交'}
  </button>{!fills.length && <span>这段历史内没有模拟成交。</span>}</div><div className="execution-pipeline"><div><span>01</span><strong>收盘信号</strong><small>仅使用当时可得数据</small></div><ArrowRight size={16}/><div><span>02</span><strong>目标仓位意图</strong><small>现金 / 库存约束</small></div><ArrowRight size={16}/><div><span>03</span><strong>次日开盘成交</strong><small>计入佣金与滑点</small></div></div></Panel><Panel title="执行事件" eyebrow="HISTORICAL FILLS" action={<span className="small muted">截至 {date}</span>}><div className="execution-events">{pastFills.length?pastFills.map((fill,index)=><div className="execution-event" key={`${fill.date}-${index}`}><span className={`event-direction ${Number(fill.quantity)>0?'buy':'sell'}`}>{Number(fill.quantity)>0?<ArrowDownLeft size={16}/>:<ArrowUpRight size={16}/>}</span><div><div><strong>{Number(fill.quantity)>0?'买入':'卖出'} {currentInstance?.symbol}</strong><span className="mono">{money(Math.abs(fill.quantity),0)} 份</span></div><p>信号 {dateLabel(fill.signal_date)} → 成交 {dateLabel(fill.date)}</p><small>${money(fill.price)} / 份 · 佣金 ${money(fill.commission)}</small></div></div>):<Empty icon={Clock3} title="尚无成交事件" description="推进历史日期，查看策略实际产生的成交。"/>}</div></Panel></div></>:<Empty title="选择一个研究实例" description="创建实例后，即可播放该实例的真实历史回测。" icon={Cpu}/>}</>;
}
