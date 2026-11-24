import { useEffect, useRef, useState } from 'react';
import { ArrowRight, Bot, Check, CheckCircle2, ChevronDown, CircleAlert, Clock3, Download, FileSearch, Fingerprint, Layers3, Loader2, Play, RotateCcw, ScanLine, ShieldCheck, Terminal, XCircle } from 'lucide-react';
import { request } from './api';
import './research-agent.css';

type Profile = 'curve_audit' | 'validation';
type SymbolName = 'SPY' | 'GLD';
type Step = { tool: string; label: string; status: string; summary: unknown };
type Response = {
  profile: Profile;
  symbol: SymbolName;
  steps: Step[];
  summary: unknown;
  findings?: { label: string; value: unknown }[];
  protocol?: unknown;
};
type Run = { data: Response; finishedAt: string; elapsedMs: number };

const profiles = {
  curve_audit: {
    title: '核对资金曲线与交易账本',
    question: '页面上的结果，能和每天的钱、每笔买卖对上吗？',
    description: '调用本地审计工具核对历史结果，检查资产计算、成交记录和曲线的一致性。',
    icon: ScanLine,
    button: '开始曲线审计',
  },
  validation: {
    title: '检查参数怎么选、怎么验证',
    question: '这些参数用了哪些数据？后面的历史有没有参与选择？',
    description: '读取研究协议与参数选择记录，检查训练、验证和最终测试的划分及对应证据。',
    icon: FileSearch,
    button: '开始验证检查',
  },
};

function present(value: unknown): string {
  if (value === null || value === undefined) return '未提供';
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '非有限值';
  return JSON.stringify(value, null, 2);
}

function StepStatus({ status }: { status: string }) {
  const state = status.toLowerCase();
  if (['completed', 'complete', 'success', 'passed', 'ok', 'done'].includes(state)) return <span className="research-agent-step-state passed"><CheckCircle2 size={14} />已完成</span>;
  if (['error', 'failed', 'fail'].includes(state)) return <span className="research-agent-step-state failed"><XCircle size={14} />失败</span>;
  if (['warning', 'warn', 'partial'].includes(state)) return <span className="research-agent-step-state warning"><CircleAlert size={14} />需关注</span>;
  if (['running', 'pending', 'queued'].includes(state)) return <span className="research-agent-step-state waiting"><Clock3 size={14} />{state === 'running' ? '运行中' : '等待中'}</span>;
  if (state === 'skipped') return <span className="research-agent-step-state"><CircleAlert size={14} />已跳过</span>;
  return <span className="research-agent-step-state"><CircleAlert size={14} />{status || '未提供状态'}</span>;
}

function saveRun(run: Run) {
  const content = { completed_at: run.finishedAt, elapsed_ms: run.elapsedMs, ...run.data };
  const blob = new Blob([JSON.stringify(content, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `research-agent_${run.data.profile}_${run.data.symbol}_${run.finishedAt.slice(0, 10)}.json`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function ResearchAgent() {
  const [profile, setProfile] = useState<Profile>('curve_audit');
  const [symbol, setSymbol] = useState<SymbolName>('SPY');
  const [running, setRunning] = useState(false);
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const controller = useRef<AbortController | null>(null);
  const startedAt = useRef(0);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; controller.current?.abort(); };
  }, []);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setElapsed(Math.floor((performance.now() - startedAt.current) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [running]);

  const start = async () => {
    if (running) return;
    const abort = new AbortController();
    controller.current = abort;
    startedAt.current = performance.now();
    setElapsed(0); setRunning(true); setError(''); setRun(null);
    try {
      const response = await request<Response>('/research-agent', {
        method: 'POST', body: JSON.stringify({ profile, symbol }), signal: abort.signal,
      });
      if (!mounted.current) return;
      if (!response || !Array.isArray(response.steps) || typeof response.profile !== 'string' || typeof response.symbol !== 'string') {
        throw new Error('研究服务返回的数据不完整，缺少任务、标的或工具步骤。');
      }
      if (response.profile !== profile || response.symbol !== symbol) throw new Error('服务返回的任务或标的与本次请求不同，请重试。');
      setRun({ data: response, finishedAt: new Date().toISOString(), elapsedMs: Math.round(performance.now() - startedAt.current) });
    } catch (reason) {
      if (!mounted.current || (reason as Error).name === 'AbortError') return;
      setError((reason as Error).message || '暂时无法完成研究检查，请确认本地服务正在运行。');
    } finally { if (mounted.current) setRunning(false); }
  };

  const selected = profiles[profile];
  const result = run?.data;
  const stale = !!result && (result.profile !== profile || result.symbol !== symbol);
  const failures = result?.steps.filter(step => ['error', 'failed', 'fail'].includes(step.status?.toLowerCase())).length || 0;

  return <div className="research-agent">
    <section className="research-agent-intro"><span className="research-agent-emblem"><Bot size={28} /></span><div><span className="eyebrow">TOOL-CALLING RESEARCH WORKFLOW</span><h2>让研究助手实际调用工具，帮你检查结果</h2><p>选一个问题，点击开始。助手会调用本地研究工具，把检查步骤和证据一起返回。</p></div><span className="pill teal"><Terminal size={12} />本地工具流程</span></section>

    <section className="panel research-agent-config" aria-labelledby="research-agent-task-title">
      <div className="panel-head"><div><h2 id="research-agent-task-title">这次想检查什么？</h2><span className="research-agent-panel-note">选择任务不会执行计算，点击“开始”才发起请求。</span></div><span className="pill">手动启动</span></div>
      <div className="research-agent-profiles" role="radiogroup" aria-label="研究检查任务">{(Object.keys(profiles) as Profile[]).map(key => {
        const option = profiles[key];
        return <button key={key} type="button" role="radio" aria-checked={key === profile} disabled={running} className={`research-agent-profile ${key === profile ? 'selected' : ''}`} onClick={() => setProfile(key)}>
          <div className="research-agent-profile-top"><span><option.icon size={22} /></span><i>{key === profile && <Check size={13} />}</i></div><strong>{option.title}</strong><span className="research-agent-question">{option.question}</span><p>{option.description}</p>
        </button>;
      })}</div>
      <div className="research-agent-actionbar"><label className="field"><span>检查标的</span><select value={symbol} disabled={running} onChange={event => setSymbol(event.target.value as SymbolName)}><option value="SPY">SPY · 标普 500 ETF</option><option value="GLD">GLD · 黄金 ETF</option></select></label><div className="research-agent-example"><span>示例任务</span><p>{profile === 'curve_audit' ? `检查 ${symbol} 的历史资金曲线，告诉我哪些记录对得上、哪些需要关注。` : `检查 ${symbol} 对应的参数研究流程，说明数据划分和结果边界。`}</p></div><button type="button" className="button primary" disabled={running} onClick={() => void start()}>{running ? <Loader2 size={16} className="spin" /> : <Play size={16} />}{running ? '正在调用工具' : selected.button}</button></div>
      <div className="research-agent-mode"><Fingerprint size={15} /><p><strong>当前采用内置工具工作流：</strong>任务和步骤由代码编排，结论依据工具返回的数据。此页不调用大语言模型；历史审计结果也不代表未来交易表现。</p></div>
    </section>

    {running && <section className="research-agent-running" role="status" aria-live="polite"><span><Loader2 size={24} className="spin" /></span><div><strong>正在执行：{selected.title} · {symbol}</strong><p>请求已发给本地研究服务。完成后显示工具实际返回的调用轨迹，等待期间不预填检查结果。</p></div><span className="research-agent-elapsed">已等待 {elapsed} 秒</span></section>}
    {error && <section className="research-agent-error" role="alert"><CircleAlert size={22} /><div><strong>本次检查未完成</strong><p>{error}</p></div><button type="button" className="button secondary" onClick={() => void start()}><RotateCcw size={14} />重新检查</button></section>}

    {run && result && <>
      <div className="research-agent-result-context"><div><span className="live-square" /><strong>{profiles[result.profile]?.title || result.profile} · {result.symbol}</strong><span className={`pill ${failures ? 'coral' : 'teal'}`}>工具流程已返回</span>{stale && <span className="pill coral">上次任务结果</span>}</div><button type="button" className="text-button" onClick={() => saveRun(run)}><Download size={14} />下载本次 JSON</button></div>
      {stale && <p className="research-agent-stale">上次结果仍对应 {result.symbol} 的“{profiles[result.profile]?.title || result.profile}”。新选择需要点击上方开始按钮后才会运行。</p>}
      <section className="panel research-agent-conclusion" aria-labelledby="research-agent-summary"><div className="panel-head"><div><span className="eyebrow">RESEARCH FINDINGS</span><h2 id="research-agent-summary">这次检查告诉我们什么？</h2></div><ShieldCheck size={21} /></div><div className="research-agent-summary">{present(result.summary)}</div>{result.findings && result.findings.length > 0 && <dl className="research-agent-findings">{result.findings.map((finding, index) => <div key={`${finding.label}-${index}`}><dt>{finding.label}</dt><dd>{present(finding.value)}</dd></div>)}</dl>}<div className="research-agent-result-meta"><Clock3 size={13} /><span>完成于 {new Date(run.finishedAt).toLocaleString('zh-CN', { hour12: false })}</span><span>{(run.elapsedMs / 1000).toFixed(2)} 秒 · {result.steps.length} 个返回步骤</span></div></section>
      <section className="panel research-agent-trace" aria-labelledby="research-agent-trace-title"><div className="panel-head"><div><h2 id="research-agent-trace-title">实际工具调用记录</h2><span className="research-agent-panel-note">以下名称、状态与摘要均由本地服务返回，可展开查看工具标识。</span></div><Terminal size={19} /></div>{result.steps.length ? <ol>{result.steps.map((step, index) => <li key={`${step.tool}-${index}`}><span className="research-agent-step-number">{String(index + 1).padStart(2, '0')}</span><div className="research-agent-step-body"><div className="research-agent-step-heading"><h3>{step.label || step.tool || '未命名工具'}</h3><StepStatus status={String(step.status || '')} /></div><p>{present(step.summary)}</p><details><summary>工具标识 <ChevronDown size={12} /></summary><code>{step.tool || '未提供'}</code></details></div></li>)}</ol> : <div className="research-agent-empty-trace"><CircleAlert size={22} /><p>服务没有返回工具步骤，无法据此确认完成了哪些检查。</p></div>}</section>
      {result.protocol != null && <details className="research-agent-protocol"><summary><span><Layers3 size={16} />查看本次使用的研究协议与数据范围</span><ChevronDown size={16} /></summary><pre>{present(result.protocol)}</pre></details>}
    </>}

    {!running && !run && !error && <section className="research-agent-first"><span><Bot size={31} /></span><h3>先提出一个具体问题，再检查证据</h3><p>左边的任务检查“数字能否对上”，右边的任务检查“参数如何验证”。任选一个任务，再点击开始。</p><div><span>选择检查任务</span><ArrowRight size={14} /><span>调用本地工具</span><ArrowRight size={14} /><span>查看发现与记录</span></div></section>}
  </div>;
}
