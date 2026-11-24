"use strict";

(() => {
  const METRICS = [
    {key:"cagr_pct",label:"年化收益",unit:"%",digits:2,preferred:"higher",help:"把期初到期末的复合收益折算为一年，按每年 252 个交易日计算。它不是逐年收益的简单平均。"},
    {key:"sharpe",label:"夏普比率",unit:"",digits:3,preferred:"higher",help:"平均每日收益除以每日收益波动，再按 252 个交易日年化。无风险利率设为零；较高仅表示这个区间收益相对波动更高。"},
    {key:"max_drawdown_pct",label:"最大回撤",unit:"%",digits:2,preferred:"lower",help:"包含初始本金的完整净值曲线，从历史峰值跌至后续低点的最大比例。这里用正数表示跌幅，较小代表历史损失幅度较小。"},
    {key:"annual_volatility_pct",label:"年化波动",unit:"%",digits:2,preferred:"lower",help:"成本后每日账户收益的样本标准差乘以 √252。波动描述收益起伏，不等于最坏亏损。"},
    {key:"mean_exposure_pct",label:"平均资产仓位",unit:"%",digits:2,preferred:null,help:"每天持有资产市值占账户总权益的比例，再对区间取平均。更低仓位通常带来更低风险，也可能降低收益。"},
    {key:"annual_turnover",label:"年换手",unit:" 倍",digits:2,preferred:"lower",help:"成交名义金额绝对值之和除以初始本金，再按区间交易日年化；包含最初建仓。它衡量交易频率与成本压力。"},
  ];
  const state = {symbol:"SPY",study:"sigmoid",metric:"cagr_pct",data:null};
  const $ = id => document.getElementById(id);
  const el = (tag,className,text) => {const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=text;return node;};
  const format = (value,metric) => Number(value).toFixed(metric.digits)+metric.unit;
  const signed = (value,digits=2) => `${value>0?"+":""}${Number(value).toFixed(digits)}`;
  const cleanZero = value => Math.abs(value)<.00000001?0:value;

  function closeHelp(except) {
    document.querySelectorAll(".help-button").forEach(button=>{if(button!==except){button.setAttribute("aria-expanded","false");const note=document.getElementById(button.getAttribute("aria-controls"));if(note)note.hidden=true;}});
  }

  function metricCard(metric,selected,baseline) {
    const card=el("article","metric-card");
    const top=el("div","metric-top");top.append(el("span","",metric.label));
    const help=el("div","metric-help");const button=el("button","help-button","?");button.type="button";button.setAttribute("aria-label",`${metric.label}的计算口径`);button.setAttribute("aria-expanded","false");button.setAttribute("aria-controls",`help-${metric.key}`);
    const description=el("p","help-description",metric.help);description.id=`help-${metric.key}`;description.hidden=true;
    button.addEventListener("click",()=>{const open=button.getAttribute("aria-expanded")==="true";closeHelp(button);button.setAttribute("aria-expanded",String(!open));description.hidden=open;});help.append(button,description);top.append(help);
    const values=el("div","metric-values");values.append(el("strong","metric-primary",format(selected[metric.key],metric)),el("span","metric-separator","vs"),el("span","metric-benchmark",format(baseline[metric.key],metric)));
    const footer=el("div","metric-footer");footer.append(el("span","",metric.preferred==="higher"?"较高更好":metric.preferred==="lower"?"较低更好":"结合收益与风险理解"));
    const diff=cleanZero(selected[metric.key]-baseline[metric.key]);const better=metric.preferred==="higher"?diff>=0:metric.preferred==="lower"?diff<=0:true;
    let difference=metric.unit==="%"?`${signed(diff)} 个百分点`:signed(diff,metric.digits)+(metric.unit||"");
    if(metric.preferred===null)difference=`相差 ${signed(diff)} 个百分点`;
    footer.append(el("span",`metric-delta${better?"":" caution"}`,difference));card.append(top,values,footer);return card;
  }

  function renderChart(selected,baseline,study) {
    const metric=METRICS.find(item=>item.key===state.metric);
    const root=$("aggregate-chart");root.replaceChildren();
    const maximum=Math.max(selected[metric.key],baseline[metric.key],.001)*1.12;
    [[study.short_label,selected[metric.key],false],["固定半仓基准",baseline[metric.key],true]].forEach(([label,value,benchmark])=>{
      const row=el("div",`bar-row${benchmark?" benchmark":""}`),heading=el("div","bar-row-label");heading.append(el("span","",label),el("strong","",format(value,metric)));
      const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("viewBox","0 0 400 12");svg.setAttribute("class","bar-svg");svg.setAttribute("role","img");svg.setAttribute("aria-label",`${label}：${metric.label} ${format(value,metric)}`);
      const track=document.createElementNS(svg.namespaceURI,"rect");track.setAttribute("width","400");track.setAttribute("height","12");track.setAttribute("rx","3");track.setAttribute("class","bar-track");
      const bar=document.createElementNS(svg.namespaceURI,"rect");bar.setAttribute("width",String(Math.max(0,value)/maximum*400));bar.setAttribute("height","12");bar.setAttribute("rx","3");bar.setAttribute("class","bar-fill");svg.append(track,bar);row.append(heading,svg);root.append(row);
    });
  }

  function render() {
    if(!state.data)return;
    const study=state.data.studies[state.study],asset=study.assets[state.symbol],baseline=state.data.baselines[state.symbol];
    document.querySelectorAll("[data-study]").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.study===state.study)));
    document.querySelectorAll("[data-symbol]").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.symbol===state.symbol)));
    $("study-title").textContent=study.title;$("strategy-legend").textContent=study.short_label;$("asset-name").textContent=`${state.symbol} · ${state.data.assets[state.symbol]}`;
    $("source-label").textContent=study.source_label;$("strategy-description").textContent=study.description;
    $("metrics-grid").replaceChildren(...METRICS.map(metric=>metricCard(metric,asset.metrics,baseline)));
    $("finding-copy").textContent=asset.finding;$("study-source").href=study.report_url;
    const uncertainty=$("uncertainty");uncertainty.replaceChildren();uncertainty.hidden=!asset.uncertainty;
    if(asset.uncertainty){const u=asset.uncertainty;uncertainty.append(el("strong","",`夏普差 ${signed(u.sharpe_delta,3)}，95% 区间 [${signed(u.ci_lower,3)}, ${signed(u.ci_upper,3)}]`),el("p","","配对移动区块 bootstrap：20 日区块、5,000 次抽样。区间包含零；未通过预先固定的证据门槛。"));}
    $("coverage-note").textContent=`${state.data.period.start} — ${state.data.period.end} · ${state.data.period.sessions.toLocaleString("en-US")} 个交易日 · 每组初始资金 $100,000`;
    renderChart(asset.metrics,baseline,study);
    $("result-body").setAttribute("aria-busy","false");
    $("live-announcement").textContent=`已显示${state.symbol}的${study.title}。${asset.finding}`;
  }

  async function load() {
    $("load-error").hidden=true;$("result-body").setAttribute("aria-busy","true");
    try {
      const response=await fetch("./summary.json",{credentials:"omit",cache:"no-cache"});
      if(!response.ok)throw new Error("Summary unavailable");
      const data=await response.json();
      if(data.schema_version!==1||!data.studies?.sigmoid?.assets?.SPY||!data.studies?.regime?.assets?.GLD)throw new Error("Invalid summary");
      state.data=data;render();
    } catch {$("load-error").hidden=false;$("result-body").setAttribute("aria-busy","false");}
  }

  document.querySelectorAll("[data-study]").forEach(button=>button.addEventListener("click",()=>{state.study=button.dataset.study;render();}));
  document.querySelectorAll("[data-symbol]").forEach(button=>button.addEventListener("click",()=>{state.symbol=button.dataset.symbol;render();}));
  $("chart-metric").addEventListener("change",event=>{state.metric=event.target.value;render();});
  $("retry-load").addEventListener("click",load);
  document.addEventListener("keydown",event=>{if(event.key==="Escape")closeHelp();});
  document.addEventListener("click",event=>{if(!event.target.closest(".metric-help"))closeHelp();});
  void load();
})();
