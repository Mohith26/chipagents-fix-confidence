import * as C from './src/confirm.js';

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 1) => (x == null || Number.isNaN(x) ? '–' : (100 * x).toFixed(d) + '%');
const fmtInt = (x) => Math.round(x).toLocaleString('en-US');
const hrs = (s) => { const h = s / 3600; return h >= 100 ? fmtInt(h) : h >= 10 ? h.toFixed(1) : h.toFixed(2); };
const day = (d) => d ? d.replace(/\./g, '-') : '–';

const STORE = 'chipagents-fixconf-v1';
const DEFAULTS = { delta: 0.05, pi: 0.563 };
const state = { data: null, settings: { ...DEFAULTS }, statuses: {}, filter: 'all', q: '', sort: { key: 'risk', dir: -1 }, sel: 0, triageSort: { key: 'fails', dir: -1 } };

function load() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE) || '{}');
    if (s.settings) state.settings = { ...DEFAULTS, ...s.settings };
    if (s.statuses) state.statuses = s.statuses;
  } catch { /* corrupted storage: start fresh */ }
}
function save() { localStorage.setItem(STORE, JSON.stringify({ settings: state.settings, statuses: state.statuses })); }

// ------------------------------------------------------------------ model per bin (memoised)
const cache = new Map();
function model(b) {
  const { delta, pi } = state.settings;
  const key = `${b.id}|${delta}|${pi}`;
  if (cache.has(key)) return cache.get(key);
  const items = b.tests.map((t) => ({ id: t.name, ...C.posterior(t.pooledFails, t.pooledRuns), cost: Math.max(1, t.rt) }));
  const runs = Object.fromEntries(b.tests.map((t) => [t.name, t.runs]));
  const nightly = C.rerunNightly(items, runs);
  const nightlyRisk = C.residualRisk(nightly.escape, pi);
  const target = C.requiredEscape(delta, pi);
  const plan = C.solvePlan(items, target);
  const seeds = Object.values(plan.plan).reduce((s, x) => s + x, 0);
  const witnessCost = b.witness.reduce((s, w) => s + (b.tests.find((t) => w.includes('.' + t.name + '.'))?.rt || 0), 0);
  const directed = !plan.feasible || plan.cost > b.ipCpu;
  const verdict = nightlyRisk <= delta ? 'nightly' : directed ? 'directed' : 'targeted';
  const rate = C.posteriorMean({ a: items.reduce((s, i) => s + i.a, 0), b: items.reduce((s, i) => s + i.b, 0) });
  const fails = b.tests.reduce((s, t) => s + t.fails, 0), runsT = b.tests.reduce((s, t) => s + t.runs, 0);
  const m = { items, nightly, nightlyRisk, target, plan, seeds, witnessCost, directed, verdict, rate, fails, runs: runsT,
    planRisk: C.residualRisk(plan.escape, pi) };
  cache.set(key, m);
  return m;
}
const status = (b) => state.statuses[b.id]?.status || 'open';
const STATUS_LABEL = { open: 'Open', confirming: 'Confirming', confirmed: 'Confirmed', reopened: 'Reopened' };
const VERDICT = { nightly: 'Nightly is enough', targeted: 'Targeted reruns', directed: 'Directed test' };

// ------------------------------------------------------------------ boot
async function boot() {
  load();
  wireChrome();
  try {
    const res = await fetch('data/app.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.data = await res.json();
  } catch (e) {
    $('#stage').innerHTML = `<div class="card empty"><b>Couldn't load the regression</b>data/app.json failed to load (${esc(e.message)}).<div class="actions" style="justify-content:center"><button class="btn" id="retry">Retry</button></div></div>`;
    $('#retry').onclick = () => location.reload();
    return;
  }
  const d = state.data;
  $('#navCount').textContent = d.regression.bins.length;
  const revs = [...new Set(d.regression.ips.map((i) => i.rev))];
  $('#runInfo').innerHTML = `nightly <b>${day(d.regression.day)}</b> · rev ${esc(revs[0])} · ${d.regression.ips.length} IPs · ${fmtInt(d.regression.ips.reduce((s, i) => s + i.runs, 0))} sims`;
  $('#buildInfo').textContent = `posterior window ${d.config.W} nightlies`;
  window.addEventListener('hashchange', route);
  route();
}

function wireChrome() {
  const app = $('#app');
  const setCollapsed = (v) => { app.classList.toggle('collapsed', v); localStorage.setItem(STORE + '-side', v ? '1' : ''); };
  setCollapsed(!!localStorage.getItem(STORE + '-side'));
  $('#tog').onclick = () => setCollapsed(!app.classList.contains('collapsed'));
  document.querySelectorAll('.side .nav').forEach((n) => {
    n.onclick = () => {
      if (n.dataset.ws === 'rca') { location.hash = '#/confirm'; return; }
      toast(`${n.textContent.trim()} isn't part of this concept demo. It adds Fix confirmation to RCA.`);
    };
  });
  $('#proj').onclick = () => toast('One project in this demo: OpenTitan top_earlgrey nightly regression.');
  $('#kbdHelp').onclick = () => $('#help').showModal();
  document.addEventListener('keydown', onKey);
  const tip = $('#tip');
  document.addEventListener('mousemove', (e) => {
    const t = e.target.closest('[data-tip]');
    if (!t) { tip.style.display = 'none'; return; }
    tip.innerHTML = t.dataset.tip; tip.style.display = 'block';
    const x = Math.min(window.innerWidth - tip.offsetWidth - 12, e.clientX + 14);
    tip.style.left = x + 'px'; tip.style.top = (e.clientY + 16) + 'px';
  });
}

function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('on');
  clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove('on'), 2600);
}

function onKey(e) {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') { e.preventDefault(); $('#tog').click(); return; }
  const typing = /INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName);
  if (e.key === 'Escape') {
    if (typing) { document.activeElement.blur(); return; }
    if (location.hash.startsWith('#/bin/')) history.length > 1 ? history.back() : (location.hash = '#/confirm');
    return;
  }
  if (typing) return;
  if (e.key === '?') { $('#help').showModal(); return; }
  if (e.key === '/') { const s = $('#q'); if (s) { e.preventDefault(); s.focus(); } return; }
  const rows = [...document.querySelectorAll('tr.rowlink')];
  if ((e.key === 'j' || e.key === 'k') && rows.length) {
    state.sel = Math.max(0, Math.min(rows.length - 1, state.sel + (e.key === 'j' ? 1 : -1)));
    rows.forEach((r, i) => r.classList.toggle('sel', i === state.sel));
    rows[state.sel].scrollIntoView({ block: 'nearest' });
    return;
  }
  if (e.key === 'Enter' && rows[state.sel]) { location.hash = rows[state.sel].dataset.href; return; }
  if (e.key === 'a') { const b = $('#approve'); if (b && !b.disabled) b.click(); }
}

// ------------------------------------------------------------------ router
function route() {
  const h = location.hash || '#/confirm';
  const [, view, arg] = h.split('/');
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('on', t.dataset.tab === (view === 'bin' ? state.lastTab || 'confirm' : view)));
  const crumb = $('#crumb');
  const base = `<a href="#/confirm">opentitan</a> / <a href="#/confirm">RCA</a> / `;
  state.sel = 0;
  if (view === 'bin') {
    const b = state.data.regression.bins.find((x) => x.id === decodeURIComponent(arg || ''));
    if (!b) { $('#stage').innerHTML = `<div class="card empty"><b>Bin not found</b>No failure bin with id ${esc(arg || '')} in this nightly. <a href="#/confirm">Back to Fix confirmation</a></div>`; return; }
    crumb.innerHTML = base + `<a href="#/${state.lastTab || 'confirm'}">${state.lastTab === 'triage' ? 'Regression triage' : 'Fix confirmation'}</a> / <b>${esc(b.ip)} · ${esc(b.id.split(':')[1])}</b>`;
    renderBin(b);
  } else if (view === 'triage') { state.lastTab = 'triage'; crumb.innerHTML = base + '<b>Regression triage</b>'; renderTriage(); }
  else if (view === 'backtest') { crumb.innerHTML = base + '<b>Backtest</b>'; renderBacktest(); }
  else { state.lastTab = 'confirm'; crumb.innerHTML = base + '<b>Fix confirmation</b>'; renderConfirm(); }
  $('#headActions').innerHTML = '';
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ shared bits
function histCells(h) {
  return `<div class="hist" aria-label="last ${h.length} nightlies">${h.map((f, i) => {
    const cls = f == null ? 'x' : f === 0 ? '' : f < 3 ? 'f1' : f < 10 ? 'f2' : 'f3';
    const label = f == null ? 'tests not run' : f === 0 ? 'clean' : `${f} failing run${f > 1 ? 's' : ''}`;
    return `<i class="${cls}" data-tip="nightly −${h.length - 1 - i}: ${label}"></i>`;
  }).join('')}</div>`;
}
const riskCls = (r, d) => (r > 4 * d ? 'hi' : r > d ? 'mid' : 'lo');
const kindBadge = (k) => `<span class="badge ${k}">${k}</span>`;
const stBadge = (b) => { const s = status(b); return `<span class="st ${s}"><i></i>${STATUS_LABEL[s]}</span>`; };

function stackedBars(groups, segs, height = 190) {
  // groups: [{label, values:[...segs], tip}]
  const max = Math.max(1, ...groups.map((g) => g.values.reduce((s, x) => s + x, 0)));
  const cols = `grid-template-columns:repeat(${groups.length},minmax(0,1fr))`;
  return `<div class="bars" style="${cols};height:${height}px">${groups.map((g) => `<div class="col" data-tip="${esc(g.tip)}">${
    g.values.map((v, i) => (v ? `<i style="height:${(100 * v) / max}%;background:${segs[i].color}"></i>` : '')).join('')}</div>`).join('')}</div>
    <div class="xl" style="${cols}">${groups.map((g) => `<span title="${esc(g.label)}">${esc(g.label)}</span>`).join('')}</div>
    <div class="legend">${segs.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join('')}</div>`;
}

function controls() {
  const { delta, pi } = state.settings;
  return `<div class="toolbar">
    <span class="ctl">Residual risk target <span class="seg" role="group" aria-label="Residual risk target">${[0.01, 0.05, 0.1].map((d) => `<button data-delta="${d}" class="${d === delta ? 'on' : ''}" aria-pressed="${d === delta}">${d * 100}%</button>`).join('')}</span></span>
    <label class="ctl">Fix confidence prior <select id="pi" aria-label="Prior probability that a proposed fix works">
      ${[[0.3, '30%'], [0.563, '56.3% (RCA top-1)'], [0.8, '80%'], [0.95, '95%']].map(([v, l]) => `<option value="${v}" ${v === pi ? 'selected' : ''}>${l}</option>`).join('')}
    </select></label></div>`;
}
function wireControls(rerender) {
  document.querySelectorAll('[data-delta]').forEach((b) => { b.onclick = () => { state.settings.delta = +b.dataset.delta; save(); rerender(); }; });
  const s = $('#pi'); if (s) s.onchange = () => { state.settings.pi = +s.value; save(); rerender(); };
}

// ------------------------------------------------------------------ Fix confirmation (new)
function renderConfirm() {
  const d = state.data, bins = d.regression.bins;
  const { delta } = state.settings;
  const ms = bins.map((b) => [b, model(b)]);
  const unsafe = ms.filter(([, m]) => m.nightlyRisk > delta);
  const targeted = ms.filter(([, m]) => m.verdict === 'targeted');
  const directed = ms.filter(([, m]) => m.verdict === 'directed');
  const planCpu = ms.filter(([, m]) => m.verdict !== 'directed').reduce((s, [, m]) => s + m.plan.cost, 0);
  const nightlyCpu = ms.filter(([, m]) => m.verdict !== 'directed').reduce((s, [, m]) => s + m.nightly.cost, 0);
  const bt = d.backtest;
  const counts = { all: bins.length, targeted: targeted.length, directed: directed.length, nightly: ms.length - targeted.length - directed.length, progress: bins.filter((b) => status(b) === 'confirming').length };

  // per-IP stacked verdict bars (top 12 IPs by bins)
  const byIp = {};
  for (const [b, m] of ms) { (byIp[b.ip] ||= { nightly: 0, targeted: 0, directed: 0 })[m.verdict]++; }
  const ipRows = Object.entries(byIp).sort((a, b) => (b[1].nightly + b[1].targeted + b[1].directed) - (a[1].nightly + a[1].targeted + a[1].directed)).slice(0, 12);
  const segs = [{ label: 'Nightly is enough', color: '#7a2d10' }, { label: 'Targeted reruns', color: '#fd6420' }, { label: 'Directed test (reruns too costly)', color: '#ffd2bc' }];

  $('#stage').innerHTML = `
  <div class="tiles">
    <div class="tile"><div class="k"><span>Failure bins</span><span>${day(d.regression.day)}</span></div><b>${bins.length}</b><span class="t">across ${new Set(bins.map((b) => b.ip)).size} IPs after conventional binning (infra failures removed)</span></div>
    <div class="tile"><div class="k"><span>Unsafe after one clean nightly</span></div><b>${unsafe.length}<small>/ ${bins.length}</small></b><span class="t">bins still above ${pct(delta, 0)} chance the bug is there if the fix lands and the next nightly is clean</span></div>
    <div class="tile"><div class="k"><span>Confirmation reruns</span><span>exact</span></div><b>${hrs(planCpu)}<small>CPU-h</small></b><span class="t">to reach ${pct(delta, 0)} on ${targeted.length + counts.nightly} bins, vs ${hrs(nightlyCpu)} CPU-h rerunning their tests at nightly seed counts</span></div>
    <div class="tile"><div class="k"><span>Came back after vanishing</span><span>held-out</span></div><b>${pct(bt.episodes.cameBackTest, 0)}</b><span class="t">of signatures that vanished from a nightly returned within ${d.config.horizon} nightlies (Jul to Oct 2024)</span></div>
  </div>
  <div class="grid2">
    <div class="card">
      <div class="ch"><div><div class="ct">How each bin gets confirmed</div><div class="cs">Cheapest evidence that a fix worked, per bin, at a ${pct(delta, 0)} residual-risk target</div></div><div class="dots"><i style="background:#7a2d10"></i><i style="background:#fd6420"></i><i style="background:#ffd2bc"></i></div></div>
      ${stackedBars(ipRows.map(([ip, v]) => ({ label: ip, values: [v.nightly, v.targeted, v.directed], tip: `<b>${ip}</b><br>${v.nightly} nightly is enough · ${v.targeted} targeted · ${v.directed} directed test` })), segs)}
      <div class="cf"><b>${targeted.length} bins</b> need targeted reruns before a final regression · <b>${directed.length}</b> are too rare to confirm by reruns</div>
      <div class="pb"><i style="width:${(100 * counts.nightly) / bins.length}%"></i></div>
    </div>
    <div class="card">
      <div class="ch"><div><div class="ct">Why a clean nightly isn't proof</div><div class="cs">Each failing test runs a few seeds a night. A bug that fires on 1 seed in 300 survives a clean 300-seed nightly about a third of the time.</div></div></div>
      ${riskScatter(ms)}
    </div>
  </div>
  <div class="card" style="padding-bottom:8px">
    <div class="ch" style="margin-bottom:12px"><div><div class="ct">Fix confirmation plans</div><div class="cs">For each bin: the posterior failure rate from the last ${d.config.W} nightlies, the risk left by one clean nightly, and the cheapest rerun plan at the fix revision.</div></div>
      <div class="toolbar"><button class="btn ghost" id="csv"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>Export CSV</button></div></div>
    <div class="toolbar" style="justify-content:space-between;margin-bottom:6px">
      <div class="toolbar">
        <label class="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg><span class="sr">Search bins</span><input id="q" placeholder="Search signature, IP or test" value="${esc(state.q)}"><span class="kbd">/</span></label>
        ${[['all', 'All'], ['targeted', 'Targeted reruns'], ['directed', 'Directed test'], ['nightly', 'Nightly is enough'], ['progress', 'Confirming']].map(([k, l]) => `<button class="chip ${state.filter === k ? 'on' : ''}" data-f="${k}">${l} <span class="n">${counts[k]}</span></button>`).join('')}
      </div>
      ${controls()}
    </div>
    <div id="tbl"></div>
  </div>`;
  wireControls(renderConfirm);
  document.querySelectorAll('[data-f]').forEach((c) => { c.onclick = () => { state.filter = c.dataset.f; renderConfirm(); }; });
  const q = $('#q'); q.oninput = () => { state.q = q.value; drawConfirmTable(); };
  $('#csv').onclick = exportCsv;
  drawConfirmTable();
}

function riskScatter(ms) {
  // x: posterior per-seed failure rate (log), y: residual risk after one clean nightly
  const W = 520, H = 210, L = 44, B = 26, T = 8, R = 8;
  const pts = ms.map(([b, m]) => ({ b, x: Math.max(1e-4, m.rate), y: m.nightlyRisk, v: m.verdict }));
  const lx = (x) => L + ((Math.log10(x) + 4) / 4) * (W - L - R);
  const maxY = Math.max(0.5, ...pts.map((p) => p.y));
  const ly = (y) => T + (1 - y / maxY) * (H - T - B);
  const col = { nightly: '#7a2d10', targeted: '#fd6420', directed: '#ffd2bc' };
  const d = state.settings.delta;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Residual risk after one clean nightly against failure rate">
    <g class="grid">${[0, 0.1, 0.2, 0.3, 0.4].filter((v) => v <= maxY).map((v) => `<line x1="${L}" x2="${W - R}" y1="${ly(v)}" y2="${ly(v)}"/><text x="${L - 6}" y="${ly(v) + 3}" text-anchor="end">${v * 100}%</text>`).join('')}
    ${[1e-4, 1e-3, 1e-2, 1e-1, 1].map((v) => `<text x="${lx(v)}" y="${H - 8}" text-anchor="middle">${v >= 0.01 ? v * 100 + '%' : v * 100 + '%'}</text>`).join('')}</g>
    <line x1="${L}" x2="${W - R}" y1="${ly(d)}" y2="${ly(d)}" stroke="#fd6420" stroke-dasharray="4 4" opacity=".7"/><text x="${W - R}" y="${ly(d) - 5}" text-anchor="end" style="fill:#ff9a66">target ${pct(d, 0)}</text>
    ${pts.map((p) => `<a href="#/bin/${encodeURIComponent(p.b.id)}"><circle cx="${lx(p.x).toFixed(1)}" cy="${ly(p.y).toFixed(1)}" r="4.2" fill="${col[p.v]}" fill-opacity=".9" stroke="#0d1117" data-tip="<b>${esc(p.b.ip)}</b> · ${esc(p.b.sig.slice(0, 70))}…<br>rate ${pct(p.x, 2)} per seed · clean-nightly risk ${pct(p.y)}"/></a>`).join('')}
    <text x="${(L + W - R) / 2}" y="${H + 6}" text-anchor="middle">posterior failure rate per seed (log)</text>
  </svg>`;
}

function filteredBins() {
  const q = state.q.trim().toLowerCase();
  return state.data.regression.bins.filter((b) => {
    const m = model(b);
    if (state.filter === 'progress' && status(b) !== 'confirming') return false;
    if (['targeted', 'directed', 'nightly'].includes(state.filter) && m.verdict !== state.filter) return false;
    if (q && !(b.sig.toLowerCase().includes(q) || b.ip.includes(q) || b.tests.some((t) => t.name.includes(q)))) return false;
    return true;
  });
}

function drawConfirmTable() {
  const rows = filteredBins();
  const { key, dir } = state.sort;
  const val = { risk: (b) => model(b).nightlyRisk, cpu: (b) => (model(b).verdict === 'directed' ? Infinity : model(b).plan.cost), rate: (b) => model(b).rate, fails: (b) => model(b).fails, ip: (b) => b.ip };
  rows.sort((a, b) => { const x = val[key](a), y = val[key](b); return (x < y ? -1 : x > y ? 1 : 0) * dir; });
  const th = (k, l, cls = '') => `<th class="s ${cls}" data-k="${k}" aria-sort="${key === k ? (dir > 0 ? 'ascending' : 'descending') : 'none'}">${l} ${key === k ? `<span class="arr">${dir > 0 ? '↑' : '↓'}</span>` : ''}</th>`;
  const d = state.settings.delta;
  if (!rows.length) { $('#tbl').innerHTML = `<div class="empty"><b>No bins match</b>Try another filter or clear the search.</div>`; return; }
  $('#tbl').innerHTML = `<div class="tblwrap"><table class="tbl fixed"><colgroup><col style="width:auto"><col style="width:92px"><col style="width:84px"><col style="width:226px"><col style="width:112px"><col style="width:168px"><col style="width:112px"></colgroup><thead><tr>
    ${th('ip', 'Failure bin')}${th('fails', 'Failing', 'num')}${th('rate', 'Rate/seed', 'num')}<th>Last 30 nightlies</th>${th('risk', 'Clean-nightly risk', 'num wrap')}${th('cpu', 'Confirmation plan')}<th>Status</th></tr></thead>
    <tbody>${rows.map((b, i) => { const m = model(b); const lo = C.betaQuantile(0.05, m.items[0].a, m.items[0].b), hi = C.betaQuantile(0.95, m.items[0].a, m.items[0].b);
      return `<tr class="rowlink ${i === state.sel ? 'sel' : ''}" data-href="#/bin/${encodeURIComponent(b.id)}" tabindex="0">
      <td><div class="sig" title="${esc(b.raw)}">${esc(b.sig)}</div><div class="sub">${kindBadge(b.kind)} ${esc(b.ip)} · ${b.tests.map((t) => esc(t.name)).join(', ')}</div></td>
      <td class="num">${m.fails}<div class="sub">of ${fmtInt(m.runs)}</div></td>
      <td class="num" data-tip="90% interval ${pct(lo, 2)} to ${pct(hi, 2)} (first test)">${pct(m.rate, m.rate < 0.01 ? 2 : 1)}</td>
      <td>${histCells(b.history)}</td>
      <td class="num"><span class="risk ${riskCls(m.nightlyRisk, d)}">${pct(m.nightlyRisk)}</span></td>
      <td class="plancell ${m.verdict === 'directed' ? 'directed' : ''}">${m.verdict === 'directed' ? `<b>Directed test</b><small>${fmtInt(m.seeds)} seeds would cost ${hrs(m.plan.cost)} CPU-h</small>` : m.verdict === 'nightly' ? `<b>${fmtInt(m.seeds)} seeds</b><small>${hrs(m.plan.cost)} CPU-h · nightly is enough</small>` : `<b>${fmtInt(m.seeds)} seeds</b><small>${hrs(m.plan.cost)} CPU-h · ${pct(m.planRisk)} risk</small>`}</td>
      <td>${stBadge(b)}</td></tr>`; }).join('')}</tbody></table></div>`;
  document.querySelectorAll('#tbl th.s').forEach((h) => { h.onclick = () => { const k = h.dataset.k; state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : -1 }; drawConfirmTable(); }; });
  document.querySelectorAll('#tbl tr.rowlink').forEach((r, i) => { r.onclick = () => { location.hash = r.dataset.href; }; r.onkeydown = (e) => { if (e.key === 'Enter') location.hash = r.dataset.href; }; r.onfocus = () => { state.sel = i; }; });
}

function exportCsv() {
  const head = ['bin_id', 'ip', 'kind', 'signature', 'tests', 'failing_runs', 'runs', 'rate_per_seed', 'risk_after_clean_nightly', 'verdict', 'plan_seeds', 'plan_cpu_hours', 'plan_residual_risk', 'plan_by_test', 'status', 'target', 'fix_prior'];
  const lines = [head.join(',')];
  for (const b of state.data.regression.bins) {
    const m = model(b);
    const plan = Object.entries(m.plan.plan).map(([t, k]) => `${t}=${k}`).join(' ');
    lines.push([b.id, b.ip, b.kind, b.sig, b.tests.map((t) => t.name).join(' '), m.fails, m.runs, m.rate.toFixed(6), m.nightlyRisk.toFixed(6), m.verdict, m.seeds, (m.plan.cost / 3600).toFixed(3), m.planRisk.toFixed(6), plan, status(b), state.settings.delta, state.settings.pi]
      .map((v) => `"${String(v).replace(/"/g, '""')}"`).join(','));
  }
  download(`fix-confirmation-${state.data.regression.day}.csv`, lines.join('\n'), 'text/csv');
  toast(`Exported ${state.data.regression.bins.length} plans as CSV`);
}
function download(name, text, type) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ------------------------------------------------------------------ Regression triage (existing view)
function renderTriage() {
  const bins = state.data.regression.bins;
  const kinds = ['check', 'timeout', 'fatal', 'assertion', 'error'];
  const colors = ['#7a2d10', '#fd6420', '#ff9a66', '#ffd2bc', '#c2410c'];
  const byIp = {};
  for (const b of bins) { (byIp[b.ip] ||= Object.fromEntries(kinds.map((k) => [k, 0])))[b.kind]++; }
  const ips = Object.entries(byIp).sort((a, b) => Object.values(b[1]).reduce((s, x) => s + x) - Object.values(a[1]).reduce((s, x) => s + x)).slice(0, 12);
  const ipInfo = Object.fromEntries(state.data.regression.ips.map((i) => [i.ip, i]));
  const failingSims = state.data.regression.ips.reduce((s, i) => s + i.failing, 0);
  const pending = bins.filter((b) => model(b).nightlyRisk > state.settings.delta).length;
  $('#stage').innerHTML = `
  <div class="card">
    <div class="ch"><div><div class="ct">Root cause analysis</div><div class="cs">Failures grouped by suspected origin · ${fmtInt(failingSims)} failing simulations in ${bins.length} bins</div></div><div class="dots">${colors.slice(0, 4).map((c) => `<i style="background:${c}"></i>`).join('')}</div></div>
    ${stackedBars(ips.map(([ip, v]) => ({ label: ip, values: kinds.map((k) => v[k]), tip: `<b>${ip}</b> · ${ipInfo[ip].failing} failing of ${fmtInt(ipInfo[ip].runs)} sims<br>${kinds.filter((k) => v[k]).map((k) => `${v[k]} ${k}`).join(' · ')}` })), kinds.map((k, i) => ({ label: k, color: colors[i] })))}
    <div class="cf"><b>${bins.length} failure bins</b> from ${state.data.regression.ips.length} IP regressions · <a href="#/confirm">${pending} need more than a clean nightly to confirm a fix →</a></div>
    <div class="pb" data-tip="${bins.length - pending} of ${bins.length} bins can be confirmed by one clean nightly"><i style="width:${(100 * (bins.length - pending)) / bins.length}%"></i></div>
  </div>
  <div class="card" style="padding-bottom:8px">
    <div class="ch" style="margin-bottom:12px"><div><div class="ct">Bins</div><div class="cs">Conventional binning of dvsim failure buckets: line numbers normalised, infrastructure failures removed. Witness seeds are the failing runs to rerun with waves.</div></div>
      <label class="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg><span class="sr">Search bins</span><input id="q" placeholder="Search signature, IP or test" value="${esc(state.q)}"><span class="kbd">/</span></label></div>
    <div id="tbl"></div>
  </div>`;
  const q = $('#q'); q.oninput = () => { state.q = q.value; drawTriage(); };
  drawTriage();
}
function drawTriage() {
  const q = state.q.trim().toLowerCase();
  const rows = state.data.regression.bins.filter((b) => !q || b.sig.toLowerCase().includes(q) || b.ip.includes(q) || b.tests.some((t) => t.name.includes(q)))
    .sort((a, b) => model(b).fails - model(a).fails);
  if (!rows.length) { $('#tbl').innerHTML = `<div class="empty"><b>No bins match</b>Clear the search to see all ${state.data.regression.bins.length} bins.</div>`; return; }
  $('#tbl').innerHTML = `<div class="tblwrap"><table class="tbl fixed"><colgroup><col style="width:auto"><col style="width:90px"><col style="width:80px"><col style="width:110px"><col style="width:140px"><col style="width:230px"><col style="width:112px"></colgroup><thead><tr><th>Failure bin</th><th>Kind</th><th class="num">Failing</th><th class="num">Witness seeds</th><th>First seen</th><th>Last 30 nightlies</th><th>Status</th></tr></thead><tbody>
    ${rows.map((b, i) => `<tr class="rowlink ${i === state.sel ? 'sel' : ''}" data-href="#/bin/${encodeURIComponent(b.id)}" tabindex="0">
      <td><div class="sig" title="${esc(b.raw)}">${esc(b.sig)}</div><div class="sub">${esc(b.ip)} · ${b.tests.map((t) => esc(t.name)).join(', ')}</div></td>
      <td>${kindBadge(b.kind)}</td><td class="num">${model(b).fails}</td><td class="num">${b.witness.length}</td>
      <td class="mono" style="font-size:12px;color:var(--mut)">${day(b.firstSeen)}<div class="sub">seen ${b.seen}/${b.nights}</div></td>
      <td>${histCells(b.history)}</td><td>${stBadge(b)}</td></tr>`).join('')}</tbody></table></div>`;
  document.querySelectorAll('#tbl tr.rowlink').forEach((r, i) => { r.onclick = () => { location.hash = r.dataset.href; }; r.onkeydown = (e) => { if (e.key === 'Enter') location.hash = r.dataset.href; }; r.onfocus = () => { state.sel = i; }; });
}

// ------------------------------------------------------------------ bin detail
function cfgPath(ip) {
  const m = ip.match(/^(aes|kmac|rom_ctrl|spi_device|sram_ctrl|lc_ctrl)_(.+)$/);
  if (ip === 'otbn') return 'hw/ip/otbn/dv/uvm/otbn_sim_cfg.hjson';
  return m ? `hw/ip/${m[1]}/dv/${ip}_sim_cfg.hjson` : `hw/ip/${ip}/dv/${ip}_sim_cfg.hjson`;
}

function renderBin(b) {
  const m = model(b);
  const { delta, pi } = state.settings;
  const st = status(b);
  const rec = state.statuses[b.id];
  const witnessSeeds = b.witness.map((w) => w.split('.')[2]);
  const planRows = m.items.map((it) => ({ it, k: m.plan.plan[it.id] || 0, t: b.tests.find((t) => t.name === it.id) }));
  const cmds = [`# 1. deterministic: rerun the witness seeds at the fix revision (must pass)`,
    ...[...new Set(b.witness.map((w) => w.split('.')[1]))].map((t) => `./util/dvsim/dvsim.py ${cfgPath(b.ip)} -i ${t} --seeds ${b.witness.filter((w) => w.split('.')[1] === t).map((w) => w.split('.')[2]).join(' ')}`),
    `# 2. fresh seeds: ${pct(delta, 0)} residual-risk plan (${fmtInt(m.seeds)} seeds, ${hrs(m.plan.cost)} CPU-h)`,
    ...planRows.filter((r) => r.k).map((r) => `./util/dvsim/dvsim.py ${cfgPath(b.ip)} -i ${r.it.id} --reseed ${r.k}`)];
  const x2 = C.rerunNightly(m.items, Object.fromEntries(b.tests.map((t) => [t.name, 2 * t.runs])));
  const compare = [
    ['Rerun witness seeds only', 0, b.witness.length, m.witnessCost, 1 - pi, 'No fresh seeds, so no evidence beyond the failures you already saw'],
    ['Next nightly comes back clean', 0, m.runs, m.nightly.cost, m.nightlyRisk, 'Same seed counts as tonight on the affected tests'],
    ['Two clean nightlies', 0, 2 * m.runs, x2.cost, C.residualRisk(x2.escape, pi), ''],
    ['This plan', 1, m.seeds, m.plan.cost, m.planRisk, m.plan.exact ? 'Exact minimum cost (branch and bound)' : `Greedy, within ${pct(m.plan.gap)} of optimal`],
  ];

  $('#stage').innerHTML = `
  <div class="card">
    <div class="ch"><div style="min-width:0"><div class="lbl">${esc(b.ip)} · ${b.tests.length} test${b.tests.length > 1 ? 's' : ''} · ${kindBadge(b.kind)} · ${stBadge(b)}</div>
      <div class="sig" style="font-size:14px;max-width:none;-webkit-line-clamp:3">${esc(b.raw)}</div></div>
      <div class="toolbar"><a class="btn ghost" href="#/${state.lastTab || 'confirm'}">← Back</a></div></div>
    <div class="big">
      <div>Failing tonight<b>${m.fails}<span style="font-size:14px;color:var(--mut)"> / ${fmtInt(m.runs)}</span></b></div>
      <div>Rate per seed<b>${pct(m.rate, m.rate < 0.01 ? 2 : 1)}</b></div>
      <div>Seen in<b>${b.seen}<span style="font-size:14px;color:var(--mut)"> / ${b.nights} nightlies</span></b></div>
      <div>First seen<b style="font-size:20px;padding-top:6px">${day(b.firstSeen)}</b></div>
      <div class="o">Risk after a clean nightly<b>${pct(m.nightlyRisk)}</b></div>
    </div>
    <div class="lbl" style="margin-top:12px">Last 30 nightlies</div>${histCells(b.history)}
  </div>
  <div class="grid2b">
    <div class="card">
      <div class="ct">Evidence</div><div class="cs">Pooled over the last ${state.data.config.W} nightlies. Only runs that passed or failed with this signature count; runs that died on another bug or on infrastructure are censored.</div>
      <div class="tblwrap"><table class="tbl" style="margin-top:12px"><thead><tr><th>Test</th><th class="num">Tonight</th><th class="num">Pooled</th><th class="num">Rate · 90% CrI</th><th class="num">Runtime</th></tr></thead><tbody>
      ${m.items.map((it) => { const t = b.tests.find((x) => x.name === it.id); return `<tr><td class="mono" style="font-size:12px;word-break:break-all">${esc(it.id)}</td><td class="num">${t.fails}/${t.runs}</td><td class="num">${t.pooledFails}/${fmtInt(t.pooledRuns)}</td>
        <td class="num">${pct(C.posteriorMean(it), 2)}<div class="sub">${pct(C.betaQuantile(0.05, it.a, it.b), 2)} to ${pct(C.betaQuantile(0.95, it.a, it.b), 2)}</div></td><td class="num">${t.rt >= 60 ? (t.rt / 60).toFixed(1) + ' min' : t.rt.toFixed(0) + ' s'}</td></tr>`; }).join('')}
      </tbody></table></div>
      <div class="lbl" style="margin-top:16px">Witness seeds (rerun with waves)</div>
      <div class="code">${b.witness.map((w) => esc(w.length > 48 ? w.slice(0, 48) + '…' : w)).join('<br>')}</div>
      <div class="lbl" style="margin-top:16px">Failing runs per nightly, last ${b.history.length}</div>
      ${(() => { const mx = Math.max(1, ...b.history.map((f) => f || 0)); return `<div class="bars" style="grid-template-columns:repeat(${b.history.length},minmax(0,1fr));gap:3px;height:96px;margin-top:6px">${b.history.map((f, i) => `<div class="col" style="border-radius:3px 3px 0 0" data-tip="nightly −${b.history.length - 1 - i}: ${f == null ? 'tests not run' : f + ' failing'}">${f ? `<i style="height:${(100 * f) / mx}%;background:${i === b.history.length - 1 ? '#fd6420' : '#c2410c'}"></i>` : ''}</div>`).join('')}</div>`; })()}
      <p class="note" style="margin-top:12px"><b>Model.</b> Seeds are independent draws at one revision, so if the fix doesn't work each fresh seed of test t fails with probability p<sub>t</sub> ~ Beta(a, b). A clean plan of m seeds has probability ∏ B(a, b+m)/B(a, b); Bayes with the fix prior gives the residual risk.</p>
    </div>
    <div class="card">
      <div class="ch"><div><div class="ct">Confirmation plan</div><div class="cs">Cheapest fresh-seed reruns at the fix revision that leave at most ${pct(delta, 0)} chance the bug is still there, given a ${pct(pi, 1)} prior that the fix works.</div></div></div>
      <div style="margin-top:12px">${controls()}</div>
      ${m.verdict === 'directed' ? `<p class="note" style="margin-top:12px;color:var(--warn)"><b style="color:var(--warn)">Reruns are the wrong tool here.</b> Reaching ${pct(delta, 0)} needs ${fmtInt(m.seeds)} seeds (${hrs(m.plan.cost)} CPU-h), more than rerunning the whole ${esc(b.ip)} regression (${hrs(b.ipCpu)} CPU-h). Turn a witness seed into a directed test or a formal property instead.</p>` : ''}
      <table class="tbl" style="margin-top:12px"><thead><tr><th>Test</th><th class="num">Fresh seeds</th><th class="num">CPU-h</th><th class="num">Share of evidence</th></tr></thead><tbody>
        ${planRows.map((r) => { const g = -C.logEscapeTerm(r.it.a, r.it.b, r.k); const tot = -Math.log(m.plan.escape) || 1; return `<tr><td class="mono" style="font-size:12px">${esc(r.it.id)}</td><td class="num">${fmtInt(r.k)}</td><td class="num">${hrs(r.k * r.it.cost)}</td><td class="num">${pct(g / tot, 0)}</td></tr>`; }).join('')}
      </tbody></table>
      <table class="tbl cmp" style="margin-top:14px"><thead><tr><th>Evidence if it all comes back clean</th><th class="num">Sims</th><th class="num">CPU-h</th><th class="num">Residual risk</th></tr></thead><tbody>
        ${compare.map(([l, me, n, c, r, tip]) => `<tr class="${me ? 'me' : ''}" ${tip ? `data-tip="${esc(tip)}"` : ''}><td>${l}</td><td class="num">${fmtInt(n)}</td><td class="num">${hrs(c)}</td><td class="num"><span class="risk ${riskCls(r, delta)}">${pct(r)}</span></td></tr>`).join('')}
      </tbody></table>
      ${frontierChart(b, m)}
      <div class="actions" id="acts">
        ${st === 'confirming' ? `<button class="btn" id="ranClean">Plan ran clean</button><button class="btn ghost" id="reproduced">Failure reproduced</button>` :
          `<button class="btn" id="approve" ${m.verdict === 'directed' ? 'disabled' : ''}>${st === 'confirmed' ? 'Re-run confirmation' : 'Approve plan'} <span class="kbd" style="color:#fff;border-color:rgba(255,255,255,.4)">a</span></button>`}
        <button class="btn ghost" id="copy">Copy dvsim commands</button>
        <button class="btn ghost" id="json">Export JSON</button>
        <span class="note">${rec ? `${STATUS_LABEL[st]} · ${new Date(rec.at).toLocaleString()}${rec.risk != null ? ` · residual risk ${pct(rec.risk)}` : ''}` : ''}</span>
      </div>
      <div class="lbl" style="margin-top:16px">Commands</div>
      <div class="code" id="cmds">${cmds.map((c) => (c.startsWith('#') ? `<span class="kw">${esc(c)}</span>` : esc(c.length > 160 ? c.slice(0, 160) + '…' : c))).join('<br>')}</div>
    </div>
  </div>`;
  wireControls(() => renderBin(b));
  const setStatus = (s, extra = {}) => { state.statuses[b.id] = { status: s, at: Date.now(), ...extra }; save(); renderBin(b); };
  const ap = $('#approve'); if (ap) ap.onclick = () => { setStatus('confirming', { plan: m.plan.plan, target: delta, pi }); toast(`Queued ${fmtInt(m.seeds)} seeds for ${b.ip}`); };
  const rc = $('#ranClean'); if (rc) rc.onclick = () => { setStatus('confirmed', { risk: m.planRisk }); toast(`Confirmed: ${pct(m.planRisk)} residual risk`); };
  const rp = $('#reproduced'); if (rp) rp.onclick = () => { setStatus('reopened'); toast('Reopened: send the failing seed back to RCA'); };
  $('#copy').onclick = async () => { try { await navigator.clipboard.writeText(cmds.join('\n')); toast('Copied dvsim commands'); } catch { toast('Clipboard unavailable: select the commands below'); } };
  $('#json').onclick = () => { download(`plan-${b.id.replace(':', '-')}.json`, JSON.stringify({ bin: b.id, ip: b.ip, signature: b.raw, target_residual_risk: delta, fix_prior: pi,
    plan: m.plan.plan, cpu_hours: m.plan.cost / 3600, escape: m.plan.escape, residual_risk: m.planRisk, exact: m.plan.exact, witness_seeds: b.witness, posterior: m.items.map(({ id, a, b: bb }) => ({ test: id, a, b: bb })) }, null, 2), 'application/json'); toast('Exported plan JSON'); };
}

function frontierChart(b, m) {
  const { pi, delta } = state.settings;
  const fr = C.optimalPlan(m.items, C.requiredEscape(Math.min(delta / 5, 0.002), pi)).frontier;
  const pts = fr.filter((p, i) => i % Math.max(1, Math.floor(fr.length / 300)) === 0 || i === fr.length - 1).filter((p) => p.cost > 0).map((p) => ({ x: p.cost / 3600, y: C.residualRisk(p.escape, pi) }));
  if (pts.length < 2) return '';
  const W = 560, H = 190, L = 44, R = 10, T = 10, B = 24;
  const xs = [...pts.map((p) => p.x), m.nightly.cost / 3600].filter((x) => x > 0);
  const x0 = Math.log10(Math.min(...xs)), x1 = Math.log10(Math.max(...xs) * 1.2);
  const lx = (x) => L + ((Math.log10(x) - x0) / (x1 - x0 || 1)) * (W - L - R);
  const ymax = Math.max(1 - pi, m.nightlyRisk) * 1.05;
  const ly = (y) => T + (1 - y / ymax) * (H - T - B);
  const path = pts.map((p, i) => `${i ? 'L' : 'M'}${lx(p.x).toFixed(1)},${ly(p.y).toFixed(1)}`).join('');
  const ticks = []; for (let e = Math.ceil(x0); e <= Math.floor(x1); e++) ticks.push(10 ** e);
  return `<div class="lbl" style="margin-top:16px">Residual risk vs CPU-hours (every point on the curve is an optimal plan)</div>
  <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Risk versus cost frontier">
    <g class="grid">${[0, 0.25, 0.5].filter((v) => v <= ymax).map((v) => `<line x1="${L}" x2="${W - R}" y1="${ly(v)}" y2="${ly(v)}"/><text x="${L - 6}" y="${ly(v) + 3}" text-anchor="end">${v * 100}%</text>`).join('')}
    ${ticks.map((t) => `<text x="${lx(t)}" y="${H - 6}" text-anchor="middle">${t >= 1 ? t : t.toPrecision(1)} h</text>`).join('')}</g>
    <path d="${path} L${lx(pts[pts.length - 1].x)},${ly(0)} L${lx(pts[0].x)},${ly(0)}Z" fill="rgba(253,100,32,.12)"/>
    <path d="${path}" fill="none" stroke="#fd6420" stroke-width="2.4"/>
    <line x1="${L}" x2="${W - R}" y1="${ly(delta)}" y2="${ly(delta)}" stroke="#ff9a66" stroke-dasharray="4 4"/><text x="${W - R}" y="${ly(delta) - 5}" text-anchor="end" style="fill:#ff9a66">target ${pct(delta, 0)}</text>
    <circle cx="${lx(m.nightly.cost / 3600)}" cy="${ly(m.nightlyRisk)}" r="5" fill="#f1efeb" data-tip="<b>clean nightly</b><br>${hrs(m.nightly.cost)} CPU-h · ${pct(m.nightlyRisk)} risk"/>
    <text x="${lx(m.nightly.cost / 3600) + 8}" y="${ly(m.nightlyRisk) + 4}" style="fill:#f1efeb">nightly</text>
    ${m.plan.cost > 0 ? `<circle cx="${lx(m.plan.cost / 3600)}" cy="${ly(m.planRisk)}" r="5.5" fill="#fd6420" stroke="#0d1117" stroke-width="2" data-tip="<b>this plan</b><br>${hrs(m.plan.cost)} CPU-h · ${pct(m.planRisk)} risk"/>` : ''}
  </svg>`;
}

// ------------------------------------------------------------------ backtest
function renderBacktest() {
  const bt = state.data.backtest, cfg = state.data.config;
  const mo = bt.model, bl = bt.baselines;
  const cal = mo.calibration;
  const W = 420, H = 260, L = 40, B = 30, T = 10, R = 10;
  const sx = (x) => L + x * (W - L - R), sy = (y) => T + (1 - y) * (H - T - B);
  $('#stage').innerHTML = `
  <div class="tiles">
    <div class="tile"><div class="k"><span>Vanishing events</span><span>2024</span></div><b>${fmtInt(bt.episodes.all)}</b><span class="t">a signature fired, then the next nightly ran its tests and it did not</span></div>
    <div class="tile"><div class="k"><span>Came back</span><span>held-out</span></div><b>${pct(bt.episodes.cameBackTest, 0)}</b><span class="t">within ${cfg.horizon} nightlies (${fmtInt(bt.episodes.test)} events from ${day(cfg.split)})</span></div>
    <div class="tile"><div class="k"><span>Ranking (AUC)</span></div><b>${mo.auc.toFixed(2)}</b><span class="t">model vs 0.50 for “vanished means fixed” and for a constant rate</span></div>
    <div class="tile"><div class="k"><span>Brier score</span><span>lower is better</span></div><b>${mo.brier.toFixed(3)}</b><span class="t">vs ${bl.trainBaseRate.brier.toFixed(3)} base rate and ${bl.vanishedMeansFixed.brier.toFixed(3)} for trusting one clean nightly</span></div>
  </div>
  <div class="grid2b">
    <div class="card">
      <div class="ct">Calibration on held-out nightlies</div><div class="cs">Predicted vs observed chance a vanished signature returns within ${cfg.horizon} nightlies. Parameters fitted on nightlies before ${day(cfg.split)}.</div>
      <svg class="chart" viewBox="0 0 ${W} ${H}" style="margin-top:12px" role="img" aria-label="Calibration plot">
        <g class="grid">${[0, 0.25, 0.5, 0.75, 1].map((v) => `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${L - 6}" y="${sy(v) + 3}" text-anchor="end">${v * 100}%</text><text x="${sx(v)}" y="${H - 10}" text-anchor="middle">${v * 100}%</text>`).join('')}</g>
        <line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="rgba(244,244,244,.25)" stroke-dasharray="4 4"/>
        ${cal.map((c) => `<rect x="${sx(c.lo) + 3}" width="${sx(c.hi) - sx(c.lo) - 6}" y="${sy(c.observed)}" height="${sy(0) - sy(c.observed)}" fill="rgba(253,100,32,.18)" rx="4" data-tip="predicted ${pct(c.predicted)} · observed ${pct(c.observed)} · n=${c.n}"/>
          <circle cx="${sx(c.predicted)}" cy="${sy(c.observed)}" r="${3 + Math.sqrt(c.n) / 4}" fill="#fd6420" stroke="#0d1117" data-tip="predicted ${pct(c.predicted)} · observed ${pct(c.observed)} · n=${c.n}"/>`).join('')}
      </svg>
      <p class="note" style="margin-top:8px">Low-prediction bins come back more often than predicted: some vanishings that look like fixes are a bug going quiet under a later change. That is why the plan runs at the fix revision, where seeds really are independent draws.</p>
    </div>
    <div class="card">
      <div class="ct">Against the alternatives</div><div class="cs">${fmtInt(mo.n)} held-out vanishing events, ${fmtInt(mo.positives)} came back. Posterior window W=${mo.W} nightlies and fix prior π=${mo.pi} were chosen on the training split, with a re-emergence rate ρ=${mo.rho} for signatures that return from a later change.</div>
      <table class="tbl" style="margin-top:12px"><thead><tr><th>Predictor</th><th class="num">Brier</th><th class="num">Log loss</th><th class="num">AUC</th></tr></thead><tbody>
        <tr class="cmp me"><td style="color:var(--tx);font-weight:600">Beta posterior + clean-run evidence (this)</td><td class="num">${mo.brier.toFixed(3)}</td><td class="num">${mo.logLoss.toFixed(3)}</td><td class="num">${mo.auc.toFixed(3)}</td></tr>
        <tr><td>Same model, last nightly only (W=1)</td><td class="num">${bl.singleNightly.brier.toFixed(3)}</td><td class="num">${bl.singleNightly.logLoss.toFixed(3)}</td><td class="num">${bl.singleNightly.auc.toFixed(3)}</td></tr>
        <tr><td>Constant training base rate (${pct(bl.trainBaseRate.p, 0)})</td><td class="num">${bl.trainBaseRate.brier.toFixed(3)}</td><td class="num">${bl.trainBaseRate.logLoss.toFixed(3)}</td><td class="num">0.500</td></tr>
        <tr><td>Vanished means fixed</td><td class="num">${bl.vanishedMeansFixed.brier.toFixed(3)}</td><td class="num">${bl.vanishedMeansFixed.logLoss > 10 ? '∞' : bl.vanishedMeansFixed.logLoss.toFixed(3)}</td><td class="num">0.500</td></tr>
      </tbody></table>
      <div class="lbl" style="margin-top:16px">Came back, by how surprising the clean nightly was</div>
      <table class="tbl"><thead><tr><th>Chance an unfixed bug stays quiet for that nightly</th><th class="num">Events</th><th class="num">Came back</th></tr></thead><tbody>
        ${bt.bands.map((x) => `<tr><td>${pct(x.lo, 0)} to ${pct(x.hi, 0)}</td><td class="num">${fmtInt(x.n)}</td><td class="num">${pct(x.cameBack, 0)}</td></tr>`).join('')}
      </tbody></table>
    </div>
  </div>
  <div class="card" style="padding-bottom:8px">
    <div class="ch" style="margin-bottom:10px"><div><div class="ct">Held-out events</div><div class="cs">First 400 of ${fmtInt(bt.episodes.test)} events, newest IPs first in file order.</div></div>
      <div class="toolbar">${[['all', 'All'], ['1', 'Came back'], ['0', 'Stayed gone']].map(([k, l]) => `<button class="chip ${(state.btf || 'all') === k ? 'on' : ''}" data-btf="${k}">${l}</button>`).join('')}</div></div>
    <div id="bttbl"></div>
  </div>`;
  document.querySelectorAll('[data-btf]').forEach((c) => { c.onclick = () => { state.btf = c.dataset.btf; renderBacktest(); }; });
  const rows = bt.sample.filter((e) => !state.btf || state.btf === 'all' || String(e.y) === state.btf).slice(0, 120);
  $('#bttbl').innerHTML = rows.length ? `<table class="tbl"><thead><tr><th>Vanished on</th><th>IP</th><th>Signature</th><th class="num">Quiet-if-unfixed</th><th class="num">Predicted return</th><th>Outcome</th></tr></thead><tbody>
    ${rows.map((e) => `<tr><td class="mono" style="font-size:12px">${day(e.day)}</td><td>${esc(e.ip)}</td><td><div class="sig">${esc(e.sig)}</div></td><td class="num">${pct(e.escape)}</td><td class="num">${pct(e.p)}</td><td>${e.y ? '<span class="st reopened"><i></i>Came back</span>' : '<span class="st confirmed"><i></i>Stayed gone</span>'}</td></tr>`).join('')}</tbody></table>`
    : `<div class="empty"><b>No events</b></div>`;
}

boot();
