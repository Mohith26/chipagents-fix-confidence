// Builds data/app.json (what the page loads) and data/results.json (headline + backtest)
// from data/opentitan_nightlies.json (parsed public OpenTitan nightly DV reports).
// usage: node scripts/build.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as C from '../src/confirm.js';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
export const CONFIG = {
  horizon: 10, // nightlies to watch for a comeback
  windows: [1, 4, 8, 16], // pooled-history window candidates (chosen on the train split)
  split: '2024.07.01', // train: vanishings before this nightly, test: on/after
  piGrid: [0.001, 0.002, 0.005, ...Array.from({ length: 99 }, (_, i) => (i + 1) / 100)],
  rcaPrior: 0.563, // ChipAgents RCA top-1 accuracy (chipagents.ai/blogs/chipagents-rca, Nov 2025)
  delta: 0.05, // default residual-risk target
  historyShown: 30,
};

export function loadHistory(raw) {
  const byIp = {};
  for (const [ip, dates] of Object.entries(raw)) {
    const nights = [];
    for (const [stamp, r] of Object.entries(dates)) {
      if (r.error || !r.tests || !Object.keys(r.tests).length) continue;
      nights.push({ day: stamp.slice(0, 10), stamp, rev: r.rev, sim: r.sim, tests: r.tests, bins: C.binsOf(r) });
    }
    nights.sort((a, b) => (a.stamp < b.stamp ? -1 : 1));
    // one report per day (keep the last)
    const dedup = [];
    for (const n of nights) { if (dedup.length && dedup[dedup.length - 1].day === n.day) dedup.pop(); dedup.push(n); }
    byIp[ip] = dedup;
  }
  return byIp;
}

export function runtimesOf(night) { return Object.fromEntries(Object.entries(night.tests).map(([t, v]) => [t, v.rt])); }
export function runsOf(night, tests) { return Object.fromEntries(tests.map((t) => [t, night.tests[t]?.runs || 0])); }

/** Every bin that fired at nightly r and was absent at r+1 although its tests ran. */
export function episodes(byIp, K) {
  const eps = [];
  for (const [ip, H] of Object.entries(byIp)) {
    for (let r = 0; r + 1 + K < H.length; r++) {
      const now = H[r], next = H[r + 1];
      for (const [sig, bin] of Object.entries(now.bins)) {
        if (next.bins[sig]) continue;
        const tests = Object.keys(bin.tests);
        const cleanRuns = Object.fromEntries(tests.map((t) => [t, C.exposure(next, t, sig)]));
        if (!Object.values(cleanRuns).some((x) => x > 0)) continue;
        const futureRuns = Object.fromEntries(tests.map((t) => [t, 0]));
        let back = 0;
        for (let j = r + 2; j <= r + 1 + K; j++) {
          for (const t of tests) futureRuns[t] += H[j].tests[t]?.pass || 0; // passes only: no outcome leakage
          if (H[j].bins[sig]) back = 1;
        }
        eps.push({ ip, r, day: next.day, sig, tests, cleanRuns, futureRuns, y: back });
      }
    }
  }
  return eps;
}

function featurize(byIp, eps, W) {
  return eps.map((e) => {
    const H = byIp[e.ip];
    const counts = C.pooledCounts(H.slice(0, e.r + 1), e.sig, e.tests, W);
    const items = C.itemsFor(counts, runtimesOf(H[e.r]));
    return { ...e, items };
  });
}

function predict(feats, pi, rho) { return feats.map((f) => C.reappearProb(f.items, f.cleanRuns, f.futureRuns, pi, rho).p); }
export const RHO_GRID = Array.from({ length: 20 }, (_, i) => i / 20);

export function backtest(byIp, cfg = CONFIG) {
  const eps = episodes(byIp, cfg.horizon);
  const train = eps.filter((e) => e.day < cfg.split), test = eps.filter((e) => e.day >= cfg.split);
  const ytr = train.map((e) => e.y), yte = test.map((e) => e.y);
  const fits = [];
  for (const W of cfg.windows) {
    const ftr = featurize(byIp, train, W);
    let best = null;
    for (const pi of cfg.piGrid) for (const rho of RHO_GRID) {
      const ll = C.logLoss(predict(ftr, pi, rho), ytr);
      if (!best || ll < best.ll) best = { pi, rho, ll };
    }
    fits.push({ W, pi: best.pi, rho: best.rho, trainLogLoss: best.ll });
  }
  const chosen = fits.reduce((a, b) => (b.trainLogLoss < a.trainLogLoss ? b : a));
  const fte = featurize(byIp, test, chosen.W);
  const pte = predict(fte, chosen.pi, chosen.rho);
  const base = ytr.reduce((s, y) => s + y, 0) / ytr.length;
  const single = fits.find((f) => f.W === 1);
  const pSingle = predict(featurize(byIp, test, 1), single.pi, single.rho);
  // escape of the clean nightly alone (no fit): how surprising was the vanishing?
  const escOnly = fte.map((f) => C.escapeProb(f.items, f.cleanRuns));
  const model = {
    W: chosen.W, pi: chosen.pi, rho: chosen.rho, n: test.length, positives: yte.reduce((s, y) => s + y, 0),
    brier: C.brier(pte, yte), logLoss: C.logLoss(pte, yte), auc: C.auc(pte, yte),
    calibration: C.calibrationBins(pte, yte),
  };
  const baselines = {
    vanishedMeansFixed: { brier: C.brier(pte.map(() => 0), yte), logLoss: C.logLoss(pte.map(() => 0), yte), auc: 0.5 },
    trainBaseRate: { p: base, brier: C.brier(pte.map(() => base), yte), logLoss: C.logLoss(pte.map(() => base), yte), auc: 0.5 },
    singleNightly: { pi: single.pi, rho: single.rho, brier: C.brier(pSingle, yte), logLoss: C.logLoss(pSingle, yte), auc: C.auc(pSingle, yte) },
    escapeOnlyRanking: { auc: C.auc(escOnly, yte) },
  };
  // how often a "one clean nightly" verdict would have been wrong, by escape band
  const bands = [[0, 0.05], [0.05, 0.25], [0.25, 0.5], [0.5, 1.0001]].map(([lo, hi]) => {
    const sel = fte.map((f, i) => [escOnly[i], yte[i]]).filter(([e]) => e >= lo && e < hi);
    return { lo, hi: Math.min(1, hi), n: sel.length, cameBack: sel.length ? sel.reduce((s, x) => s + x[1], 0) / sel.length : null };
  });
  const sample = fte.slice(0, 400).map((f, i) => ({ ip: f.ip, day: f.day, sig: f.sig, tests: f.tests.length,
    escape: +escOnly[i].toFixed(4), p: +pte[i].toFixed(4), y: f.y }));
  return {
    episodes: { all: eps.length, train: train.length, test: test.length,
      cameBackAll: eps.reduce((s, e) => s + e.y, 0) / eps.length,
      cameBackTest: yte.reduce((s, y) => s + y, 0) / yte.length },
    fits, model, baselines, bands, sample,
  };
}

function ipCost(night) { return Object.values(night.tests).reduce((s, t) => s + t.runs * t.rt, 0); }

export function currentRegression(byIp, W, cfg = CONFIG) {
  // the last nightly day present for every IP
  const lastDay = Object.values(byIp).map((H) => H[H.length - 1].day).sort().pop();
  const bins = [];
  const ips = [];
  for (const [ip, H] of Object.entries(byIp)) {
    const night = H[H.length - 1];
    if (night.day !== lastDay) continue;
    const runtimes = runtimesOf(night);
    const totalRuns = Object.values(night.tests).reduce((s, t) => s + t.runs, 0);
    const failing = Object.values(night.tests).reduce((s, t) => s + (t.runs - t.pass), 0);
    ips.push({ ip, rev: night.rev, sim: night.sim, tests: Object.keys(night.tests).length, runs: totalRuns, failing,
      cpu: Math.round(ipCost(night)), bins: Object.keys(night.bins).length });
    for (const [sig, bin] of Object.entries(night.bins)) {
      const tests = Object.keys(bin.tests).filter((t) => night.tests[t]);
      if (!tests.length) continue;
      const counts = C.pooledCounts(H, sig, tests, W);
      const shown = H.slice(-cfg.historyShown).map((n) => {
        const ran = tests.some((t) => n.tests[t]?.runs);
        if (!ran) return null;
        const f = n.bins[sig] ? Object.values(n.bins[sig].tests).reduce((s, x) => s + x, 0) : 0;
        return f;
      });
      let first = null;
      for (const n of H) if (n.bins[sig]) { first = n.day; break; }
      const streak = (() => { let k = 0; for (let i = H.length - 1; i >= 0 && H[i].bins[sig]; i--) k++; return k; })();
      const seen = H.filter((n) => n.bins[sig]).length;
      bins.push({
        id: `${ip}:${hash(sig)}`, ip, sig, raw: bin.raw, kind: C.classify(bin.raw),
        tests: tests.map((t) => ({ name: t, fails: bin.tests[t], runs: night.tests[t].runs, rt: runtimes[t],
          pooledFails: counts[t].fails, pooledRuns: counts[t].runs })),
        witness: bin.witness, firstSeen: first, streak, seen, nights: H.length, history: shown,
        ipCpu: Math.round(ipCost(night)),
      });
    }
  }
  return { day: lastDay, ips: ips.sort((a, b) => b.bins - a.bins), bins };
}

function hash(s) { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return (h >>> 0).toString(36).slice(0, 6); }

/** Plan summary for one bin, same code path the page uses. */
export function planFor(bin, W, delta, pi) {
  const items = bin.tests.map((t) => ({ id: t.name, ...C.posterior(t.pooledFails, t.pooledRuns), cost: Math.max(1, t.rt) }));
  const nightly = C.rerunNightly(items, Object.fromEntries(bin.tests.map((t) => [t.name, t.runs])));
  const target = C.requiredEscape(delta, pi);
  const opt = C.solvePlan(items, target);
  return { items, nightly, target, opt, nightlyRisk: C.residualRisk(nightly.escape, pi), optRisk: C.residualRisk(opt.escape, pi) };
}

export function summarize(cur, W, cfg = CONFIG) {
  const rows = cur.bins.map((b) => ({ b, p: planFor(b, W, cfg.delta, cfg.rcaPrior) }));
  const unsafe = rows.filter((r) => r.p.nightlyRisk > cfg.delta);
  const overkill = rows.filter((r) => r.p.nightlyRisk <= cfg.delta);
  const optCpu = rows.reduce((s, r) => s + r.p.opt.cost, 0);
  const nightlyCpu = rows.reduce((s, r) => s + r.p.nightly.cost, 0);
  const worst = rows.reduce((a, r) => (r.p.nightlyRisk > a.p.nightlyRisk ? r : a), rows[0]);
  const verdict = (r) => (r.p.nightlyRisk <= cfg.delta ? 'nightly' : !r.p.opt.feasible || r.p.opt.cost > r.b.ipCpu ? 'directed' : 'targeted');
  const by = (v) => rows.filter((r) => verdict(r) === v);
  const cpuH = (rs, k) => rs.reduce((s, r) => s + r.p[k].cost, 0) / 3600;
  return {
    bins: rows.length, ipsWithBins: new Set(rows.map((r) => r.b.ip)).size,
    unsafeAfterCleanNightly: unsafe.length,
    medianNightlyRisk: median(rows.map((r) => r.p.nightlyRisk)),
    optimalCpuHours: optCpu / 3600, nightlyAffectedCpuHours: nightlyCpu / 3600,
    worst: { id: worst.b.id, ip: worst.b.ip, nightlyRisk: worst.p.nightlyRisk, optSeeds: Object.values(worst.p.opt.plan).reduce((s, x) => s + x, 0) },
    maxGap: Math.max(...rows.map((r) => r.p.opt.gap)), exactCount: rows.filter((r) => r.p.opt.exact).length,
    overkillCpuSaved: overkill.reduce((s, r) => s + Math.max(0, r.p.nightly.cost - r.p.opt.cost), 0) / 3600,
    overkillBins: overkill.length,
    verdicts: { nightly: by('nightly').length, targeted: by('targeted').length, directed: by('directed').length },
    nightlyEnough: { planCpuHours: cpuH(by('nightly'), 'opt'), nightlyCpuHours: cpuH(by('nightly'), 'nightly') },
    targeted: { planCpuHours: cpuH(by('targeted'), 'opt'), nightlyCpuHours: cpuH(by('targeted'), 'nightly'),
      medianNightlyRisk: median(by('targeted').map((r) => r.p.nightlyRisk)) },
  };
}
/** One rerun batch for every bin that can be confirmed by reruns (shared tests counted once). */
export function batchFor(cur, W, cfg = CONFIG) {
  const rows = cur.bins.map((b) => ({ b, p: planFor(b, W, cfg.delta, cfg.rcaPrior) })).filter((r) => r.p.opt.feasible && r.p.opt.cost <= r.b.ipCpu);
  const bins = rows.map((r) => ({ id: r.b.id, target: r.p.target, items: r.p.items.map((it) => ({ ...it, id: `${r.b.ip}/${it.id}` })) }));
  const bp = C.batchPlan(bins);
  const separate = rows.reduce((s, r) => s + r.p.opt.cost, 0);
  const merged = {}, cost = {};
  for (const r of rows) for (const it of r.p.items) { const k = `${r.b.ip}/${it.id}`; merged[k] = Math.max(merged[k] || 0, r.p.opt.plan[it.id] || 0); cost[k] = it.cost; }
  const mergedCost = Object.entries(merged).reduce((s, [k, v]) => s + v * cost[k], 0);
  return { bins: rows.length, tests: Object.keys(bp.plan).length, cpuHours: bp.cost / 3600, lowerBoundHours: bp.lowerBound / 3600, gap: bp.gap,
    separateHours: separate / 3600, maxMergedHours: mergedCost / 3600, feasible: bp.feasible,
    savedVsSeparate: 1 - bp.cost / separate };
}

function median(xs) { const s = [...xs].sort((a, b) => a - b); const n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null; }

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const t0 = Date.now();
  const raw = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/opentitan_nightlies.json'), 'utf8'));
  const byIp = loadHistory(raw);
  const bt = backtest(byIp);
  const cur = currentRegression(byIp, bt.model.W);
  const sum = summarize(cur, bt.model.W);
  const app = { source: 'reports.opentitan.org nightly DV reports (lowRISC, Apache-2.0)', config: { ...CONFIG, W: bt.model.W, fittedPi: bt.model.pi },
    regression: cur, backtest: { ...bt, sample: bt.sample } };
  fs.writeFileSync(path.join(ROOT, 'data/app.json'), JSON.stringify(app));
  const results = { generated: 'node scripts/build.mjs', config: app.config, backtest: { ...bt, sample: undefined }, current: { day: cur.day, ...sum }, batch: batchFor(cur, bt.model.W) };
  fs.writeFileSync(path.join(ROOT, 'data/results.json'), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
  console.log('built in', Date.now() - t0, 'ms; app.json', fs.statSync(path.join(ROOT, 'data/app.json')).size, 'bytes');
}
