import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as C from '../src/confirm.js';
import { loadHistory, backtest, currentRegression, summarize, planFor, CONFIG } from '../scripts/build.mjs';

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b}`);

test('lgamma matches known values', () => {
  close(C.lgamma(0.5), Math.log(Math.sqrt(Math.PI)), 1e-12);
  close(C.lgamma(10), Math.log(362880), 1e-10);
  close(C.lgamma(1), 0, 1e-12);
  close(C.lgamma(123.4), 469.3360974421906, 1e-9); // Python math.lgamma(123.4)
});

test('escape term equals the Beta expectation of (1-p)^m (numerical integral)', () => {
  for (const [a, b, m] of [[1.5, 40.5, 30], [0.5, 300.5, 200], [3.5, 10.5, 5], [2, 2, 1]]) {
    const N = 200000; let s = 0;
    const lb = C.lbeta(a, b);
    for (let i = 0; i < N; i++) { // substitute p = u^2 to remove the p^(a-1) singularity
      const u = (i + 0.5) / N, p = u * u;
      s += 2 * u * Math.exp((a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p) - lb) * (1 - p) ** m;
    }
    close(Math.exp(C.logEscapeTerm(a, b, m)), s / N, 2e-3, `a=${a} b=${b} m=${m}`);
  }
  assert.equal(C.logEscapeTerm(2, 3, 0), 0);
});

test('property: log-escape is convex and decreasing in m (random posteriors)', () => {
  const r = C.rng(7);
  for (let k = 0; k < 300; k++) {
    const a = 0.5 + r() * 20, b = 0.5 + r() * 2000;
    let prev = 0, prevDrop = Infinity;
    for (let m = 1; m < 60; m++) {
      const g = C.logEscapeTerm(a, b, m);
      const drop = prev - g;
      assert.ok(drop > 0, 'decreasing');
      assert.ok(drop <= prevDrop + 1e-12, `convex a=${a} b=${b} m=${m}`);
      prev = g; prevDrop = drop;
    }
  }
});

test('beta quantiles: closed forms and symmetry', () => {
  close(C.betaQuantile(0.5, 1, 1), 0.5, 1e-9);
  close(C.betaQuantile(0.9, 1, 3), 1 - Math.pow(0.1, 1 / 3), 1e-9); // Beta(1,b): 1-(1-q)^(1/b)
  close(C.betaQuantile(0.05, 4.5, 30.5) + C.betaQuantile(0.95, 30.5, 4.5), 1, 1e-9);
  close(C.ibeta(0.3, 2, 5), 0.579825, 1e-6); // scipy.special.betainc(2,5,0.3)
});

test('requiredEscape inverts residualRisk', () => {
  const r = C.rng(11);
  for (let k = 0; k < 200; k++) {
    const pi = 0.05 + 0.9 * r(), delta = 0.01 + 0.2 * r();
    const e = C.requiredEscape(delta, pi);
    if (e < 1) close(C.residualRisk(e, pi), delta, 1e-12);
  }
  assert.equal(C.requiredEscape(0.05, 1), 1);
});

function randomItems(r, n) {
  // observed bins always have >= 1 failure, so a >= 1.5 under the Jeffreys prior
  return Array.from({ length: n }, (_, i) => ({ id: 't' + i, a: 1.5 + Math.floor(r() * 5), b: 0.5 + Math.floor(5 + r() * 60), cost: 1 + Math.floor(r() * 9) }));
}

test('property: every greedy prefix is optimal for the risk it reaches (brute force)', () => {
  const r = C.rng(3);
  for (let k = 0; k < 60; k++) {
    const items = randomItems(r, 1 + Math.floor(r() * 3));
    const g = C.optimalPlan(items, 0.02);
    for (const pt of g.frontier.filter((_, i) => i % 3 === 0).slice(1, 8)) {
      const bf = C.bruteForcePlan(items, pt.escape * (1 + 1e-9), 300, pt);
      assert.ok(bf, 'brute force found a plan');
      close(bf.cost, pt.cost, 1e-9, 'prefix cost equals optimum');
    }
  }
});

test('property: solvePlan is exactly optimal and the dual bound is valid', () => {
  const r = C.rng(5);
  for (let k = 0; k < 80; k++) {
    const items = randomItems(r, 1 + Math.floor(r() * 3));
    const target = [0.2, 0.05, 0.01][k % 3];
    const s = C.solvePlan(items, target);
    const bf = C.bruteForcePlan(items, target, 300, C.optimalPlan(items, target));
    assert.ok(s.exact && bf);
    close(s.cost, bf.cost, 1e-9, 'exact cost');
    assert.ok(s.escape <= target * (1 + 1e-9));
    const g = C.optimalPlan(items, target);
    assert.ok(g.lowerBound <= bf.cost + 1e-9 && bf.cost <= g.cost + 1e-9, 'lower bound <= optimum <= greedy');
  }
});

test('simulation: clean plans meet the residual-risk target when p follows the posterior', () => {
  const r = C.rng(42);
  const items = [{ id: 'a', a: 2.5, b: 180.5, cost: 60 }, { id: 'b', a: 1.5, b: 400.5, cost: 20 }];
  const pi = 0.563, delta = 0.05;
  const plan = C.solvePlan(items, C.requiredEscape(delta, pi)).plan;
  const betaSample = (a, b) => { // via two gammas (Marsaglia-Tsang)
    const gam = (k) => { if (k < 1) return gam(k + 1) * r() ** (1 / k); const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) { let x, v; do { const u1 = r(), u2 = r(); x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); v = 1 + c * x; } while (v <= 0);
        v = v ** 3; const u = r(); if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v; } };
    const x = gam(a), y = gam(b); return x / (x + y);
  };
  let clean = 0, cleanBroken = 0;
  for (let k = 0; k < 20000; k++) {
    const fixed = r() < pi;
    let failed = false;
    if (!fixed) for (const it of items) { const p = betaSample(it.a, it.b); for (let s = 0; s < plan[it.id] && !failed; s++) if (r() < p) failed = true; }
    if (!failed) { clean++; if (!fixed) cleanBroken++; }
  }
  assert.ok(cleanBroken / clean <= delta + 0.006, `realised residual risk ${cleanBroken / clean}`);
  assert.ok(cleanBroken / clean >= delta - 0.02, 'not wildly conservative');
});

test('signatures: line-number churn collapses, wrappers are stripped, infra is dropped', () => {
  const a = C.normalizeSignature('UVM_ERROR (uart_scoreboard.sv:442) [scoreboard] Check failed item.d_data[i] == * Interrupt: TxEmpty');
  const b = C.normalizeSignature('Exit reason: Error: User command failed UVM_ERROR (uart_scoreboard.sv:450) [scoreboard] Check failed item.d_data[i] == * Interrupt: TxEmpty');
  assert.equal(a, b);
  assert.equal(C.normalizeSignature('UVM_ERROR (i2c_scoreboard.sv:812) item 17 uncompared'), 'UVM_ERROR (i2c_scoreboard.sv) item * uncompared');
  assert.ok(C.isInfra('launch_task.returncode != *, err: lswatcher.go:*] Failed to connect to Watcher service'));
  assert.ok(C.isInfra('Job killed most likely because its dependent job failed.'));
  assert.equal(C.classify('UVM_FATAL (uvm_phase.svh:1512) [PH_TIMEOUT] Explicit timeout of * ps hit'), 'timeout');
  const bins = C.binsOf({ buckets: [
    { sig: 'UVM_ERROR (x.sv:1) boom', tests: { t1: 2 }, witness: ['1.t1.9'] },
    { sig: 'UVM_ERROR (x.sv:7) boom', tests: { t1: 1, t2: 1 } },
    { sig: 'Job killed most likely because its dependent job failed.', tests: { t3: 9 } }] });
  assert.deepEqual(Object.keys(bins), ['UVM_ERROR (x.sv) boom']);
  assert.deepEqual(bins['UVM_ERROR (x.sv) boom'].tests, { t1: 3, t2: 1 });
});

test('exposure: runs that died on another bug or on infrastructure are censored', () => {
  const night = { tests: { t: { runs: 50, pass: 40 } }, bins: { s: { tests: { t: 3 } }, other: { tests: { t: 7 } } } };
  assert.equal(C.exposure(night, 't', 's'), 43);
  assert.equal(C.exposure(night, 't', 'absent'), 40);
  const pooled = C.pooledCounts([night, { tests: { t: { runs: 10, pass: 10 } }, bins: {} }], 's', ['t'], 8);
  assert.deepEqual(pooled.t, { fails: 3, runs: 53 });
});

test('rerunning the nightly: rare bins keep large residual risk, common bins do not', () => {
  const rare = [{ id: 't', ...C.posterior(1, 300), cost: 100 }];
  const common = [{ id: 't', ...C.posterior(15, 50), cost: 100 }];
  const rr = C.residualRisk(C.rerunNightly(rare, { t: 300 }).escape, 0.563);
  const rc = C.residualRisk(C.rerunNightly(common, { t: 50 }).escape, 0.563);
  assert.ok(rr > 0.2, `rare ${rr}`);
  assert.ok(rc < 1e-4, `common ${rc}`);
  // and the confirmation plan for the common bin is far cheaper than a 50-seed rerun
  assert.ok(C.solvePlan(common, C.requiredEscape(0.05, 0.563)).cost < 50 * 100 / 4);
});

test('metrics: AUC, Brier, log loss and calibration bins', () => {
  close(C.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75, 1e-12);
  close(C.auc([0.5, 0.5], [0, 1]), 0.5, 1e-12);
  close(C.brier([0.5, 1], [1, 1]), 0.125, 1e-12);
  close(C.logLoss([0.5], [1]), Math.log(2), 1e-12);
  const cb = C.calibrationBins([0.1, 0.15, 0.9], [0, 1, 1]);
  assert.equal(cb[0].n, 2); close(cb[0].observed, 0.5, 1e-12);
});

const raw = JSON.parse(fs.readFileSync(new URL('../data/opentitan_nightlies.json', import.meta.url)));
const results = JSON.parse(fs.readFileSync(new URL('../data/results.json', import.meta.url)));
const app = JSON.parse(fs.readFileSync(new URL('../data/app.json', import.meta.url)));
const byIp = loadHistory(raw);

test('headline: committed results reproduce from the committed OpenTitan data', () => {
  const bt = backtest(byIp);
  assert.equal(bt.episodes.test, results.backtest.episodes.test);
  close(bt.episodes.cameBackTest, results.backtest.episodes.cameBackTest, 1e-12, 'came-back share');
  assert.ok(bt.episodes.cameBackTest > 0.7, 'most vanished signatures came back');
  close(bt.model.brier, results.backtest.model.brier, 1e-12);
  assert.ok(bt.model.brier < bt.baselines.trainBaseRate.brier, 'model beats the base rate on held-out nightlies');
  assert.ok(bt.model.logLoss < bt.baselines.trainBaseRate.logLoss);
  assert.ok(bt.model.auc > 0.6);
  const cur = currentRegression(byIp, bt.model.W);
  const sum = summarize(cur, bt.model.W);
  assert.equal(sum.bins, results.current.bins);
  assert.equal(sum.unsafeAfterCleanNightly, results.current.unsafeAfterCleanNightly);
  assert.equal(sum.exactCount, sum.bins, 'every plan solved to optimality');
});

test('app data matches results and every plan meets its target', () => {
  assert.equal(app.regression.bins.length, results.current.bins);
  const W = app.config.W;
  for (const b of app.regression.bins) {
    const p = planFor(b, W, CONFIG.delta, CONFIG.rcaPrior);
    assert.ok(p.optRisk <= CONFIG.delta + 1e-9, b.id);
    for (const t of b.tests) assert.ok(t.pooledFails <= t.pooledRuns && t.fails <= t.runs);
  }
});
