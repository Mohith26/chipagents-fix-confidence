// Fix confirmation: how many reruns prove a regression fix, and which ones.
//
// A failure bin (one root cause) fires on test t with an unknown per-seed rate p_t.
// From the regression history we get a Beta(a_t, b_t) posterior for p_t.
// If a proposed fix does NOT work, a plan of m_t fresh seeds per test stays clean with
// probability  e(m) = prod_t E[(1 - p_t)^m_t] = prod_t B(a_t, b_t + m_t) / B(a_t, b_t)
// (the "escape" probability). g_t(m) = log E[(1-p_t)^m] is a cumulant generating function
// in m, hence convex, so the cheapest plan reaching a target escape is found by a greedy
// over marginal (log-escape drop) / (sim cost), and every greedy prefix is exactly optimal
// for the risk it reaches (it minimises cost + mu * log-escape for some mu >= 0).
// A Lagrangian dual bound certifies the gap at the requested target.
//
// Pure functions, no DOM. Used by the app (browser) and the scripts/tests (node).

const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

/** log Gamma(x) for x > 0 (Lanczos, g = 7). */
export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let s = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) s += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(s);
}

export function lbeta(a, b) { return lgamma(a) + lgamma(b) - lgamma(a + b); }

/** log E[(1-p)^m] for p ~ Beta(a, b). Zero at m = 0, convex and decreasing in m. */
export function logEscapeTerm(a, b, m) {
  if (m <= 0) return 0;
  return lgamma(b + m) - lgamma(b) - lgamma(a + b + m) + lgamma(a + b);
}

/** Beta posterior for a per-seed failure rate from pooled counts. Jeffreys prior by default. */
export function posterior(fails, runs, prior = { a: 0.5, b: 0.5 }) {
  if (fails < 0 || runs < fails) throw new Error('bad counts');
  return { a: prior.a + fails, b: prior.b + runs - fails };
}

export function posteriorMean({ a, b }) { return a / (a + b); }

// Regularized incomplete beta I_x(a, b) (continued fraction, Numerical Recipes betacf).
export function ibeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta(a, b));
  const cf = (x, a, b) => {
    const FPMIN = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d;
      const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-14) break;
    }
    return h;
  };
  return x < (a + 1) / (a + b + 2) ? (bt * cf(x, a, b)) / a : 1 - (bt * cf(1 - x, b, a)) / b;
}

/** q-quantile of Beta(a, b) by bisection on I_x(a, b). */
export function betaQuantile(q, a, b) {
  let lo = 0, hi = 1;
  for (let i = 0; i < 80; i++) { const mid = (lo + hi) / 2; if (ibeta(mid, a, b) < q) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}

/** Probability that a plan {test: seeds} stays clean although the bug is still there. */
export function escapeProb(items, plan) {
  let L = 0;
  for (const it of items) L += logEscapeTerm(it.a, it.b, plan[it.id] || 0);
  return Math.exp(L);
}

/**
 * Bayes: probability the bug is still present after a clean plan, given the prior
 * probability pi that the fix works (for example the RCA confidence score).
 */
export function residualRisk(escape, pi) {
  const u = (1 - pi) * escape;
  return u / (pi + u);
}

/** Escape probability a plan must reach so that residualRisk <= delta. */
export function requiredEscape(delta, pi) {
  if (pi >= 1) return 1;
  if (pi <= 0) return 0;
  return Math.min(1, (delta * pi) / ((1 - delta) * (1 - pi)));
}

/**
 * Cheapest plan with escape <= target. items: [{id, a, b, cost}] (cost = CPU seconds per seed).
 * Returns {plan, cost, escape, steps, lowerBound, gap, frontier} where frontier lists every
 * greedy prefix (each one exactly optimal for the escape it reaches).
 */
export function optimalPlan(items, targetEscape, opts = {}) {
  const maxSeeds = opts.maxSeeds ?? 20000;
  const logTarget = Math.log(Math.max(targetEscape, 1e-300));
  const m = Object.fromEntries(items.map((it) => [it.id, 0]));
  const g = Object.fromEntries(items.map((it) => [it.id, 0]));
  let L = 0, cost = 0, total = 0, lastRatio = Infinity;
  const frontier = [{ cost: 0, escape: 1 }];
  if (logTarget >= 0) return { plan: m, cost: 0, escape: 1, lowerBound: 0, gap: 0, frontier, feasible: true };
  while (L > logTarget && total < maxSeeds * items.length) {
    let best = null, bestR = -1, bestG = 0;
    for (const it of items) {
      if (m[it.id] >= maxSeeds) continue;
      const gNext = logEscapeTerm(it.a, it.b, m[it.id] + 1);
      const r = (g[it.id] - gNext) / it.cost;
      if (r > bestR) { bestR = r; best = it; bestG = gNext; }
    }
    if (!best || bestR <= 0) break;
    L += bestG - g[best.id];
    g[best.id] = bestG;
    m[best.id] += 1;
    cost += best.cost;
    total += 1;
    lastRatio = bestR;
    frontier.push({ cost, escape: Math.exp(L) });
  }
  const feasible = L <= logTarget + 1e-12;
  // Lagrangian dual bound at mu = 1 / lastRatio: for any plan x,
  // cost(x) >= min_y [cost(y) + mu (L(y) - logTarget)] when L(x) <= logTarget.
  let lowerBound = 0;
  if (feasible && Number.isFinite(lastRatio)) {
    const mu = 1 / lastRatio;
    let val = -mu * logTarget;
    for (const it of items) val += minCostPlusMuG(it, mu, maxSeeds);
    lowerBound = Math.max(0, val);
  }
  return {
    plan: m, cost, escape: Math.exp(L), feasible, lowerBound,
    gap: feasible && cost > 0 ? (cost - lowerBound) / cost : 0, frontier,
  };
}

// min over integer k of cost*k + mu*g(k); g convex so stop when the increment turns positive.
function minCostPlusMuG(it, mu, maxSeeds) {
  let best = 0, k = 0, gk = 0;
  while (k < maxSeeds) {
    const gn = logEscapeTerm(it.a, it.b, k + 1);
    const inc = it.cost + mu * (gn - gk);
    if (inc >= 0) break;
    best += inc; k += 1; gk = gn;
  }
  return best;
}

/**
 * Exact minimum-cost plan by depth-first branch and bound, seeded with the greedy plan as the
 * incumbent. The last test's seed count is solved in closed form (smallest m reaching the
 * remaining log-escape budget). Gives up after `nodeBudget` nodes and returns null.
 */
export function exactPlan(items, targetEscape, incumbent, nodeBudget = 300000) {
  const logTarget = Math.log(targetEscape);
  if (logTarget >= 0) return { plan: Object.fromEntries(items.map((i) => [i.id, 0])), cost: 0, escape: 1, exact: true };
  const order = [...items].sort((x, y) => y.cost - x.cost); // enumerate expensive tests, solve the cheapest
  const last = order[order.length - 1];
  let best = incumbent ? { plan: { ...incumbent.plan }, cost: incumbent.cost } : null;
  let nodes = 0;
  const minSeeds = (it, budget) => { // smallest m with g(m) <= budget (budget < 0), or Infinity
    if (budget >= 0) return 0;
    let lo = 0, hi = 1;
    while (logEscapeTerm(it.a, it.b, hi) > budget) { hi *= 2; if (hi > 1e7) return Infinity; }
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (logEscapeTerm(it.a, it.b, mid) <= budget) hi = mid; else lo = mid; }
    return hi;
  };
  const rec = (i, plan, L, cost) => {
    if (++nodes > nodeBudget) return false;
    if (best && cost >= best.cost) return true;
    if (i === order.length - 1) {
      const m = minSeeds(last, logTarget - L);
      if (m === Infinity) return true;
      const c = cost + m * last.cost;
      if (!best || c < best.cost - 1e-9) best = { plan: { ...plan, [last.id]: m }, cost: c };
      return true;
    }
    const it = order[i];
    for (let k = 0; ; k++) {
      const c = cost + k * it.cost;
      if (best && c >= best.cost) break;
      plan[it.id] = k;
      if (!rec(i + 1, plan, L + logEscapeTerm(it.a, it.b, k), c)) return false;
      if (L + logEscapeTerm(it.a, it.b, k) <= logTarget) break; // more seeds here only add cost
    }
    delete plan[it.id];
    return true;
  };
  const ok = rec(0, {}, 0, 0);
  if (!ok || !best) return null;
  return { plan: best.plan, cost: best.cost, escape: escapeProb(items, best.plan), exact: true, nodes };
}

/** Greedy + certificate, refined to the exact optimum when the search is small enough. */
export function solvePlan(items, targetEscape, opts = {}) {
  const g = optimalPlan(items, targetEscape, opts);
  if (!g.feasible) return { ...g, exact: false, greedyCost: g.cost };
  if (g.gap < 1e-9) return { ...g, exact: true, greedyCost: g.cost };
  const ex = exactPlan(items, targetEscape, g, opts.nodeBudget);
  if (!ex) return { ...g, exact: false, greedyCost: g.cost };
  return { ...g, plan: ex.plan, cost: ex.cost, escape: ex.escape, exact: true, gap: 0, greedyCost: g.cost };
}

/** Exhaustive search for tiny instances (used in tests to certify the greedy). */
export function bruteForcePlan(items, targetEscape, maxPerTest, upper) {
  const logTarget = Math.log(targetEscape);
  // optional feasible upper bound (cost + 1e-9) only prunes; the optimum is still searched for
  let best = upper ? { cost: upper.cost + 1e-9, plan: null } : null;
  const rec = (i, plan, L, cost) => {
    if (best && cost >= best.cost) return;
    if (i === items.length) {
      if (L <= logTarget + 1e-12) best = { plan: { ...plan }, cost, escape: Math.exp(L) };
      return;
    }
    const it = items[i];
    for (let k = 0; k <= maxPerTest; k++) {
      plan[it.id] = k;
      rec(i + 1, plan, L + logEscapeTerm(it.a, it.b, k), cost + k * it.cost);
    }
    delete plan[it.id];
  };
  rec(0, {}, 0, 0);
  if (best && !best.plan) { // upper bound was optimal: recover it exactly by re-searching at its cost
    return upper.plan ? { plan: upper.plan, cost: upper.cost, escape: escapeProb(items, upper.plan) } : null;
  }
  return best;
}

/**
 * Escape of the plan most teams use today: rerun the same regression once (n_t seeds per
 * test as scheduled). Returns {escape, cost} where cost is the CPU time of all affected tests.
 */
export function rerunNightly(items, nightlyRuns) {
  const plan = {};
  let cost = 0;
  for (const it of items) { plan[it.id] = nightlyRuns[it.id] || 0; cost += plan[it.id] * it.cost; }
  return { plan, cost, escape: escapeProb(items, plan) };
}

// ---------------------------------------------------------------- signatures and bins

const INFRA = /launch_task|Job killed most likely because|Failed to connect|envelope|chubby|returncode != |Job returned non-zero exit code|License|licen[cs]e checkout|No space left|ssh: /i;
const WRAP = /^(Job \S+ killed due to: )?Exit reason: Error: User command failed\s*/;

/** Normalise a dvsim failure-bucket signature so that line-number churn does not split a bin. */
export function normalizeSignature(sig) {
  return sig.replace(WRAP, '')
    .replace(/\(([\w./-]+\.(?:sv|svh|v|vh|cc|c|h|py)):[\d*]+\)/g, '($1)')
    .replace(/(?<![A-Za-z_])(?:0x[0-9a-fA-F]+|\d+)(?![A-Za-z_\d])/g, '*') // numbers, not names like i2c
    .replace(/\s+/g, ' ')
    .trim();
}

export function isInfra(sig) { return INFRA.test(sig); }

export function classify(sig) {
  const s = sig.replace(WRAP, '');
  if (isInfra(sig)) return 'infra';
  if (/PH_TIMEOUT|timeout|timed out|spinwait/i.test(s)) return 'timeout';
  if (/^UVM_FATAL/.test(s)) return 'fatal';
  if (/Offending|Assertion|ASRT/i.test(s)) return 'assertion';
  if (/scoreboard|Check failed|mismatch/i.test(s)) return 'check';
  return 'error';
}

/** Merge raw dvsim buckets of one report into {normSig: {tests:{t:fails}, raw, witness}} (infra dropped). */
export function binsOf(report) {
  const out = {};
  for (const b of report.buckets || []) {
    if (isInfra(b.sig)) continue;
    const k = normalizeSignature(b.sig);
    const o = out[k] || (out[k] = { tests: {}, raw: b.sig.replace(WRAP, ''), witness: [] });
    for (const [t, f] of Object.entries(b.tests)) o.tests[t] = (o.tests[t] || 0) + f;
    for (const w of b.witness || []) if (o.witness.length < 6) o.witness.push(w);
  }
  return out;
}

/**
 * Runs of test t that could have shown bin s: runs that passed plus runs that failed with s.
 * dvsim buckets a failing run by its first error, so a run that died on another bug or on
 * infrastructure says nothing about s (it is censored, not clean).
 */
export function exposure(night, t, sig) {
  const rec = night.tests[t];
  if (!rec) return 0;
  const own = night.bins[sig]?.tests[t] || 0;
  return Math.min(rec.runs, rec.pass + own);
}

/**
 * Pooled counts for one bin over a window of nightlies (most recent last): failures of the
 * bin on each test and the exposed runs (see exposure), including nightlies where the bin
 * did not fire. history: [{tests:{t:{runs,pass}}, bins:{sig:{tests}}}].
 */
export function pooledCounts(history, sig, tests, window) {
  const h = history.slice(Math.max(0, history.length - window));
  const res = {};
  for (const t of tests) res[t] = { fails: 0, runs: 0 };
  for (const night of h) {
    for (const t of tests) {
      const runs = exposure(night, t, sig);
      if (!runs) continue;
      res[t].runs += runs;
      res[t].fails += Math.min(runs, night.bins[sig]?.tests[t] || 0);
    }
  }
  return res;
}

/** Turn pooled counts + runtimes into solver items. */
export function itemsFor(counts, runtimes, prior) {
  return Object.entries(counts)
    .filter(([, c]) => c.runs > 0)
    .map(([id, c]) => ({ id, ...posterior(c.fails, c.runs, prior), cost: Math.max(1, runtimes[id] || 1), fails: c.fails, runs: c.runs }));
}

/**
 * Backtest prediction for a bin that vanished from nightly r+1:
 * P(it comes back within the next K nightlies) =
 *   P(unfixed | clean at r+1) * P(fires in the next K | unfixed)
 * with the posterior updated by the clean nightly.
 */
export function reappearProb(items, cleanRuns, futureRuns, pi, rho = 0) {
  const e = escapeProb(items, cleanRuns);
  const pUnfixed = residualRisk(e, pi);
  const updated = items.map((it) => ({ ...it, b: it.b + (cleanRuns[it.id] || 0) }));
  const pMiss = escapeProb(updated, futureRuns);
  const same = pUnfixed * (1 - pMiss);
  // rho: chance the same signature re-emerges from a later change (new cause, other revision)
  return { escape: e, pUnfixed, pFire: 1 - pMiss, same, p: 1 - (1 - same) * (1 - rho) };
}

// ---------------------------------------------------------------- evaluation helpers

export function brier(ps, ys) { let s = 0; ps.forEach((p, i) => { s += (p - ys[i]) ** 2; }); return s / ps.length; }
export function logLoss(ps, ys) {
  let s = 0;
  ps.forEach((p, i) => { const q = Math.min(1 - 1e-9, Math.max(1e-9, p)); s -= ys[i] ? Math.log(q) : Math.log(1 - q); });
  return s / ps.length;
}
/** AUC by the rank-sum (Mann-Whitney) statistic with tie correction. */
export function auc(ps, ys) {
  const idx = ps.map((p, i) => [p, ys[i]]).sort((x, y) => x[0] - y[0]);
  let rank = 0, sumPos = 0, nPos = 0, nNeg = 0, i = 0;
  while (i < idx.length) {
    let j = i;
    while (j < idx.length && idx[j][0] === idx[i][0]) j++;
    const avg = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) { if (idx[k][1]) { sumPos += avg; nPos++; } else nNeg++; }
    rank = j; i = j;
  }
  if (!nPos || !nNeg) return NaN;
  return (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}
export function calibrationBins(ps, ys, edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]) {
  const out = [];
  for (let i = 0; i < edges.length - 1; i++) {
    const sel = ps.map((p, k) => [p, ys[k]]).filter(([p]) => p >= edges[i] && p < edges[i + 1]);
    if (!sel.length) continue;
    out.push({ lo: edges[i], hi: Math.min(1, edges[i + 1]), n: sel.length,
      predicted: sel.reduce((s, x) => s + x[0], 0) / sel.length,
      observed: sel.reduce((s, x) => s + x[1], 0) / sel.length });
  }
  return out;
}

/** Deterministic PRNG (mulberry32) for tests and simulations. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
