// Shared computation for the homepage comps. Real algorithms, synthetic data (labelled as such).
// 1. A 1-D grid Bayesian filter for cycle level with a contamination mixture (robust model).
// 2. A faithful port of the production FemoraAI update (Kalman + EWMA blend + 3-sigma gate).

const RF = (() => {
  const X = []; for (let x = 14; x <= 70.0001; x += 0.25) X.push(x);
  const N = X.length, dx = 0.25;
  const npdf = (z, m, s) => Math.exp(-0.5 * ((z - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

  function makeKernel(q) {
    const half = Math.ceil(5 * Math.sqrt(q) / dx), k = [];
    let t = 0;
    for (let i = -half; i <= half; i++) { const v = npdf(i * dx, 0, Math.sqrt(q)); k.push(v); t += v; }
    return { k: k.map(v => v / t), half };
  }

  function gridFilter({ prior = 29, priorSd = 5, q = 0.5, s = 2.0, pm = 0.05, pl = 0.04, tau = 10 } = {}) {
    let p = X.map(x => npdf(x, prior, priorSd));
    const norm = a => { const t = a.reduce((u, v) => u + v, 0); return a.map(v => v / t); };
    p = norm(p);
    const { k, half } = makeKernel(q);
    const propagate = a => {
      const out = new Array(N).fill(0);
      for (let i = 0; i < N; i++) {
        let acc = 0;
        for (let j = -half; j <= half; j++) { const ii = Math.min(N - 1, Math.max(0, i + j)); acc += a[ii] * k[j + half]; }
        out[i] = acc;
      }
      return norm(out);
    };
    const exg = (z, m) => { // ex-Gaussian density
      const lam = 1 / tau, a = (m + lam * s * s - z) / (Math.SQRT2 * s);
      return 0.5 * lam * Math.exp(0.5 * lam * (2 * m + lam * s * s - 2 * z)) * erfc(a);
    };
    const lik = z => X.map(x => (1 - pm - pl) * npdf(z, x, s) + pm * npdf(z, 2 * x, Math.SQRT2 * s) + pl * exg(z, x));
    return {
      X, get density() { return p.map(v => v / dx); },
      update(z) { const pr = propagate(p), l = lik(z); p = norm(pr.map((v, i) => v * l[i])); },
      mean() { return p.reduce((u, v, i) => u + v * X[i], 0); },
      quantile(a) { let c = 0; for (let i = 0; i < N; i++) { c += p[i]; if (c >= a) return X[i]; } return X[N - 1]; },
    };
  }

  function erfc(x) { // Numerical Recipes erfcc, fractional error < 1.2e-7
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }

  function production(init = 28) {
    const S = { mu: init, e: init, ev: 9, k: init, P: 9, a: 0.3, q: 0.5, r: 4, w: 0.5 };
    return {
      S,
      observe(z) {
        z = Math.max(15, Math.min(60, z));
        if (Math.abs(z - S.e) / Math.sqrt(Math.max(S.ev, 1e-6)) > 3) return false;
        const prev = S.e; S.e = S.a * z + (1 - S.a) * S.e;
        S.ev = (1 - S.a) * (S.ev + S.a * (z - prev) ** 2);
        const pp = S.P + S.q, K = pp / (pp + S.r); S.k += K * (z - S.k); S.P = (1 - K) * pp;
        S.mu = S.w * S.k + (1 - S.w) * S.e; return true;
      },
      sigma() { return Math.sqrt(S.P + S.ev); },
    };
  }

  // Deterministic synthetic sequence: regular user, one missed log at cycle 6, an 8-day level shift at cycle 11.
  function rng(seed) { return () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }; }
  function synthetic(seed = 7, n = 22) {
    const u = rng(seed), g = () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
    const out = []; let lvl = 28.5;
    for (let c = 0; c < n; c++) {
      lvl += 0.4 * g(); const base = (c >= 11 ? 8 : 0) + lvl;
      let z = Math.round(base + 1.6 * g());
      const missed = c === 6; if (missed) z += Math.round(base + 1.6 * g());
      out.push({ z, level: base, missed });
    }
    return out;
  }

  function riccati(P, q, r) { const m = P + q; return m * r / (m + r); }
  function steady(q, r) { const m = (q + Math.sqrt(q * q + 4 * q * r)) / 2; return { K: m / (m + r), P: m - q }; }

  return { X, gridFilter, production, synthetic, riccati, steady };
})();
