// Hero figure. The scalar Kalman filter from FemoraAI's CycleService (q = 0.5, r = 4, x0 = 28, P0 = 9),
// run live on a simulated user drawn from the filter's own model. Each step: predict, draw the 80% range
// for the next cycle, observe, update.
(function () {
  const svg = document.getElementById("hero-kf"); if (!svg) return;
  const NS = "http://www.w3.org/2000/svg", W = 520, H = 330, L = 30, R = 10, T = 16, B = 30, WIN = 36;
  const q = 0.5, r = 4, Z80 = 1.2816;
  let seed = 20250522;
  const u = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return (seed + 0.5) / 4294967296; };
  const g = () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
  let level, x, P, k, hits, hist;
  const set = (id, t) => { const e = document.getElementById(id); if (e) e.textContent = t; };

  function reset() { level = 28 + 2 * g(); x = 28; P = 9; k = 0; hits = 0; hist = []; }
  function step() {
    const M = P + q, sd = Math.sqrt(M + r);                // prior variance and predictive sd of the next cycle
    level += Math.sqrt(q) * g();
    const z = level + Math.sqrt(r) * g();
    const lo = x - Z80 * sd, hi = x + Z80 * sd, hit = z >= lo && z <= hi;
    const K = M / (M + r);
    hist.push({ mu: x, lo, hi, z, hit, level });
    x += K * (z - x); P = (1 - K) * M; k++; hits += hit ? 1 : 0;
    if (hist.length > WIN) hist.shift();
    set("kf-k", k); set("kf-K", K.toFixed(4)); set("kf-cov", (100 * hits / k).toFixed(0) + "%");
    if (k >= 400) reset();
    draw();
  }

  function draw() {
    svg.innerHTML = "";
    const el = (n, a, t) => { const e = document.createElementNS(NS, n); for (const p in a) e.setAttribute(p, a[p]); if (t != null) e.textContent = t; svg.append(e); return e; };
    const vals = hist.flatMap(h => [h.lo, h.hi, h.z]);
    const c = vals.length ? (Math.min(...vals) + Math.max(...vals)) / 2 : 28;
    const ymin = Math.min(c - 9, ...vals) - 1, ymax = Math.max(c + 9, ...vals) + 1;
    const sx = i => L + (i + WIN - hist.length) / (WIN - 1) * (W - L - R);
    const sy = y => T + (ymax - y) / (ymax - ymin) * (H - T - B);
    const t0 = Math.ceil(ymin / 4) * 4;
    for (let y = t0; y <= ymax; y += 4) {
      el("line", { x1: L, x2: W - R, y1: sy(y), y2: sy(y), stroke: "#ECE9E2" });
      el("text", { x: L - 6, y: sy(y) + 3.5, "text-anchor": "end", "font-size": 10, fill: "#8A8C8F", "font-family": "Geist Mono" }, y);
    }
    el("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: "#DCD8CF" });
    el("text", { x: W - R, y: H - 8, "text-anchor": "end", "font-size": 10, fill: "#8A8C8F", "font-family": "Geist Mono" }, "cycle k, days on the vertical axis");
    if (!hist.length) return;
    let up = "", dn = "", mu = "", lv = "";
    hist.forEach((h, i) => { up += (i ? " L" : "M") + sx(i).toFixed(1) + "," + sy(h.hi).toFixed(1); });
    for (let i = hist.length - 1; i >= 0; i--) dn += " L" + sx(i).toFixed(1) + "," + sy(hist[i].lo).toFixed(1);
    el("path", { d: up + dn + " Z", fill: "#9D93C8", "fill-opacity": .16, stroke: "none" });
    hist.forEach((h, i) => { mu += (i ? " L" : "M") + sx(i).toFixed(1) + "," + sy(h.mu).toFixed(1); lv += (i ? " L" : "M") + sx(i).toFixed(1) + "," + sy(h.level).toFixed(1); });
    el("path", { d: lv, fill: "none", stroke: "#A3B59B", "stroke-width": 1, "stroke-dasharray": "2 3" });
    el("path", { d: mu, fill: "none", stroke: "#6F64A8", "stroke-width": 1.6, "stroke-linejoin": "round" });
    hist.forEach((h, i) => {
      const last = i === hist.length - 1;
      el("circle", { cx: sx(i), cy: sy(h.z), r: last ? 3.6 : 2.6, fill: h.hit ? "#9FBCD6" : "#FBFAF7", stroke: h.hit ? "#6F95BA" : "#9A5B5B", "stroke-width": 1.2 });
    });
  }

  reset(); for (let i = 0; i < 24; i++) step();
  let vis = true; new IntersectionObserver(es => { vis = es[0].isIntersecting; }).observe(svg);
  setInterval(() => { if (vis && !document.hidden) step(); }, 900);
})();
