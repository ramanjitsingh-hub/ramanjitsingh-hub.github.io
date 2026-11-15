// Life on a logarithmic time axis: x = 1 - log(T - t) / log(T - t0), T = 2029.
// Recent years carry most of the events, so the log scale spreads them out.
(function () {
  const svg = document.getElementById("lifeline");
  if (!svg) return;
  const NS = "http://www.w3.org/2000/svg", T = 2029, T0 = 2004.0, NOW = 2026.87;
  const EV = [
    [2004.55, "born, Phuket", 0], [2010.5, "first laptop, age 6", 0], [2021.6, "diploma, LPU", 1],
    [2023.75, "Pink Parcel", 0], [2025.0, "FemoraAI", 1], [2026.5, "Luna", 0], [2027.45, "B.Tech", 1, true],
  ];
  const el = (n, a, t) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); if (t != null) e.textContent = t; svg.append(e); return e; };
  function draw() {
    svg.innerHTML = "";
    const W = svg.clientWidth, H = svg.clientHeight, L = 4, R = 4, Y = Math.round(H * 0.52);
    const X = t => L + (1 - Math.log(T - t) / Math.log(T - T0)) * (W - L - R);
    el("line", { x1: L, x2: W - R, y1: Y, y2: Y, stroke: "#1D1E20", "stroke-width": 1 });
    for (let yr = 2004; yr <= 2028; yr++) {
      const major = [2004, 2010, 2015, 2020, 2025, 2027].includes(yr);
      el("line", { x1: X(yr), x2: X(yr), y1: Y, y2: Y + (major ? 7 : 4), stroke: major ? "#5B5E63" : "#C9C5BC", "stroke-width": 1 });
      if (major && W > 560) el("text", { x: X(yr), y: Y + 19, "text-anchor": yr === 2004 ? "start" : "middle", "font-size": 10, fill: "#8A8C8F", "font-family": "Geist Mono" }, yr);
    }
    const xn = X(NOW);
    el("line", { x1: xn, x2: xn, y1: 4, y2: Y + 9, stroke: "#6F64A8", "stroke-width": 1, "stroke-dasharray": "2 3" });
    el("text", { x: xn + 5, y: 11, "font-size": 10, fill: "#6F64A8", "font-family": "Geist Mono" }, "now");
    // labels: preferred row first; a label that would collide with one already placed in that row
    // moves to the other row, and is dropped only if both rows are taken (narrow screens)
    const placed = [[], []];
    EV.forEach(([t, label, pref, future]) => {
      const x = X(t);
      el("circle", { cx: x, cy: Y, r: 3.6, fill: future ? "#F6F4EF" : "#1D1E20", stroke: "#1D1E20", "stroke-width": 1.2 });
      const txt = el("text", { x, y: Y - 9, "text-anchor": t < 2006 ? "start" : "middle", "font-size": 10.5, fill: "#1D1E20", "font-family": "Geist" }, label);
      let b = txt.getBBox();
      const shift = Math.max(0, L - b.x) - Math.max(0, b.x + b.width - (W - R));   // keep inside the frame
      if (shift) { txt.setAttribute("x", x + shift); b = txt.getBBox(); }
      const a = b.x - 4, z = b.x + b.width + 4;
      const fits = r => placed[r].every(([p, q]) => z < p || a > q) && (r === 0 || z < xn - 2 || a > xn + 30);
      const row = [pref, 1 - pref].find(fits);
      if (row == null) { txt.remove(); return; }
      placed[row].push([a, z]);
      if (row) txt.setAttribute("dy", -11);
    });
    el("text", { x: W - R, y: H - 2, "text-anchor": "end", "font-size": 9.5, fill: "#8A8C8F", "font-family": "Geist Mono" }, "log time · x ∝ −log(2029 − t)");
  }
  draw();
  let tm; addEventListener("resize", () => { clearTimeout(tm); tm = setTimeout(draw, 120); });
})();
