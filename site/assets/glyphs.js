// Small computed figures for the homepage margins. Each is generated from its formula at load.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const INK = "#1D1E20", MUTE = "#8A8C8F", RULE = "#C9C5BC", LAV = "#6F64A8", LAVL = "#9D93C8", POW = "#6F95BA";
  const mk = (p, n, a, t) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); if (t != null) e.textContent = t; p.append(e); return e; };
  const path = pts => pts.map((q, i) => (i ? "L" : "M") + q[0].toFixed(2) + "," + q[1].toFixed(2)).join("");
  const svg = (host, w, h) => { const s = document.createElementNS(NS, "svg"); s.setAttribute("viewBox", `0 0 ${w} ${h}`); s.setAttribute("width", "100%"); s.setAttribute("role", "img"); host.append(s); return s; };

  const G = {
    // fixed-point iteration x_{k+1} = cos x_k, cobweb from x0 = 0.1, converging to 0.739085
    cobweb(host) {
      const W = 180, H = 150, s = svg(host, W, H), L = 14, B = 14, S = 124;
      const X = x => L + x * S, Y = y => H - B - y * S;
      mk(s, "line", { x1: X(0), x2: X(1), y1: Y(0), y2: Y(0), stroke: RULE });
      mk(s, "line", { x1: X(0), x2: X(0), y1: Y(0), y2: Y(1), stroke: RULE });
      mk(s, "line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: MUTE, "stroke-dasharray": "2 3" });
      const c = []; for (let i = 0; i <= 60; i++) { const x = i / 60; c.push([X(x), Y(Math.cos(x))]); }
      mk(s, "path", { d: path(c), fill: "none", stroke: INK, "stroke-width": 1.2 });
      let x = 0.1; const w = [[X(x), Y(0)]];
      for (let k = 0; k < 60; k++) { const y = Math.cos(x); if (k < 14) w.push([X(x), Y(y)], [X(y), Y(y)]); x = y; }   // draw 14, iterate to convergence
      mk(s, "path", { d: path(w), fill: "none", stroke: LAVL, "stroke-width": 1 });
      mk(s, "circle", { cx: X(x), cy: Y(x), r: 2.6, fill: LAV });
      return `x<sub>k+1</sub> = cos x<sub>k</sub> &nbsp;→&nbsp; ${x.toFixed(4)}`;
    },
    // prior N(0, 1) and the posterior after 5 observations with noise sd 0.8, mean 0.9
    posterior(host) {
      const W = 180, H = 120, s = svg(host, W, H), L = 6, R = 6, B = 16;
      const n = 5, s2 = 0.64, m = 0.9, pv = 1 / (1 + n / s2), pm = pv * (n * m / s2);
      const pdf = (x, mu, v) => Math.exp(-0.5 * (x - mu) ** 2 / v) / Math.sqrt(2 * Math.PI * v);
      const X = x => L + (x + 3) / 6 * (W - L - R), ymax = pdf(pm, pm, pv) * 1.08, Y = y => H - B - y / ymax * (H - B - 8);
      mk(s, "line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: RULE });
      const curve = (mu, v) => { const p = []; for (let i = 0; i <= 120; i++) { const x = -3 + 6 * i / 120; p.push([X(x), Y(pdf(x, mu, v))]); } return p; };
      mk(s, "path", { d: path(curve(0, 1)), fill: "none", stroke: MUTE, "stroke-width": 1, "stroke-dasharray": "3 3" });
      const post = curve(pm, pv);
      mk(s, "path", { d: path(post) + `L${X(3)},${H - B}L${X(-3)},${H - B}Z`, fill: LAVL, "fill-opacity": 0.18, stroke: "none" });
      mk(s, "path", { d: path(post), fill: "none", stroke: LAV, "stroke-width": 1.4 });
      [0.6, 1.3, 0.4, 1.1, 1.1].forEach(v => mk(s, "line", { x1: X(v), x2: X(v), y1: H - B, y2: H - B - 7, stroke: POW, "stroke-width": 1.4 }));
      return `prior → p(x | data), &nbsp;σ: 1 → ${Math.sqrt(pv).toFixed(2)}`;
    },
    // Lissajous figure x = sin(3t + π/2), y = sin 2t
    lissajous(host) {
      const W = 160, H = 120, s = svg(host, W, H), p = [];
      for (let i = 0; i <= 600; i++) { const t = 2 * Math.PI * i / 600; p.push([W / 2 + 64 * Math.sin(3 * t + Math.PI / 2), H / 2 + 50 * Math.sin(2 * t)]); }
      mk(s, "path", { d: path(p) + "Z", fill: "none", stroke: LAV, "stroke-width": 1.2 });
      return "x = sin(3t + π/2), &nbsp;y = sin 2t";
    },
    // three ventures as a directed path on a time axis
    ventures(host) {
      const W = 120, H = 44, s = svg(host, W, H), pts = [[12, 30], [58, 20], [104, 10]];
      mk(s, "line", { x1: 4, x2: W - 4, y1: 38, y2: 38, stroke: RULE });
      for (let i = 0; i < 2; i++) mk(s, "path", { d: `M${pts[i][0] + 4},${pts[i][1]}L${pts[i + 1][0] - 4},${pts[i + 1][1]}`, stroke: INK, "stroke-width": 1, fill: "none" });
      pts.forEach((q, i) => mk(s, "circle", { cx: q[0], cy: q[1], r: 3.2, fill: i === 2 ? LAV : INK }));
    },
    // a Gaussian density
    gauss(host) {
      const W = 120, H = 44, s = svg(host, W, H), p = [];
      for (let i = 0; i <= 80; i++) { const x = -3 + 6 * i / 80; p.push([6 + (x + 3) / 6 * 108, 38 - 30 * Math.exp(-0.5 * x * x)]); }
      mk(s, "line", { x1: 4, x2: W - 4, y1: 38, y2: 38, stroke: RULE });
      mk(s, "path", { d: path(p), fill: "none", stroke: LAV, "stroke-width": 1.3 });
      mk(s, "line", { x1: 60, x2: 60, y1: 8, y2: 38, stroke: MUTE, "stroke-dasharray": "2 2" });
    },
    // the 5-point Laplacian stencil on a small grid
    stencil(host) {
      const W = 120, H = 44, s = svg(host, W, H), h = 9, ox = 8, oy = 4;
      for (let i = 0; i <= 4; i++) { mk(s, "line", { x1: ox, x2: ox + 4 * h, y1: oy + i * h, y2: oy + i * h, stroke: RULE }); mk(s, "line", { x1: ox + i * h, x2: ox + i * h, y1: oy, y2: oy + 4 * h, stroke: RULE }); }
      [[2, 2, LAV, 3.6], [1, 2, INK, 2.6], [3, 2, INK, 2.6], [2, 1, INK, 2.6], [2, 3, INK, 2.6]].forEach(([i, j, c, r]) => mk(s, "circle", { cx: ox + i * h, cy: oy + j * h, r, fill: c }));
    },
  };
  document.querySelectorAll("[data-glyph]").forEach(host => {
    const f = G[host.dataset.glyph]; if (!f) return;
    const cap = f(host);
    if (cap) { const c = document.createElement("div"); c.className = "gcap"; c.innerHTML = cap; host.append(c); }
  });
})();
