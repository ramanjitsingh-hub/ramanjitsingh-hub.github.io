// Minimal SVG plotting helpers shared by investigation pages. No dependencies.
const C = { ink: "#1D1E20", graphite: "#5B5E63", faint: "#8A8C8F", rule: "#DCD8CF", stone: "#EBE8E1",
  lav: "#9D93C8", lavD: "#6F64A8", powder: "#9FBCD6", powderD: "#6F95BA", sage: "#A3B59B", sageD: "#6E8565" };
const NS = "http://www.w3.org/2000/svg";
function mk(svg, n, a, t) { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); if (t != null) e.textContent = t; svg.append(e); return e; }
function txt(svg, x, y, t, o = {}) { return mk(svg, "text", Object.assign({ x, y, "font-size": 11, fill: C.faint, "font-family": "Geist Mono" }, o), t); }
const poly = pts => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
// axes with optional log scales; returns {sx, sy}
function axes(svg, { W, H, L, R, T, B, x0, x1, y0, y1, xt, yt, xl, yl, xlog, ylog, xfmt, yfmt }) {
  const fx = xlog ? Math.log10 : v => v, fy = ylog ? Math.log10 : v => v;
  const sx = x => L + (fx(x) - fx(x0)) / (fx(x1) - fx(x0)) * (W - L - R);
  const sy = y => T + (1 - (fy(y) - fy(y0)) / (fy(y1) - fy(y0))) * (H - T - B);
  yt.forEach(v => { mk(svg, "line", { x1: L, x2: W - R, y1: sy(v), y2: sy(v), stroke: C.stone }); txt(svg, L - 8, sy(v) + 4, yfmt ? yfmt(v) : v, { "text-anchor": "end" }); });
  xt.forEach(v => txt(svg, sx(v), H - B + 18, xfmt ? xfmt(v) : v, { "text-anchor": "middle" }));
  mk(svg, "line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: C.rule });
  if (xl) txt(svg, W - R, H - 4, xl, { "text-anchor": "end" });
  if (yl) txt(svg, L, T - 10, yl);
  return { sx, sy };
}
const sup = e => { const m = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" }; return String(e).split("").map(c => m[c] || c).join(""); };
const pow10 = v => "10" + sup(Math.round(Math.log10(v)));
