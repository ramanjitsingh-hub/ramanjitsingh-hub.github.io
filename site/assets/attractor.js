// Clifford attractor for the About section.
//   x' = sin(a y) + c cos(a x),   y' = sin(b x) + d cos(b y)
// Iterates are accumulated into a density histogram and tone-mapped on a log scale, so the figure
// is computed in the visitor's browser and builds up over the first few seconds.
(function () {
  const cv = document.getElementById("attractor");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const A = -1.4, B = 1.6, C = 1.0, D = 0.7, TOTAL = 900000, PER_FRAME = 30000;
  const readout = document.getElementById("att-n");
  let W, H, hist, x = 0.1, y = 0.1, n = 0, img, maxc = 1;
  // ivory -> lavender -> deep lavender -> charcoal
  const stops = [[246, 244, 239], [190, 184, 220], [111, 100, 168], [29, 30, 32]];
  function col(t) {
    const s = Math.min(stops.length - 1.0001, t * (stops.length - 1)), i = Math.floor(s), f = s - i;
    return stops[i].map((v, k) => v + (stops[i + 1][k] - v) * f);
  }
  function size() {
    const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    W = Math.max(1, Math.round(r.width * dpr)); H = Math.max(1, Math.round(r.height * dpr));
    cv.width = W; cv.height = H; hist = new Float32Array(W * H); img = ctx.createImageData(W, H);
    n = 0; maxc = 1; x = 0.1; y = 0.1;
  }
  function iterate(k) {
    // the attractor lives in roughly [-2.4, 2.4] x [-1.7, 1.7]; fit with a margin, keep aspect
    const sx = W / 4.5, sy = H / 3.5, s = Math.min(sx, sy), ox = W / 2, oy = H / 2;
    for (let i = 0; i < k; i++) {
      const nx = Math.sin(A * y) + C * Math.cos(A * x), ny = Math.sin(B * x) + D * Math.cos(B * y);
      x = nx; y = ny;
      const px = (ox + x * s) | 0, py = (oy - y * s) | 0;
      if (px >= 0 && px < W && py >= 0 && py < H) { const j = py * W + px; const v = ++hist[j]; if (v > maxc) maxc = v; }
    }
    n += k;
  }
  function draw() {
    const d = img.data, L = Math.log(1 + maxc);
    for (let j = 0; j < hist.length; j++) {
      const t = hist[j] ? Math.pow(Math.log(1 + hist[j]) / L, 0.85) : 0;
      const c = col(t * 0.92), o = 4 * j;
      d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = t ? Math.min(255, 60 + 195 * t) : 0;
    }
    ctx.putImageData(img, 0, 0);
    if (readout) readout.textContent = n.toLocaleString("en-US");
  }
  size();
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) { iterate(TOTAL); draw(); }
  else (function loop() { if (n < TOTAL) { iterate(PER_FRAME); draw(); requestAnimationFrame(loop); } })();
  let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { size(); iterate(TOTAL); draw(); }, 200); });
})();
