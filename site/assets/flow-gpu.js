// flow-gpu.js  WebGPU port of systems study S1's solver for the homepage hero.
//
// Same discretisation as src/ns2d.cu: MAC staggered grid, doubly periodic, conservative central
// advection, 5-point viscous term, projection by geometric multigrid V(2,2) with red-black
// Gauss-Seidel. Differences, stated on the page: FP32, Heun (RK2) instead of SSP-RK3, a fixed two
// V-cycles per projection warm-started from the previous pressure, linear drag, and a passive dye
// advected semi-Lagrangian for display.
//
// export: startGpuFlow(canvas, {onStats}) -> Promise<boolean>  (false if WebGPU is unavailable)

const NX = 256, NY = 128, H = 1 / NY, DT = 0.003, NU = 2e-4, DAMP = 0.25;

const COMMON = /* wgsl */`
struct U { nx:u32, ny:u32, color:u32, pad:u32, h:f32, h2:f32, ih2:f32, nu:f32, dt:f32, a:f32, b:f32, damp:f32 };
@group(0) @binding(0) var<uniform> P : U;
fn wr(i:i32, n:i32) -> i32 { return (i + n) % n; }
fn ix(i:i32, j:i32) -> u32 { return u32(wr(j, i32(P.ny)) * i32(P.nx) + wr(i, i32(P.nx))); }
`;

const SH = {
  rhs: COMMON + `
@group(0) @binding(1) var<storage, read> u : array<f32>;
@group(0) @binding(2) var<storage, read> v : array<f32>;
@group(0) @binding(3) var<storage, read_write> ru : array<f32>;
@group(0) @binding(4) var<storage, read_write> rv : array<f32>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = i32(g.x); let j = i32(g.y); if (g.x >= P.nx || g.y >= P.ny) { return; }
  let ih = 1.0 / P.h; let k = ix(i, j);
  let uc = u[k]; let uE = u[ix(i+1,j)]; let uW = u[ix(i-1,j)]; let uN = u[ix(i,j+1)]; let uS = u[ix(i,j-1)];
  let vc = v[k]; let vE = v[ix(i+1,j)]; let vW = v[ix(i-1,j)]; let vN = v[ix(i,j+1)]; let vS = v[ix(i,j-1)];
  let a = 0.5*(uc+uE); let b = 0.5*(uW+uc);
  let uvT = 0.5*(uc+uN) * 0.5*(v[ix(i-1,j+1)] + vN);
  let uvB = 0.5*(uS+uc) * 0.5*(vW + vc);
  ru[k] = -((a*a - b*b) + (uvT - uvB)) * ih + P.nu * P.ih2 * (uE+uW+uN+uS-4.0*uc) - P.damp * uc;
  let c = 0.5*(vc+vN); let d = 0.5*(vS+vc);
  let uvR = 0.5*(u[ix(i+1,j-1)] + uE) * 0.5*(vc + vE);
  rv[k] = -((uvR - uvB) + (c*c - d*d)) * ih + P.nu * P.ih2 * (vE+vW+vN+vS-4.0*vc) - P.damp * vc;
}`,
  stage: COMMON + `
@group(0) @binding(1) var<storage, read> u0 : array<f32>;
@group(0) @binding(2) var<storage, read_write> uk : array<f32>;
@group(0) @binding(3) var<storage, read> ru : array<f32>;
@group(0) @binding(4) var<storage, read> v0 : array<f32>;
@group(0) @binding(5) var<storage, read_write> vk : array<f32>;
@group(0) @binding(6) var<storage, read> rv : array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  let k = g.x; if (k >= P.nx * P.ny) { return; }
  uk[k] = P.a * u0[k] + P.b * (uk[k] + P.dt * ru[k]);
  vk[k] = P.a * v0[k] + P.b * (vk[k] + P.dt * rv[k]);
}`,
  div: COMMON + `
@group(0) @binding(1) var<storage, read> u : array<f32>;
@group(0) @binding(2) var<storage, read> v : array<f32>;
@group(0) @binding(3) var<storage, read_write> f : array<f32>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = i32(g.x); let j = i32(g.y); if (g.x >= P.nx || g.y >= P.ny) { return; }
  f[ix(i,j)] = (u[ix(i+1,j)] - u[ix(i,j)] + v[ix(i,j+1)] - v[ix(i,j)]) / P.h;
}`,
  correct: COMMON + `
@group(0) @binding(1) var<storage, read_write> u : array<f32>;
@group(0) @binding(2) var<storage, read_write> v : array<f32>;
@group(0) @binding(3) var<storage, read> p : array<f32>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = i32(g.x); let j = i32(g.y); if (g.x >= P.nx || g.y >= P.ny) { return; }
  let pc = p[ix(i,j)];
  u[ix(i,j)] -= (pc - p[ix(i-1,j)]) / P.h;
  v[ix(i,j)] -= (pc - p[ix(i,j-1)]) / P.h;
}`,
  smooth: COMMON + `
@group(0) @binding(1) var<storage, read_write> p : array<f32>;
@group(0) @binding(2) var<storage, read> f : array<f32>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = i32(g.x); let j = i32(g.y); if (g.x >= P.nx || g.y >= P.ny) { return; }
  if (u32((i + j) & 1) != P.color) { return; }
  p[ix(i,j)] = 0.25 * (p[ix(i-1,j)] + p[ix(i+1,j)] + p[ix(i,j-1)] + p[ix(i,j+1)] - P.h2 * f[ix(i,j)]);
}`,
  resid: COMMON + `
@group(0) @binding(1) var<storage, read> p : array<f32>;
@group(0) @binding(2) var<storage, read> f : array<f32>;
@group(0) @binding(3) var<storage, read_write> r : array<f32>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = i32(g.x); let j = i32(g.y); if (g.x >= P.nx || g.y >= P.ny) { return; }
  r[ix(i,j)] = f[ix(i,j)] - (p[ix(i-1,j)] + p[ix(i+1,j)] + p[ix(i,j-1)] + p[ix(i,j+1)] - 4.0 * p[ix(i,j)]) * P.ih2;
}`,
  // P describes the COARSE level here
  restrict: COMMON + `
@group(0) @binding(1) var<storage, read> r : array<f32>;
@group(0) @binding(2) var<storage, read_write> fc : array<f32>;
@group(0) @binding(3) var<storage, read_write> pc : array<f32>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let nf = 2u * P.nx; let i = 2u * g.x; let j = 2u * g.y;
  fc[g.y * P.nx + g.x] = 0.25 * (r[j*nf+i] + r[j*nf+i+1u] + r[(j+1u)*nf+i] + r[(j+1u)*nf+i+1u]);
  pc[g.y * P.nx + g.x] = 0.0;
}`,
  // P describes the FINE level here
  prolong: COMMON + `
@group(0) @binding(1) var<storage, read> ec : array<f32>;
@group(0) @binding(2) var<storage, read_write> p : array<f32>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = i32(g.x); let j = i32(g.y); if (g.x >= P.nx || g.y >= P.ny) { return; }
  let ncx = i32(P.nx) / 2; let ncy = i32(P.ny) / 2; let I = i / 2; let J = j / 2;
  let I2 = wr(I + select(-1, 1, (i & 1) == 1), ncx); let J2 = wr(J + select(-1, 1, (j & 1) == 1), ncy);
  let e = 0.5625 * ec[J*ncx+I] + 0.1875 * (ec[J*ncx+I2] + ec[J2*ncx+I]) + 0.0625 * ec[J2*ncx+I2];
  p[ix(i,j)] += e;
}`,
  coarse: COMMON + `
@group(0) @binding(1) var<storage, read_write> p : array<f32>;
@group(0) @binding(2) var<storage, read> f : array<f32>;
var<workgroup> sp : array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) t : u32) {
  let n = P.nx * P.ny; let act = t < n;
  let i = i32(t % P.nx); let j = i32(t / P.nx);
  var ff = 0.0; if (act) { ff = f[t]; sp[t] = 0.0; }
  workgroupBarrier();
  for (var s = 0; s < 40; s++) {
    for (var c = 0u; c < 2u; c++) {
      if (act && u32((i + j) & 1) == c) {
        sp[t] = 0.25 * (sp[ix(i-1,j)] + sp[ix(i+1,j)] + sp[ix(i,j-1)] + sp[ix(i,j+1)] - P.h2 * ff);
      }
      workgroupBarrier();
    }
  }
  if (act) { p[t] = sp[t]; }
}`,
  splat: COMMON + `
struct Sp { s : array<vec4<f32>, 8> };   // per splat: (x, y, fx, fy) then (radius, dye channel, amount, 0)
@group(0) @binding(1) var<uniform> S : Sp;
@group(0) @binding(2) var<storage, read_write> u : array<f32>;
@group(0) @binding(3) var<storage, read_write> v : array<f32>;
@group(0) @binding(4) var<storage, read_write> dye : array<vec4<f32>>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.nx || g.y >= P.ny) { return; }
  let k = g.y * P.nx + g.x; let x = f32(g.x); let y = f32(g.y);
  for (var q = 0; q < 4; q++) {
    let a = S.s[2*q]; let b = S.s[2*q+1]; if (b.x <= 0.0) { continue; }
    var dx = abs(x - a.x); dx = min(dx, f32(P.nx) - dx);
    var dy = abs(y - a.y); dy = min(dy, f32(P.ny) - dy);
    let w = exp(-(dx*dx + dy*dy) / (b.x*b.x));
    u[k] += a.z * w; v[k] += a.w * w;
    var d = dye[k]; let ch = u32(b.y);
    if (ch == 0u) { d.x = min(1.6, d.x + b.z * w); } else if (ch == 1u) { d.y = min(1.6, d.y + b.z * w); } else { d.z = min(1.6, d.z + b.z * w); }
    dye[k] = d;
  }
}`,
  dye: COMMON + `
@group(0) @binding(1) var<storage, read> u : array<f32>;
@group(0) @binding(2) var<storage, read> v : array<f32>;
@group(0) @binding(3) var<storage, read> d0 : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> d1 : array<vec4<f32>>;
fn samp(x:f32, y:f32) -> vec4<f32> {
  let fx = x - 0.5; let fy = y - 0.5; let i0 = i32(floor(fx)); let j0 = i32(floor(fy));
  let s = fx - floor(fx); let t = fy - floor(fy);
  return mix(mix(d0[ix(i0,j0)], d0[ix(i0+1,j0)], s), mix(d0[ix(i0,j0+1)], d0[ix(i0+1,j0+1)], s), t);
}
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = i32(g.x); let j = i32(g.y); if (g.x >= P.nx || g.y >= P.ny) { return; }
  let uc = 0.5 * (u[ix(i,j)] + u[ix(i+1,j)]); let vc = 0.5 * (v[ix(i,j)] + v[ix(i,j+1)]);
  let x = f32(i) + 0.5 - P.dt * uc / P.h; let y = f32(j) + 0.5 - P.dt * vc / P.h;
  d1[ix(i,j)] = samp(x, y) * 0.9965;
}`,
};

const RENDER = /* wgsl */`
struct R { nx:f32, ny:f32, sx:f32, sy:f32, ox:f32, oy:f32, pad0:f32, pad1:f32 };
@group(0) @binding(0) var<uniform> Rq : R;
@group(0) @binding(1) var<storage, read> dye : array<vec4<f32>>;
struct VO { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi : u32) -> VO {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o : VO; o.pos = vec4(p[vi], 0.0, 1.0); o.uv = vec2(0.5 * (p[vi].x + 1.0), 0.5 * (1.0 - p[vi].y)); return o;
}
fn at(i:i32, j:i32) -> vec4<f32> {
  let nx = i32(Rq.nx); let ny = i32(Rq.ny);
  return dye[u32(((j % ny + ny) % ny) * nx + ((i % nx + nx) % nx))];
}
@fragment fn fs(in : VO) -> @location(0) vec4<f32> {
  // cover-fit the 2:1 domain into the canvas, then bilinear sample cell-centred dye
  let x = (Rq.ox + in.uv.x * Rq.sx) * Rq.nx - 0.5; let y = (Rq.oy + (1.0 - in.uv.y) * Rq.sy) * Rq.ny - 0.5;
  let i0 = i32(floor(x)); let j0 = i32(floor(y)); let s = x - floor(x); let t = y - floor(y);
  let d = mix(mix(at(i0,j0), at(i0+1,j0), s), mix(at(i0,j0+1), at(i0+1,j0+1), s), t);
  var c = vec3(246.0, 244.0, 239.0) / 255.0;
  let cols = array<vec3<f32>, 3>(vec3(157.0,147.0,200.0)/255.0, vec3(159.0,188.0,214.0)/255.0, vec3(163.0,181.0,155.0)/255.0);
  let a = array<f32, 3>(min(1.0, d.x) * 0.78, min(1.0, d.y) * 0.78, min(1.0, d.z) * 0.78);
  for (var k = 0; k < 3; k++) { c = c * (1.0 - a[k]) + cols[k] * a[k]; }
  return vec4(c, 1.0);
}`;

export async function startGpuFlow(canvas, { onStats } = {}) {
  if (!navigator.gpu) return false;
  let adapter, device;
  try { adapter = await navigator.gpu.requestAdapter(); if (!adapter) return false; device = await adapter.requestDevice(); }
  catch { return false; }
  const ctx = canvas.getContext("webgpu"); if (!ctx) return false;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const SB = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const buf = (n, usage = SB) => device.createBuffer({ size: Math.max(16, n * 4), usage });
  const uni = (vals) => {   // U struct: 4 u32 then 8 f32
    const b = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const a = new ArrayBuffer(48), iv = new Uint32Array(a, 0, 4), fv = new Float32Array(a, 16, 8);
    iv.set([vals.nx, vals.ny, vals.color || 0, 0]);
    fv.set([vals.h, vals.h * vals.h, 1 / (vals.h * vals.h), NU, DT, vals.a || 0, vals.b || 0, DAMP]);
    device.queue.writeBuffer(b, 0, a); return b;
  };
  const pipes = {};
  for (const [k, code] of Object.entries(SH)) {
    pipes[k] = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  }
  const bg = (pipe, buffers) => device.createBindGroup({ layout: pipes[pipe].getBindGroupLayout(0),
    entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })) });

  const M = NX * NY;
  const u = buf(M), v = buf(M), u1 = buf(M), v1 = buf(M), ru = buf(M), rv = buf(M);
  const dye0 = buf(4 * M), dye1 = buf(4 * M);
  // multigrid levels down to <= 256 cells
  const lv = []; for (let nx = NX, ny = NY, h = H; ; nx /= 2, ny /= 2, h *= 2) {
    lv.push({ nx, ny, h, p: buf(nx * ny), f: buf(nx * ny), r: buf(nx * ny) }); if (nx * ny <= 256) break; }
  const L = lv.length - 1;
  lv.forEach(l => { l.U = uni(l); l.U0 = uni({ ...l, color: 0 }); l.U1 = uni({ ...l, color: 1 }); });
  const U0 = lv[0].U;
  const Ust1 = uni({ nx: NX, ny: NY, h: H, a: 0, b: 1 }), Ust2 = uni({ nx: NX, ny: NY, h: H, a: 0.5, b: 0.5 });
  const splatBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const ops = [];   // [pipeline, bindgroup, wx, wy]
  const g2 = (nx, ny) => [Math.ceil(nx / 8), Math.ceil(ny / 8)];
  const op = (name, buffers, wx, wy = 1) => ops.push([pipes[name], bg(name, buffers), wx, wy]);
  const vcycle = (l) => {
    const c = lv[l];
    if (l === L) { op("coarse", [c.U, c.p, c.f], 1); return; }
    for (let s = 0; s < 2; s++) { op("smooth", [c.U0, c.p, c.f], ...g2(c.nx, c.ny)); op("smooth", [c.U1, c.p, c.f], ...g2(c.nx, c.ny)); }
    op("resid", [c.U, c.p, c.f, c.r], ...g2(c.nx, c.ny));
    const d = lv[l + 1]; op("restrict", [d.U, c.r, d.f, d.p], ...g2(d.nx, d.ny));
    vcycle(l + 1);
    op("prolong", [c.U, d.p, c.p], ...g2(c.nx, c.ny));
    for (let s = 0; s < 2; s++) { op("smooth", [c.U0, c.p, c.f], ...g2(c.nx, c.ny)); op("smooth", [c.U1, c.p, c.f], ...g2(c.nx, c.ny)); }
  };
  const project = (uu, vv) => {
    op("div", [U0, uu, vv, lv[0].f], ...g2(NX, NY));
    vcycle(0); vcycle(0);
    op("correct", [U0, uu, vv, lv[0].p], ...g2(NX, NY));
  };
  op("splat", [U0, splatBuf, u, v, dye0], ...g2(NX, NY));
  // Heun: u1 = u + dt L(u); P.  u = 1/2 u + 1/2 (u1 + dt L(u1)); P.
  // stage shader computes uk = a*u0 + b*(uk + dt*r) in place, so copy u into u1 first
  ops.push(["copy"]);
  op("rhs", [U0, u, v, ru, rv], ...g2(NX, NY));
  op("stage", [Ust1, u, u1, ru, v, v1, rv], Math.ceil(M / 64));
  project(u1, v1);
  op("rhs", [U0, u1, v1, ru, rv], ...g2(NX, NY));
  op("stage", [Ust2, u, u1, ru, v, v1, rv], Math.ceil(M / 64));
  project(u1, v1);
  ops.push(["swapback"]);           // u <- u1
  op("dye", [U0, u, v, dye0, dye1], ...g2(NX, NY));
  ops.push(["dyeback"]);

  // diagnostics: divergence of the final field, read back asynchronously
  const divOut = buf(M), readBuf = device.createBuffer({ size: M * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const divOp = [pipes.div, bg("div", [U0, u, v, divOut]), ...g2(NX, NY)];
  let reading = false;

  const rpipe = device.createRenderPipeline({ layout: "auto",
    vertex: { module: device.createShaderModule({ code: RENDER }), entryPoint: "vs" },
    fragment: { module: device.createShaderModule({ code: RENDER }), entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" } });
  const rU = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const rbg = device.createBindGroup({ layout: rpipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: rU } }, { binding: 1, resource: { buffer: dye0 } }] });

  let view = { sx: 1, sy: 1, ox: 0, oy: 0 };
  function size() {
    const r = canvas.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, r.width * d); canvas.height = Math.max(1, r.height * d);
    const ca = r.width / Math.max(1, r.height), da = NX / NY;
    view = ca > da ? { sx: 1, sy: da / ca, ox: 0, oy: (1 - da / ca) / 2 } : { sx: ca / da, sy: 1, ox: (1 - ca / da) / 2, oy: 0 };
    device.queue.writeBuffer(rU, 0, new Float32Array([NX, NY, view.sx, view.sy, view.ox, view.oy, 0, 0]));
  }
  size(); addEventListener("resize", size);

  let pointer = null, time = 0, frame = 0, fpsT = performance.now(), fpsN = 0, maxDiv = null;
  const host = canvas.parentElement;
  host.addEventListener("pointermove", e => {
    const r = canvas.getBoundingClientRect();
    const x = (view.ox + (e.clientX - r.left) / r.width * view.sx) * NX, y = (view.oy + (1 - (e.clientY - r.top) / r.height) * view.sy) * NY;
    if (pointer) { pointer.dx = x - pointer.x; pointer.dy = y - pointer.y; } else pointer = { dx: 0, dy: 0 };
    pointer.x = x; pointer.y = y; pointer.fresh = true;
  });
  const em = [{ a: .31, b: .23, ph: 0 }, { a: .19, b: .29, ph: 2.1 }, { a: .27, b: .17, ph: 4.2 }];

  function step() {
    time += 0.016;
    const s = new Float32Array(32);
    em.forEach((e, c) => {
      const x = .70 + .22 * Math.sin(e.a * time + e.ph), y = .45 - .32 * Math.sin(e.b * time * 1.3 + e.ph * .7);
      const dx = .22 * e.a * Math.cos(e.a * time + e.ph), dy = -.32 * e.b * 1.3 * Math.cos(e.b * time * 1.3 + e.ph * .7);
      s.set([x * NX, y * NY, dx * 0.06, dy * 0.06, 5.0, c, 0.07, 0], 8 * c);
    });
    if (pointer && pointer.fresh) { s.set([pointer.x, pointer.y, pointer.dx * 0.012, pointer.dy * 0.012, 6.0, frame % 3, 0.12, 0], 24); pointer.fresh = false; }
    device.queue.writeBuffer(splatBuf, 0, s);
    const enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    for (const o of ops) {
      if (o[0] === "copy" || o[0] === "swapback" || o[0] === "dyeback") {
        pass.end();
        if (o[0] === "copy") { enc.copyBufferToBuffer(u, 0, u1, 0, M * 4); enc.copyBufferToBuffer(v, 0, v1, 0, M * 4); }
        else if (o[0] === "swapback") { enc.copyBufferToBuffer(u1, 0, u, 0, M * 4); enc.copyBufferToBuffer(v1, 0, v, 0, M * 4); }
        else enc.copyBufferToBuffer(dye1, 0, dye0, 0, M * 16);
        pass = enc.beginComputePass(); continue;
      }
      pass.setPipeline(o[0]); pass.setBindGroup(0, o[1]); pass.dispatchWorkgroups(o[2], o[3]);
    }
    const diag = !reading && frame % 30 === 0;
    if (diag) { pass.setPipeline(divOp[0]); pass.setBindGroup(0, divOp[1]); pass.dispatchWorkgroups(divOp[2], divOp[3]); }
    pass.end();
    if (diag) enc.copyBufferToBuffer(divOut, 0, readBuf, 0, M * 4);
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: .965, g: .957, b: .937, a: 1 } }] });
    rp.setPipeline(rpipe); rp.setBindGroup(0, rbg); rp.draw(3); rp.end();
    device.queue.submit([enc.finish()]);
    if (diag) {
      reading = true;
      readBuf.mapAsync(GPUMapMode.READ).then(() => {
        const a = new Float32Array(readBuf.getMappedRange()); let m = 0; for (let k = 0; k < a.length; k++) { const x = Math.abs(a[k]); if (x > m) m = x; }
        readBuf.unmap(); maxDiv = m; reading = false;
      }).catch(() => { reading = false; });
    }
    frame++; fpsN++;
    const now = performance.now();
    if (now - fpsT > 1000) { onStats && onStats({ fps: fpsN * 1000 / (now - fpsT), maxDiv, grid: `${NX} × ${NY}`, kernels: ops.filter(o => typeof o[0] !== "string").length }); fpsT = now; fpsN = 0; }
  }
  let visible = true; new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(canvas);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) { for (let k = 0; k < 200; k++) step(); }
  else (function loop() {
    if (visible && !document.hidden) step();
    else { fpsT = performance.now(); fpsN = 0; }   // do not count paused time in the fps window
    requestAnimationFrame(loop);
  })();
  return true;
}
