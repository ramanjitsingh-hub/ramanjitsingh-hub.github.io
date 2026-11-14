// ns2d.cu  Incompressible Navier-Stokes on a doubly periodic square, GPU.
//
// Discretisation
//   MAC staggered grid, N x N cells, h = 2*pi/N.
//   u[i,j] at (i h, (j+1/2) h), v[i,j] at ((i+1/2) h, j h), p[i,j] at cell centres.
//   Advection in conservative form d(uu)/dx + d(uv)/dy with central interpolation (2nd order).
//   Viscous term: 5-point Laplacian. Time: SSP-RK3, exact projection after every stage.
//   Pressure Poisson: geometric multigrid V(2,2) cycles, red-black Gauss-Seidel smoother,
//   cell-centred full-weighting restriction, bilinear prolongation.
//
// Modes
//   bw      measure achievable device bandwidth (copy kernel)
//   tg      Taylor-Green vortex to time T, report error against the analytic solution
//   mg      multigrid convergence on a random mean-zero right-hand side
//   bench   time per step with a per-phase breakdown (median of 5 repeats)
//
// Precision: --prec f64 | f32 | mixed (FP64 iterate and residual, FP32 V-cycle corrections)
// Ablation:  --graph --coarse1 --warm --adaptive (0/1), or --baseline 1 for the milestone-1 solver
//
// Usage: ns2d <mode> [--N 256] [--prec f64|f32] [--nu 0.01] [--T 1.0] [--cfl 0.5]
//                    [--tol 1e-10] [--steps 50] [--dump file]
// Output: one JSON object on stdout.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static const double PI = 3.14159265358979323846;

__device__ __forceinline__ int wr(int i, int n) { return i < 0 ? i + n : (i >= n ? i - n : i); }

static dim3 B2(32, 8);
static dim3 G2(int n) { return dim3((n + 31) / 32, (n + 7) / 8); }

// ------------------------------------------------------------------ physics kernels
template <class T>
__global__ void k_rhs(const T* __restrict__ u, const T* __restrict__ v, T* __restrict__ ru, T* __restrict__ rv,
                      int n, T ih, T nu_ih2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= n) return;
  int im = wr(i - 1, n), ip = wr(i + 1, n), jm = wr(j - 1, n), jp = wr(j + 1, n);
  T uc = u[j*n+i];
  T uE = u[j*n+ip], uW = u[j*n+im], uN = u[jp*n+i], uS = u[jm*n+i];
  T vc = v[j*n+i];
  T vE = v[j*n+ip], vW = v[j*n+im], vN = v[jp*n+i], vS = v[jm*n+i];
  // ---- u-momentum at (i, j+1/2)
  T a = T(0.5) * (uc + uE), b = T(0.5) * (uW + uc);
  T duu = (a * a - b * b) * ih;
  // corners (i, j+1) and (i, j): u averaged in y, v averaged in x
  T vNW = v[jp*n+im];
  T uvT = T(0.5) * (uc + uN) * T(0.5) * (vNW + vN);
  T uvB = T(0.5) * (uS + uc) * T(0.5) * (vW + vc);
  T duv = (uvT - uvB) * ih;
  ru[j*n+i] = -(duu + duv) + nu_ih2 * (uE + uW + uN + uS - T(4) * uc);
  // ---- v-momentum at (i+1/2, j)
  T c = T(0.5) * (vc + vN), d = T(0.5) * (vS + vc);
  T dvv = (c * c - d * d) * ih;
  // corners (i+1, j) and (i, j)
  T uSE = u[jm*n+ip];
  T uvR = T(0.5) * (uSE + uE) * T(0.5) * (vc + vE);
  T uvL = T(0.5) * (uS + uc) * T(0.5) * (vW + vc);
  T dvu = (uvR - uvL) * ih;
  rv[j*n+i] = -(dvu + dvv) + nu_ih2 * (vE + vW + vN + vS - T(4) * vc);
}

// out = a*x0 + b*(xk + dt*rk), for both components
template <class T>
__global__ void k_stage(int m, T a, T b, T dt, const T* u0, const T* uk, const T* ru, T* uo,
                        const T* v0, const T* vk, const T* rv, T* vo) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= m) return;
  uo[k] = a * u0[k] + b * (uk[k] + dt * ru[k]);
  vo[k] = a * v0[k] + b * (vk[k] + dt * rv[k]);
}

template <class T>
__global__ void k_div(const T* u, const T* v, T* dv, int n, T ih) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= n) return;
  dv[j*n+i] = (u[j*n+wr(i+1,n)] - u[j*n+i] + v[wr(j+1,n)*n+i] - v[j*n+i]) * ih;
}

template <class T>
__global__ void k_correct(T* u, T* v, const T* phi, int n, T ih) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= n) return;
  T pc = phi[j*n+i];
  u[j*n+i] -= (pc - phi[j*n+wr(i-1,n)]) * ih;
  v[j*n+i] -= (pc - phi[wr(j-1,n)*n+i]) * ih;
}

// ------------------------------------------------------------------ multigrid kernels
template <class T>
__global__ void k_smooth(T* p, const T* f, int n, T h2, int color) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= n || ((i + j) & 1) != color) return;
  T s = p[j*n+wr(i-1,n)] + p[j*n+wr(i+1,n)] + p[wr(j-1,n)*n+i] + p[wr(j+1,n)*n+i];
  p[j*n+i] = T(0.25) * (s - h2 * f[j*n+i]);
}

template <class T>
__global__ void k_resid(const T* p, const T* f, T* r, int n, T ih2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= n) return;
  T lap = (p[j*n+wr(i-1,n)] + p[j*n+wr(i+1,n)] + p[wr(j-1,n)*n+i] + p[wr(j+1,n)*n+i] - T(4) * p[j*n+i]) * ih2;
  r[j*n+i] = f[j*n+i] - lap;
}

template <class T>
__global__ void k_restrict(const T* r, T* fc, int nc) {
  int I = blockIdx.x * blockDim.x + threadIdx.x, J = blockIdx.y * blockDim.y + threadIdx.y;
  if (I >= nc || J >= nc) return;
  int nf = 2 * nc, i = 2 * I, j = 2 * J;
  fc[J*nc+I] = T(0.25) * (r[j*nf+i] + r[j*nf+i+1] + r[(j+1)*nf+i] + r[(j+1)*nf+i+1]);
}

template <class T>
__global__ void k_prolong_add(const T* ec, T* p, int nf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nf || j >= nf) return;
  int nc = nf / 2, I = i >> 1, J = j >> 1;
  int di = (i & 1) ? 1 : -1, dj = (j & 1) ? 1 : -1;
  int I2 = wr(I + di, nc), J2 = wr(J + dj, nc);
  T e = T(0.5625) * ec[J*nc+I] + T(0.1875) * (ec[J*nc+I2] + ec[J2*nc+I]) + T(0.0625) * ec[J2*nc+I2];
  p[j*nf+i] += e;
}

template <class T>
__global__ void k_fill(T* x, int m, T val) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) x[k] = val;
}
template <class T>
__global__ void k_shift(T* x, int m, T val) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < m) x[k] -= val;
}

// reductions: per-block partial sums (double) and max-abs
template <class T>
__global__ void k_reduce(const T* x, int m, double* psum, double* pmax) {
  __shared__ double ss[256], sm[256];
  double s = 0, mx = 0;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < m; k += gridDim.x * blockDim.x) {
    double val = (double)x[k]; s += val; mx = fmax(mx, fabs(val));
  }
  ss[threadIdx.x] = s; sm[threadIdx.x] = mx; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) { ss[threadIdx.x] += ss[threadIdx.x + o]; sm[threadIdx.x] = fmax(sm[threadIdx.x], sm[threadIdx.x + o]); }
    __syncthreads();
  }
  if (threadIdx.x == 0) { psum[blockIdx.x] = ss[0]; pmax[blockIdx.x] = sm[0]; }
}

__global__ void k_copy4(const float4* __restrict__ a, float4* __restrict__ b, size_t m) {
  for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < m; k += (size_t)gridDim.x * blockDim.x) b[k] = a[k];
}

// single-block coarse solve: all red-black sweeps in shared memory, one launch (n*n <= 1024)
template <class T>
__global__ void k_coarse(T* p, const T* f, int n, T h2, int sweeps) {
  __shared__ T sp[1024];
  int t = threadIdx.x, i = t % n, j = t / n; bool act = t < n * n;
  T ff = act ? f[t] : T(0);
  if (act) sp[t] = T(0);
  __syncthreads();
  for (int s = 0; s < sweeps; s++)
    for (int color = 0; color < 2; color++) {
      if (act && ((i + j) & 1) == color)
        sp[t] = T(0.25) * (sp[j*n+wr(i-1,n)] + sp[j*n+wr(i+1,n)] + sp[wr(j-1,n)*n+i] + sp[wr(j+1,n)*n+i] - h2 * ff);
      __syncthreads();
    }
  if (act) p[t] = sp[t];
}
__global__ void k_d2f(const double* a, float* b, int m) { int k = blockIdx.x * blockDim.x + threadIdx.x; if (k < m) b[k] = (float)a[k]; }
__global__ void k_addf(const float* a, double* b, int m) { int k = blockIdx.x * blockDim.x + threadIdx.x; if (k < m) b[k] += (double)a[k]; }

// ------------------------------------------------------------------ host side
static cudaStream_t S;
static const int RB = 120;  // reduction blocks
static double *d_psum, *d_pmax, *h_psum, *h_pmax;
static long long g_syncs = 0;

template <class T>
static void reduce(const T* x, int m, double& sum, double& mx) {
  k_reduce<T><<<RB, 256, 0, S>>>(x, m, d_psum, d_pmax);
  CK(cudaMemcpyAsync(h_psum, d_psum, RB * sizeof(double), cudaMemcpyDeviceToHost, S));
  CK(cudaMemcpyAsync(h_pmax, d_pmax, RB * sizeof(double), cudaMemcpyDeviceToHost, S));
  CK(cudaStreamSynchronize(S)); g_syncs++;
  sum = 0; mx = 0;
  for (int b = 0; b < RB; b++) { sum += h_psum[b]; mx = std::max(mx, h_pmax[b]); }
}

// Ablation switches. All false reproduces the milestone-1 solver.
struct Cfg { bool graph = true, coarse1 = true, warm = true, adaptive = true; };

static int cycles_for(double rtol) { return std::max(1, (int)ceil(log(rtol) / log(0.07)) - 1); }

template <class T>
struct MG {
  int L; std::vector<int> n; std::vector<T*> p, f, r; double h0; Cfg cfg;
  int pre = 2, post = 2, coarse_sweeps = 60, last = 0;
  cudaGraphExec_t gx = nullptr;
  MG(int N, double h, Cfg c) : h0(h), cfg(c) {
    for (int m = N; m >= 4; m /= 2) n.push_back(m);
    L = (int)n.size() - 1;
    for (int l = 0; l <= L; l++) {
      size_t b = (size_t)n[l] * n[l] * sizeof(T);
      T *a, *cc, *d; CK(cudaMalloc(&a, b)); CK(cudaMalloc(&cc, b)); CK(cudaMalloc(&d, b));
      CK(cudaMemset(a, 0, b));
      p.push_back(a); f.push_back(cc); r.push_back(d);
    }
  }
  double h(int l) const { return h0 * (double)(1 << l); }
  void smooth(int l, int sweeps) {
    T h2 = (T)(h(l) * h(l));
    for (int s = 0; s < sweeps; s++) {
      k_smooth<T><<<G2(n[l]), B2, 0, S>>>(p[l], f[l], n[l], h2, 0);
      k_smooth<T><<<G2(n[l]), B2, 0, S>>>(p[l], f[l], n[l], h2, 1);
    }
  }
  void vcycle(int l) {
    if (l == L) {
      if (cfg.coarse1) k_coarse<T><<<1, n[l] * n[l], 0, S>>>(p[l], f[l], n[l], (T)(h(l) * h(l)), coarse_sweeps);
      else smooth(l, coarse_sweeps);
      return;
    }
    smooth(l, pre);
    k_resid<T><<<G2(n[l]), B2, 0, S>>>(p[l], f[l], r[l], n[l], (T)(1.0 / (h(l) * h(l))));
    k_restrict<T><<<G2(n[l+1]), B2, 0, S>>>(r[l], f[l+1], n[l+1]);
    int mc = n[l+1] * n[l+1];
    k_fill<T><<<(mc + 255) / 256, 256, 0, S>>>(p[l+1], mc, T(0));
    vcycle(l + 1);
    k_prolong_add<T><<<G2(n[l]), B2, 0, S>>>(p[l+1], p[l], n[l]);
    smooth(l, post);
  }
  void cycle() {
    if (!cfg.graph) { vcycle(0); return; }
    if (!gx) {
      cudaGraph_t g;
      CK(cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal)); vcycle(0); CK(cudaStreamEndCapture(S, &g));
      CK(cudaGraphInstantiate(&gx, g, 0));
    }
    CK(cudaGraphLaunch(gx, S));
  }
  double resid_max() {
    k_resid<T><<<G2(n[0]), B2, 0, S>>>(p[0], f[0], r[0], n[0], (T)(1.0 / (h0 * h0)));
    double s, m; reduce(r[0], n[0] * n[0], s, m); return m;
  }
  // Solve L p = f[0]. Returns V-cycles used.
  int solve(double rtol, int maxc, std::vector<double>* hist = nullptr) {
    int m = n[0] * n[0];
    double s, fm; reduce(f[0], m, s, fm);
    k_shift<T><<<(m + 255) / 256, 256, 0, S>>>(f[0], m, (T)(s / m));   // compatibility: mean-zero RHS
    if (!cfg.warm || fm == 0) k_fill<T><<<(m + 255) / 256, 256, 0, S>>>(p[0], m, T(0));
    if (fm == 0) return 0;
    int c = 0; double rm;
    if (cfg.adaptive && !hist) {          // run the expected number of cycles blind, then check
      int c0 = (cfg.warm && last > 0) ? std::max(1, last - 1) : cycles_for(rtol);
      for (; c < c0; c++) cycle();
      rm = resid_max();
    } else {
      rm = resid_max(); if (hist) hist->push_back(rm / fm);
    }
    while (c < maxc && rm > rtol * fm) { cycle(); c++; rm = resid_max(); if (hist) hist->push_back(rm / fm); }
    last = c;
    return c;
  }
};

// Mixed precision: FP64 iterate and residual, one FP32 V-cycle per correction (iterative refinement).
struct Mixed {
  MG<double>& o; MG<float> in; Cfg cfg; cudaGraphExec_t gx = nullptr; int last = 0;
  Mixed(MG<double>& outer, Cfg c) : o(outer), in(outer.n[0], outer.h0, c), cfg(c) {}
  void outer_iter() {
    int N = o.n[0], m = N * N;
    k_resid<double><<<G2(N), B2, 0, S>>>(o.p[0], o.f[0], o.r[0], N, 1.0 / (o.h0 * o.h0));
    k_d2f<<<(m + 255) / 256, 256, 0, S>>>(o.r[0], in.f[0], m);
    k_fill<float><<<(m + 255) / 256, 256, 0, S>>>(in.p[0], m, 0.f);
    in.vcycle(0);
    k_addf<<<(m + 255) / 256, 256, 0, S>>>(in.p[0], o.p[0], m);
  }
  void iter() {
    if (!cfg.graph) { outer_iter(); return; }
    if (!gx) {
      cudaGraph_t g;
      CK(cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal)); outer_iter(); CK(cudaStreamEndCapture(S, &g));
      CK(cudaGraphInstantiate(&gx, g, 0));
    }
    CK(cudaGraphLaunch(gx, S));
  }
  int solve(double rtol, int maxc, std::vector<double>* hist = nullptr) {
    int m = o.n[0] * o.n[0];
    double s, fm; reduce(o.f[0], m, s, fm);
    k_shift<double><<<(m + 255) / 256, 256, 0, S>>>(o.f[0], m, s / m);
    if (!cfg.warm || fm == 0) k_fill<double><<<(m + 255) / 256, 256, 0, S>>>(o.p[0], m, 0.0);
    if (fm == 0) return 0;
    int c = 0; double rm;
    if (cfg.adaptive && !hist) {
      int c0 = (cfg.warm && last > 0) ? std::max(1, last - 1) : cycles_for(rtol);
      for (; c < c0; c++) iter();
      rm = o.resid_max();
    } else { rm = o.resid_max(); if (hist) hist->push_back(rm / fm); }
    while (c < maxc && rm > rtol * fm) { iter(); c++; rm = o.resid_max(); if (hist) hist->push_back(rm / fm); }
    last = c;
    return c;
  }
};

struct Opt {
  std::string mode = "tg", prec = "f64", dump; int N = 256, steps = 50;
  double nu = 0.01, T = 1.0, cfl = 0.5, tol = -1; Cfg cfg;
};

template <class T>
struct Solver {
  int n, m; double h, nu; T *u, *v, *u1, *v1, *ru, *rv; MG<T> mg; Mixed* mix = nullptr; double tol;
  long long mg_cycles = 0, solves = 0;
  float t_rhs = 0, t_stage = 0, t_proj = 0; bool prof = false; cudaEvent_t e0, e1;
  Solver(int N, double nu_, double tol_, Cfg c) : n(N), m(N * N), h(2 * PI / N), nu(nu_), mg(N, 2 * PI / N, c), tol(tol_) {
    size_t b = (size_t)m * sizeof(T);
    for (T** q : {&u, &v, &u1, &v1, &ru, &rv}) CK(cudaMalloc(q, b));
    cudaEventCreate(&e0); cudaEventCreate(&e1);
  }
  void tic() { if (prof) cudaEventRecord(e0, S); }
  void toc(float& acc) { if (prof) { cudaEventRecord(e1, S); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); acc += ms; } }
  void set_taylor_green(double t) {
    std::vector<T> hu(m), hv(m); double dec = exp(-2 * nu * t);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
      hu[j*n+i] = (T)(sin(i * h) * cos((j + 0.5) * h) * dec); hv[j*n+i] = (T)(-cos((i + 0.5) * h) * sin(j * h) * dec);
    }
    CK(cudaMemcpy(u, hu.data(), m * sizeof(T), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(v, hv.data(), m * sizeof(T), cudaMemcpyHostToDevice));
  }
  void project(T* uu, T* vv) {
    tic();
    k_div<T><<<G2(n), B2, 0, S>>>(uu, vv, mg.f[0], n, (T)(1.0 / h));
    mg_cycles += mix ? mix->solve(tol, 40) : mg.solve(tol, 40); solves++;
    k_correct<T><<<G2(n), B2, 0, S>>>(uu, vv, mg.p[0], n, (T)(1.0 / h));
    toc(t_proj);
  }
  void rhs(const T* uu, const T* vv) {
    tic(); k_rhs<T><<<G2(n), B2, 0, S>>>(uu, vv, ru, rv, n, (T)(1.0 / h), (T)(nu / (h * h))); toc(t_rhs);
  }
  void stage(double a, double b, double dt, const T* uk, const T* vk, T* uo, T* vo) {
    tic(); k_stage<T><<<(m + 255) / 256, 256, 0, S>>>(m, (T)a, (T)b, (T)dt, u, uk, ru, uo, v, vk, rv, vo); toc(t_stage);
  }
  void step(double dt) {  // SSP-RK3 (Shu-Osher), projection after each stage
    rhs(u, v);   stage(0.0, 1.0, dt, u, v, u1, v1); project(u1, v1);
    rhs(u1, v1); stage(0.75, 0.25, dt, u1, v1, u1, v1); project(u1, v1);
    rhs(u1, v1); stage(1.0 / 3, 2.0 / 3, dt, u1, v1, u, v); project(u, v);
  }
  void errors(double t, double& linf, double& l2) {
    std::vector<T> hu(m), hv(m);
    CK(cudaMemcpy(hu.data(), u, m * sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hv.data(), v, m * sizeof(T), cudaMemcpyDeviceToHost));
    double dec = exp(-2 * nu * t); linf = 0; l2 = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
      double eu = hu[j*n+i] - sin(i * h) * cos((j + 0.5) * h) * dec;
      double ev = hv[j*n+i] + cos((i + 0.5) * h) * sin(j * h) * dec;
      linf = std::max(linf, std::max(fabs(eu), fabs(ev))); l2 += eu * eu + ev * ev;
    }
    l2 = sqrt(l2 / (2.0 * m));
  }
  double max_div() { k_div<T><<<G2(n), B2, 0, S>>>(u, v, mg.r[0], n, (T)(1.0 / h)); double s, mx; reduce(mg.r[0], m, s, mx); return mx; }
  void dump(const std::string& fn) {
    std::vector<double> out(2 * m); std::vector<T> hu(m), hv(m);
    CK(cudaMemcpy(hu.data(), u, m * sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hv.data(), v, m * sizeof(T), cudaMemcpyDeviceToHost));
    for (int k = 0; k < m; k++) { out[k] = hu[k]; out[m + k] = hv[k]; }
    FILE* fp = fopen(fn.c_str(), "wb"); fwrite(out.data(), sizeof(double), 2 * m, fp); fclose(fp);
  }
};

static void cfg_json(const Cfg& c) { printf("\"cfg\":{\"graph\":%d,\"coarse1\":%d,\"warm\":%d,\"adaptive\":%d}", c.graph, c.coarse1, c.warm, c.adaptive); }

template <class T>
static void run(const Opt& o, bool mixed) {
  const char* P = mixed ? "mixed" : (sizeof(T) == 8 ? "f64" : "f32");
  double tol = o.tol > 0 ? o.tol : (sizeof(T) == 8 ? 1e-10 : 1e-5);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  if (o.mode == "mg") {
    Cfg c = o.cfg; c.warm = false;
    MG<T> mg(o.N, 2 * PI / o.N, c); Mixed* mx = nullptr;
    if constexpr (sizeof(T) == 8) { if (mixed) mx = new Mixed(mg, c); }
    std::vector<T> hf((size_t)o.N * o.N); std::mt19937 g(1); std::normal_distribution<double> nd;
    for (auto& x : hf) x = (T)nd(g);
    CK(cudaMemcpy(mg.f[0], hf.data(), hf.size() * sizeof(T), cudaMemcpyHostToDevice));
    std::vector<double> hist;
    int cyc = mx ? mx->solve(1e-30, 16, &hist) : mg.solve(1e-30, 16, &hist);
    for (int k = 0; k < 3; k++) { if (mx) mx->iter(); else mg.cycle(); }
    CK(cudaStreamSynchronize(S));
    cudaEventRecord(a, S); for (int k = 0; k < 50; k++) { if (mx) mx->iter(); else mg.cycle(); } cudaEventRecord(b, S);
    cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);
    printf("{\"mode\":\"mg\",\"prec\":\"%s\",\"N\":%d,\"levels\":%d,\"cycles\":%d,\"ms_per_cycle\":%.5f,", P, o.N, mg.L + 1, cyc, ms / 50);
    cfg_json(o.cfg); printf(",\"rel_resid\":[");
    for (size_t k = 0; k < hist.size(); k++) printf("%s%.6e", k ? "," : "", hist[k]);
    printf("]}\n");
    return;
  }
  Solver<T> s(o.N, o.nu, tol, o.cfg);
  if constexpr (sizeof(T) == 8) { if (mixed) s.mix = new Mixed(s.mg, o.cfg); }
  s.set_taylor_green(0.0);
  s.project(s.u, s.v);
  double dt = std::min(o.cfl * s.h, 0.2 * s.h * s.h / o.nu);
  if (o.mode == "tg") {
    int nsteps = (int)ceil(o.T / dt); dt = o.T / nsteps;
    s.mg_cycles = 0; s.solves = 0;
    cudaEventRecord(a, S);
    for (int k = 0; k < nsteps; k++) s.step(dt);
    cudaEventRecord(b, S); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);
    double linf, l2; s.errors(o.T, linf, l2);
    if (!o.dump.empty()) s.dump(o.dump);
    printf("{\"mode\":\"tg\",\"prec\":\"%s\",\"N\":%d,\"nu\":%g,\"T\":%g,\"dt\":%.6e,\"steps\":%d,\"linf\":%.6e,\"l2\":%.6e,"
           "\"max_div\":%.3e,\"mg_cycles_per_solve\":%.3f,\"wall_ms\":%.2f,",
           P, o.N, o.nu, o.T, dt, nsteps, linf, l2, s.max_div(), (double)s.mg_cycles / s.solves, ms);
    cfg_json(o.cfg); printf("}\n");
    return;
  }
  if (o.mode == "bench") {
    for (int k = 0; k < 5; k++) s.step(dt);
    CK(cudaStreamSynchronize(S));
    s.mg_cycles = 0; s.solves = 0; long long sy0 = g_syncs;
    std::vector<float> per;
    for (int rep = 0; rep < 5; rep++) {
      cudaEventRecord(a, S);
      for (int k = 0; k < o.steps; k++) s.step(dt);
      cudaEventRecord(b, S); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b); per.push_back(ms / o.steps);
    }
    double syncs = (double)(g_syncs - sy0) / (5.0 * o.steps);
    std::sort(per.begin(), per.end());
    double cyc = (double)s.mg_cycles / s.solves;
    s.prof = true; for (int k = 0; k < o.steps; k++) s.step(dt); s.prof = false;
    double rhs_ms = s.t_rhs / (3.0 * o.steps);
    printf("{\"mode\":\"bench\",\"prec\":\"%s\",\"N\":%d,\"ms_per_step_median\":%.4f,\"ms_per_step_min\":%.4f,\"ms_per_step_max\":%.4f,"
           "\"mg_cycles_per_solve\":%.3f,\"host_syncs_per_step\":%.2f,"
           "\"phase_ms_per_step\":{\"rhs\":%.4f,\"stage\":%.4f,\"projection\":%.4f},\"rhs_GBps\":%.2f,\"cells_per_s\":%.4e,",
           P, o.N, per[2], per[0], per[4], cyc, syncs, s.t_rhs / o.steps, s.t_stage / o.steps, s.t_proj / o.steps,
           4.0 * s.m * sizeof(T) / (rhs_ms * 1e6), (double)s.m / (per[2] * 1e-3));
    cfg_json(o.cfg); printf("}\n");
  }
}

int main(int argc, char** argv) {
  Opt o; if (argc > 1) o.mode = argv[1];
  for (int k = 2; k + 1 < argc; k += 2) {
    std::string a = argv[k], b = argv[k + 1];
    if (a == "--N") o.N = atoi(b.c_str()); else if (a == "--prec") o.prec = b; else if (a == "--nu") o.nu = atof(b.c_str());
    else if (a == "--T") o.T = atof(b.c_str()); else if (a == "--cfl") o.cfl = atof(b.c_str()); else if (a == "--tol") o.tol = atof(b.c_str());
    else if (a == "--steps") o.steps = atoi(b.c_str()); else if (a == "--dump") o.dump = b;
    else if (a == "--graph") o.cfg.graph = atoi(b.c_str()); else if (a == "--coarse1") o.cfg.coarse1 = atoi(b.c_str());
    else if (a == "--warm") o.cfg.warm = atoi(b.c_str()); else if (a == "--adaptive") o.cfg.adaptive = atoi(b.c_str());
    else if (a == "--baseline" && atoi(b.c_str())) o.cfg = Cfg{false, false, false, false};
  }
  CK(cudaStreamCreate(&S));
  CK(cudaMalloc(&d_psum, RB * sizeof(double))); CK(cudaMalloc(&d_pmax, RB * sizeof(double)));
  CK(cudaMallocHost(&h_psum, RB * sizeof(double))); CK(cudaMallocHost(&h_pmax, RB * sizeof(double)));
  if (o.mode == "bw") {
    size_t bytes = (size_t)256 << 20; float4 *a, *b; CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes));
    size_t m = bytes / sizeof(float4); k_copy4<<<20 * 32, 256, 0, S>>>(a, b, m); CK(cudaStreamSynchronize(S));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<double> gbs;
    for (int r = 0; r < 20; r++) { cudaEventRecord(e0, S); k_copy4<<<20 * 32, 256, 0, S>>>(a, b, m); cudaEventRecord(e1, S);
      cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); gbs.push_back(2.0 * bytes / (ms * 1e6)); }
    std::sort(gbs.begin(), gbs.end());
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("{\"mode\":\"bw\",\"device\":\"%s\",\"sm\":\"%d.%d\",\"sms\":%d,\"bus_bits\":%d,\"copy_GBps_median\":%.2f,\"copy_GBps_max\":%.2f}\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.memoryBusWidth, gbs[10], gbs.back());
    return 0;
  }
  if (o.prec == "f32") run<float>(o, false);
  else run<double>(o, o.prec == "mixed");
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  return 0;
}
