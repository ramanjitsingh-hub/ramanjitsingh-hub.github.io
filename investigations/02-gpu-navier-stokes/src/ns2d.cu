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
//   bench   time per step with a per-phase breakdown
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

// ------------------------------------------------------------------ host side
static const int RB = 120;  // reduction blocks
static double *d_psum, *d_pmax, h_psum[RB], h_pmax[RB];

template <class T>
static void reduce(const T* x, int m, double& sum, double& mx) {
  k_reduce<T><<<RB, 256>>>(x, m, d_psum, d_pmax);
  CK(cudaMemcpy(h_psum, d_psum, RB * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_pmax, d_pmax, RB * sizeof(double), cudaMemcpyDeviceToHost));
  sum = 0; mx = 0;
  for (int b = 0; b < RB; b++) { sum += h_psum[b]; mx = std::max(mx, h_pmax[b]); }
}

template <class T>
struct MG {
  int L; std::vector<int> n; std::vector<T*> p, f, r; double h0;
  int pre = 2, post = 2, coarse_sweeps = 60;
  MG(int N, double h) : h0(h) {
    for (int m = N; m >= 4; m /= 2) n.push_back(m);
    L = (int)n.size() - 1;
    for (int l = 0; l <= L; l++) {
      size_t b = (size_t)n[l] * n[l] * sizeof(T);
      T *a, *c, *d; CK(cudaMalloc(&a, b)); CK(cudaMalloc(&c, b)); CK(cudaMalloc(&d, b));
      p.push_back(a); f.push_back(c); r.push_back(d);
    }
  }
  double h(int l) const { return h0 * (double)(1 << l); }
  void smooth(int l, int sweeps) {
    T h2 = (T)(h(l) * h(l));
    for (int s = 0; s < sweeps; s++) {
      k_smooth<T><<<G2(n[l]), B2>>>(p[l], f[l], n[l], h2, 0);
      k_smooth<T><<<G2(n[l]), B2>>>(p[l], f[l], n[l], h2, 1);
    }
  }
  void vcycle(int l) {
    if (l == L) { smooth(l, coarse_sweeps); return; }
    smooth(l, pre);
    k_resid<T><<<G2(n[l]), B2>>>(p[l], f[l], r[l], n[l], (T)(1.0 / (h(l) * h(l))));
    k_restrict<T><<<G2(n[l+1]), B2>>>(r[l], f[l+1], n[l+1]);
    int mc = n[l+1] * n[l+1];
    k_fill<T><<<(mc + 255) / 256, 256>>>(p[l+1], mc, T(0));
    vcycle(l + 1);
    k_prolong_add<T><<<G2(n[l]), B2>>>(p[l+1], p[l], n[l]);
    smooth(l, post);
  }
  double resid_max() {
    k_resid<T><<<G2(n[0]), B2>>>(p[0], f[0], r[0], n[0], (T)(1.0 / (h0 * h0)));
    double s, m; reduce(r[0], n[0] * n[0], s, m); return m;
  }
  // solve L p = f[0] (f must already hold the RHS). Returns cycles used.
  int solve(double rtol, int maxc, std::vector<double>* hist = nullptr) {
    int m = n[0] * n[0];
    double s, fm; reduce(f[0], m, s, fm);
    k_shift<T><<<(m + 255) / 256, 256>>>(f[0], m, (T)(s / m));   // compatibility: mean-zero RHS
    if (fm == 0) { k_fill<T><<<(m + 255) / 256, 256>>>(p[0], m, T(0)); return 0; }
    k_fill<T><<<(m + 255) / 256, 256>>>(p[0], m, T(0));
    int c = 0; double rm = resid_max();
    if (hist) hist->push_back(rm / fm);
    while (c < maxc && rm > rtol * fm) { vcycle(0); c++; rm = resid_max(); if (hist) hist->push_back(rm / fm); }
    return c;
  }
};

struct Opt { std::string mode = "tg", prec = "f64", dump; int N = 256, steps = 50; double nu = 0.01, T = 1.0, cfl = 0.5, tol = -1; };


template <class T>
struct Solver {
  int n, m; double h, nu; T *u, *v, *u0, *v0, *u1, *v1, *ru, *rv; MG<T> mg; double tol;
  long long mg_cycles = 0, solves = 0;
  float t_rhs = 0, t_stage = 0, t_proj = 0; bool prof = false; cudaEvent_t e0, e1;
  Solver(int N, double nu_, double tol_) : n(N), m(N * N), h(2 * PI / N), nu(nu_), mg(N, 2 * PI / N), tol(tol_) {
    size_t b = (size_t)m * sizeof(T);
    for (T** q : {&u, &v, &u0, &v0, &u1, &v1, &ru, &rv}) CK(cudaMalloc(q, b));
    cudaEventCreate(&e0); cudaEventCreate(&e1);
  }
  void tic() { if (prof) cudaEventRecord(e0); }
  void toc(float& acc) { if (prof) { cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); acc += ms; } }
  void set_taylor_green(double t) {
    std::vector<T> hu(m), hv(m); double dec = exp(-2 * nu * t);
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
      double xu = i * h, yu = (j + 0.5) * h, xv = (i + 0.5) * h, yv = j * h;
      hu[j*n+i] = (T)(sin(xu) * cos(yu) * dec); hv[j*n+i] = (T)(-cos(xv) * sin(yv) * dec);
    }
    CK(cudaMemcpy(u, hu.data(), m * sizeof(T), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(v, hv.data(), m * sizeof(T), cudaMemcpyHostToDevice));
  }
  void project(T* uu, T* vv) {
    tic();
    k_div<T><<<G2(n), B2>>>(uu, vv, mg.f[0], n, (T)(1.0 / h));
    mg_cycles += mg.solve(tol, 40); solves++;
    k_correct<T><<<G2(n), B2>>>(uu, vv, mg.p[0], n, (T)(1.0 / h));
    toc(t_proj);
  }
  void rhs(const T* uu, const T* vv) {
    tic(); k_rhs<T><<<G2(n), B2>>>(uu, vv, ru, rv, n, (T)(1.0 / h), (T)(nu / (h * h))); toc(t_rhs);
  }
  void stage(double a, double b, double dt, const T* uk, const T* vk, T* uo, T* vo) {
    tic(); k_stage<T><<<(m + 255) / 256, 256>>>(m, (T)a, (T)b, (T)dt, u, uk, ru, uo, v, vk, rv, vo); toc(t_stage);
  }
  void step(double dt) {  // SSP-RK3 (Shu-Osher), projection after each stage
    rhs(u, v);   stage(0.0, 1.0, dt, u, v, u1, v1); project(u1, v1);         // u1 = u + dt L(u)
    rhs(u1, v1); stage(0.75, 0.25, dt, u1, v1, u1, v1); project(u1, v1);  // u2
    rhs(u1, v1); stage(1.0 / 3, 2.0 / 3, dt, u1, v1, u, v); project(u, v);   // u3
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
  double max_div() { k_div<T><<<G2(n), B2>>>(u, v, mg.r[0], n, (T)(1.0 / h)); double s, mx; reduce(mg.r[0], m, s, mx); return mx; }
  void dump(const std::string& fn) {
    std::vector<double> out(2 * m); std::vector<T> hu(m), hv(m);
    CK(cudaMemcpy(hu.data(), u, m * sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hv.data(), v, m * sizeof(T), cudaMemcpyDeviceToHost));
    for (int k = 0; k < m; k++) { out[k] = hu[k]; out[m + k] = hv[k]; }
    FILE* fp = fopen(fn.c_str(), "wb"); fwrite(out.data(), sizeof(double), 2 * m, fp); fclose(fp);
  }
};

template <class T>
static void run(const Opt& o) {
  const char* P = sizeof(T) == 8 ? "f64" : "f32";
  double tol = o.tol > 0 ? o.tol : (sizeof(T) == 8 ? 1e-10 : 1e-5);
  if (o.mode == "mg") {
    MG<T> mg(o.N, 2 * PI / o.N);
    std::vector<T> hf((size_t)o.N * o.N); std::mt19937 g(1); std::normal_distribution<double> nd;
    for (auto& x : hf) x = (T)nd(g);
    CK(cudaMemcpy(mg.f[0], hf.data(), hf.size() * sizeof(T), cudaMemcpyHostToDevice));
    std::vector<double> hist;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); int c = mg.solve(1e-30, 14, &hist); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    // time pure V-cycles (no residual readback)
    cudaEventRecord(a); for (int k = 0; k < 20; k++) mg.vcycle(0); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms20; cudaEventElapsedTime(&ms20, a, b);
    printf("{\"mode\":\"mg\",\"prec\":\"%s\",\"N\":%d,\"levels\":%d,\"cycles\":%d,\"ms_per_vcycle\":%.5f,\"rel_resid\":[", P, o.N, mg.L + 1, c, ms20 / 20);
    for (size_t k = 0; k < hist.size(); k++) printf("%s%.6e", k ? "," : "", hist[k]);
    printf("]}\n");
    return;
  }
  Solver<T> s(o.N, o.nu, tol);
  s.set_taylor_green(0.0);
  s.project(s.u, s.v);            // discrete-divergence-free initial condition
  double dt = std::min(o.cfl * s.h, 0.2 * s.h * s.h / o.nu);
  if (o.mode == "tg") {
    int nsteps = (int)ceil(o.T / dt); dt = o.T / nsteps;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for (int k = 0; k < nsteps; k++) s.step(dt);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    double linf, l2; s.errors(o.T, linf, l2);
    if (!o.dump.empty()) s.dump(o.dump);
    printf("{\"mode\":\"tg\",\"prec\":\"%s\",\"N\":%d,\"nu\":%g,\"T\":%g,\"dt\":%.6e,\"steps\":%d,\"linf\":%.6e,\"l2\":%.6e,"
           "\"max_div\":%.3e,\"mg_cycles_per_solve\":%.3f,\"wall_ms\":%.2f}\n",
           P, o.N, o.nu, o.T, dt, nsteps, linf, l2, s.max_div(), (double)s.mg_cycles / s.solves, ms);
    return;
  }
  if (o.mode == "bench") {
    for (int k = 0; k < 5; k++) s.step(dt);    // warm-up
    CK(cudaDeviceSynchronize());
    s.mg_cycles = 0; s.solves = 0;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for (int k = 0; k < o.steps; k++) s.step(dt);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    double cyc = (double)s.mg_cycles / s.solves;
    s.prof = true; for (int k = 0; k < o.steps; k++) s.step(dt); s.prof = false;
    double bytes_rhs = 4.0 * s.m * sizeof(T);   // ideal traffic: read u,v, write ru,rv
    double rhs_ms = s.t_rhs / (3.0 * o.steps);
    printf("{\"mode\":\"bench\",\"prec\":\"%s\",\"N\":%d,\"ms_per_step\":%.4f,\"mg_cycles_per_solve\":%.3f,"
           "\"phase_ms_per_step\":{\"rhs\":%.4f,\"stage\":%.4f,\"projection\":%.4f},\"rhs_GBps\":%.2f,"
           "\"cells_per_s\":%.4e}\n",
           P, o.N, ms / o.steps, cyc, s.t_rhs / o.steps, s.t_stage / o.steps, s.t_proj / o.steps,
           bytes_rhs / (rhs_ms * 1e6), (double)s.m * o.steps / (ms * 1e-3));
  }
}

int main(int argc, char** argv) {
  Opt o; if (argc > 1) o.mode = argv[1];
  for (int k = 2; k + 1 < argc; k += 2) {
    std::string a = argv[k], b = argv[k + 1];
    if (a == "--N") o.N = atoi(b.c_str()); else if (a == "--prec") o.prec = b; else if (a == "--nu") o.nu = atof(b.c_str());
    else if (a == "--T") o.T = atof(b.c_str()); else if (a == "--cfl") o.cfl = atof(b.c_str()); else if (a == "--tol") o.tol = atof(b.c_str());
    else if (a == "--steps") o.steps = atoi(b.c_str()); else if (a == "--dump") o.dump = b;
  }
  CK(cudaMalloc(&d_psum, RB * sizeof(double))); CK(cudaMalloc(&d_pmax, RB * sizeof(double)));
  if (o.mode == "bw") {
    size_t bytes = (size_t)256 << 20; float4 *a, *b; CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes));
    size_t m = bytes / sizeof(float4); k_copy4<<<20 * 32, 256>>>(a, b, m); CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<double> gbs;
    for (int r = 0; r < 20; r++) { cudaEventRecord(e0); k_copy4<<<20 * 32, 256>>>(a, b, m); cudaEventRecord(e1);
      cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); gbs.push_back(2.0 * bytes / (ms * 1e6)); }
    std::sort(gbs.begin(), gbs.end());
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("{\"mode\":\"bw\",\"device\":\"%s\",\"sm\":\"%d.%d\",\"sms\":%d,\"bus_bits\":%d,\"copy_GBps_median\":%.2f,\"copy_GBps_max\":%.2f}\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.memoryBusWidth, gbs[10], gbs.back());
    return 0;
  }
  if (o.prec == "f32") run<float>(o); else run<double>(o);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  return 0;
}
